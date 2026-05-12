import pandas as pd
import numpy as np         
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib              
import mlflow
import mlflow.sklearn
import mlflow.pyfunc       
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from sklearn.metrics import precision_recall_curve, log_loss
from mlflow.models.signature import infer_signature 
import shap 
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


class SmartPredictiveMaintenance(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Me-load model dan scaler saat Docker API dinyalakan"""
        import joblib
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        """Fungsi ini berjalan setiap kali API menerima request data mentah"""
        import numpy as np
        import pandas as pd
        
        df_input = model_input.copy()
        print(f"DEBUG: Input mentah diterima: \n{df_input.head(1)}")

        # 1. FEATURE ENGINEERING
        if 'Process temperature [K]' in df_input.columns and 'Air temperature [K]' in df_input.columns:
            df_input['Temp_Difference'] = df_input['Process temperature [K]'] - df_input['Air temperature [K]']
        if 'Torque [Nm]' in df_input.columns and 'Rotational speed [rpm]' in df_input.columns:
            df_input['Power'] = df_input['Torque [Nm]'] * df_input['Rotational speed [rpm]']
        if 'Tool wear [min]' in df_input.columns and 'Rotational speed [rpm]' in df_input.columns:
            df_input['Strain'] = df_input['Tool wear [min]'] * df_input['Rotational speed [rpm]']

        # 2. FEATURE SELECTION (Hapus kolom tidak relevan & suhu asli)
        cols_to_drop = ['Air temperature [K]', 'Process temperature [K]', 'UDI', 'Product ID', 'Failure Type']
        df_input = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns], errors='ignore')

        # 3. ENCODING KATEGORIKAL
        type_mapping = {'L': 0, 'M': 1, 'H': 2}
        if 'Type' in df_input.columns:
            df_input['Type'] = df_input['Type'].replace(type_mapping)

        # 4. TRANSFORMASI LOGARITMIK
        skewed_cols = ['Rotational speed [rpm]', 'Tool wear [min]', 'Power', 'Strain']
        for col in skewed_cols:
            if col in df_input.columns:
                df_input[col] = np.log1p(df_input[col])

        # 5. SUSUN URUTAN KOLOM (Sangat penting agar Scaler tidak salah posisi)
        expected_cols = ['Type', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Temp_Difference', 'Power', 'Strain']
        df_input = df_input[expected_cols]

        # 6. SCALING
        scaled_input = self.scaler.transform(df_input)
        
        # 7. PREDIKSI
        return self.model.predict(scaled_input)

# 1. KONFIGURASI DAGSHUB (MLFLOW ONLINE)
# Path Dinamis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

dagshub_uri = "https://dagshub.com/oscar-sinaga/Predictive-Maintenance-MLOps.mlflow"
mlflow.set_tracking_uri(dagshub_uri)
mlflow.set_experiment("Advance_Predictive_Maintenance")

if __name__ == "__main__":

    # 2. DATA LOADING & SPLITTING
    df_train = pd.read_csv(os.path.join(BASE_DIR, "predictive_maintenance_preprocessing", "predictive_maintenance_train_processed.csv"))
    df_test = pd.read_csv(os.path.join(BASE_DIR, "predictive_maintenance_preprocessing", "predictive_maintenance_test_processed.csv"))
    
    X_train = df_train.drop('Target', axis=1)
    y_train = df_train['Target']
    X_test = df_test.drop('Target', axis=1)
    y_test = df_test['Target']

    # 3. HYPERPARAMETER TUNING (Optimasi F1-Score)
    print("Memulai Hyperparameter Tuning untuk F1-Score...")

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # 4. MANUAL LOGGING MLFLOW
    with mlflow.start_run(nested=True):
        print("Logging metrik dan artefak ke DagsHub...")

        mlflow.set_tag("developer", "oscar-sinaga")
        mlflow.set_tag("model_type", "Custom_PyFunc_RandomForest")
        
        mlflow.log_params(grid_search.best_params_)
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "log_loss": log_loss(y_test, y_pred_proba)
        }
        mlflow.log_metrics(metrics)
        print("Metrik berhasil dicatat!")
        
        # 5. PEMBUATAN ARTEFAK VISUAL & TEKS
        os.makedirs("artifacts", exist_ok=True)
        
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        plt.figure(figsize=(8,6))
        feature_importance = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10) 
        sns.barplot(x=feature_importance, y=feature_importance.index, hue=feature_importance.index, palette='viridis', legend=False)
        plt.title('Top 10 Feature Importance')
        fi_path = "artifacts/feature_importance.png"
        plt.savefig(fi_path)
        plt.tight_layout()
        mlflow.log_artifact(fi_path)
        plt.close()

        plt.figure(figsize=(6,6))
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        roc_path = "artifacts/roc_curve.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

        report = classification_report(y_test, y_pred)
        report_path = "artifacts/classification_report.txt"
        with open(report_path, "w") as f:
            f.write("Classification Report\n")
            f.write("=====================\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)

        plt.figure(figsize=(6,6))
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall_vals, precision_vals, color='purple', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        pr_path = "artifacts/pr_curve.png"
        plt.savefig(pr_path)
        mlflow.log_artifact(pr_path)
        plt.close()

        print("Menghitung SHAP Values...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values_target = shap_values[1]
        else:
            shap_values_target = shap_values[:, :, 1]
        
        plt.figure() 
        shap.summary_plot(shap_values_target, X_test, feature_names=X_test.columns, show=False)
        shap_path = "artifacts/shap_summary_plot.png"
        plt.savefig(shap_path, bbox_inches='tight') 
        mlflow.log_artifact(shap_path)
        plt.close()

        # 6. LOGGING CUSTOM PYFUNC MODEL KE MLFLOW
        print("Menyiapkan Custom PyFunc Model...")

        # A. Simpan model Random Forest sementara ke disk untuk dijadikan artefak
        temp_model_path = "artifacts/temp_rf_model.pkl"
        joblib.dump(best_model, temp_model_path)
        
        # B. Tentukan lokasi Scaler yang sudah disave saat preprocessing
        scaler_path = os.path.join(BASE_DIR, "predictive_maintenance_preprocessing", "scaler.pkl")
        
        # C. Bungkus Scaler dan Model menjadi satu kamus (dictionary)
        artifacts = {
            "scaler": scaler_path,
            "model": temp_model_path
        }

        # D. DEFINISIKAN SKEMA SECARA EKSPLISIT (EXPLICIT SIGNATURE)
        input_schema = Schema([
            ColSpec("string", "Type"), 
            ColSpec("double", "Air temperature [K]"),
            ColSpec("double", "Process temperature [K]"),
            ColSpec("double", "Rotational speed [rpm]"),
            ColSpec("double", "Torque [Nm]"),
            ColSpec("double", "Tool wear [min]")
        ])
        
        output_schema = Schema([ColSpec("integer")]) # Prediksinya berupa 0 atau 1
        
        # Gabungkan menjadi satu Signature mutlak
        explicit_signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Contoh data mentah (hanya untuk preview di DagsHub UI)
        raw_cols = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        X_raw_example = pd.DataFrame([['L', 298.1, 308.6, 1551.0, 42.8, 0.0]], columns=raw_cols)

        # E. Simpan Model menggunakan mlflow.pyfunc
        mlflow.pyfunc.log_model(
            artifact_path="random_forest_model",
            python_model=SmartPredictiveMaintenance(),
            artifacts=artifacts,
            signature=explicit_signature,         
            input_example=X_raw_example,    
        )
        
        print("✅ Eksperimen sukses! Custom Model (Praproses[Scaler + LogTransform] + RF) berhasil dikirim ke DagsHub.")