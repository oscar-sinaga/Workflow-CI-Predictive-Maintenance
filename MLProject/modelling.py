import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from sklearn.metrics import precision_recall_curve, log_loss
from mlflow.models.signature import infer_signature 
import shap 

# 1. KONFIGURASI DAGSHUB (MLFLOW ONLINE)

# Tentukan path secara dinamis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

dagshub_uri = "https://dagshub.com/oscar-sinaga/Predictive-Maintenance-MLOps.mlflow"
mlflow.set_tracking_uri(dagshub_uri)
mlflow.set_experiment("Advance_Predictive_Maintenance")

if __name__ == "__main__":

    # 2. DATA LOADING & SPLITTING
    df = pd.read_csv(os.path.join(BASE_DIR, "predictive_maintenance_preprocessing", "predictive_maintenance_processed.csv"))
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. HYPERPARAMETER TUNING (Optimasi F1-Score)
    print("Memulai Hyperparameter Tuning untuk F1-Score...")

    # class_weight='balanced' dipertahankan untuk imbalanced data
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # 4. MANUAL LOGGING MLFLOW (TANPA AUTOLOG)
    # Menggunakan nested=True agar kompatibel saat dijalankan via 'mlflow run'
    with mlflow.start_run(nested=True):
        print("Logging metrik dan artefak ke DagsHub...")

        # Set Tag
        mlflow.set_tag("developer", "oscar-sinaga")
        mlflow.set_tag("model_type", "RandomForest_Tuned")
        
        # A. Log Parameter (Manual)
        mlflow.log_params(grid_search.best_params_)
        
        # B. Prediksi dan Evaluasi Metrik
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
        
        # 5. PEMBUATAN ARTEFAK VISUAL & TEKS (4 ARTEFAK)
        os.makedirs("artifacts", exist_ok=True)
        
        # Artefak 1: Confusion Matrix
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
        
        # Artefak 2: Feature Importance
        plt.figure(figsize=(8,6))
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10) 
        sns.barplot(x=feature_importance, y=feature_importance.index, hue=feature_importance.index, palette='viridis', legend=False)
        plt.title('Top 10 Feature Importance')
        fi_path = "artifacts/feature_importance.png"
        plt.savefig(fi_path)
        plt.tight_layout()
        mlflow.log_artifact(fi_path)
        plt.close()

        # Artefak 3: Kurva ROC (Permintaan Baru)
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

        # Artefak 4: Classification Report (Teks)
        report = classification_report(y_test, y_pred)
        report_path = "artifacts/classification_report.txt"
        with open(report_path, "w") as f:
            f.write("Classification Report\n")
            f.write("=====================\n\n")
            f.write(report)
        mlflow.log_artifact(report_path)

        # Artefak 5: Precision-Recall Curve
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

        # Artefak 5: SHAP
        print("Menghitung SHAP Values (Proses ini mungkin memakan waktu beberapa menit)...")
        
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        
        # Penanganan khusus untuk kompabilitas berbagai versi library SHAP
        if isinstance(shap_values, list):
            # Untuk SHAP versi lama
            shap_values_target = shap_values[1]
        else:
            # Untuk SHAP versi terbaru (Array 3D)
            shap_values_target = shap_values[:, :, 1]
        
        plt.figure() 
        shap.summary_plot(shap_values_target, X_test, feature_names=X.columns, show=False)
        
        shap_path = "artifacts/shap_summary_plot.png"
        plt.savefig(shap_path, bbox_inches='tight') 
        mlflow.log_artifact(shap_path)
        plt.close()

        # Model Signature & Input Example
        # Infer signature secara otomatis menebak skema (kolom dan tipe data) dari X_train dan y_train
        signature = infer_signature(X_train, y_train)
        
        # Ambil 1 baris contoh dari data test untuk disimpan bersama model
        input_example = X_test.iloc[[0]] 
        
        # Simpan Model Utama dengan Signature
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="random_forest_model",
            signature=signature,           # Menambahkan Skema
            input_example=input_example    # Menambahkan Contoh Data
        )
        
        print("✅ Eksperimen sukses! Kurva ROC dan Report Text berhasil dikirim ke DagsHub.")