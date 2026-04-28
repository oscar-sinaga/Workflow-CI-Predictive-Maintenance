# 1. Menggunakan base image Python 3.12-slim (ringan dan sesuai versi Anda)
FROM python:3.12-slim

# 2. Menentukan direktori kerja di dalam kontainer
WORKDIR /app

# 3. Instalasi dependencies sistem yang mungkin dibutuhkan (build-essential untuk library seperti SHAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Melakukan upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 5. Instalasi library sesuai dengan versi presisi di conda.yaml Anda
# Kita langsung pasang versinya di sini agar proses build Docker lebih cepat
RUN pip install --no-cache-dir \
    dagshub==0.7.0 \
    ipykernel==7.2.0 \
    kaggle==2.0.1 \
    mlflow==2.19.0 \
    python-dotenv==1.2.2 \
    seaborn==0.13.2 \
    unzip==1.0.0 \
    pandas==2.3.3 \
    scikit-learn==1.8.0 \
    matplotlib==3.10.8 \
    shap==0.51.0

# 6. Copy seluruh file proyek (modelling.py, data, conda.yaml, dll) ke dalam kontainer
COPY . /app

# 7. Perintah default saat kontainer dijalankan
# Ini akan menjalankan proses training model secara otomatis
CMD ["python", "modelling.py"]