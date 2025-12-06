Titanic Survival Prediction - MLOps Pipeline

Proyek ini adalah implementasi End-to-End MLOps untuk memprediksi keselamatan penumpang Titanic. Proyek ini mencakup Experiment Tracking, Model Registry, dan Real-time Monitoring.

🛠️ Tech Stack

Machine Learning: Scikit-learn (Random Forest)

Tracking & Registry: MLflow & DagsHub

Monitoring: Prometheus & Grafana

Version Control: Git & DagsHub

📂 Struktur Project

├── MLproject           # Entry point MLflow
├── conda.yaml          # Definisi Environment
├── modelling.py        # Script Training & Monitoring
├── prometheus.yml      # Config Prometheus
├── titanic_clean.csv   # Dataset
└── requirements.txt    # Dependencies


🚀 Cara Menjalankan

1. Setup Environment

Pastikan Python 3.9+ dan library terinstall:

pip install mlflow dagshub pandas scikit-learn matplotlib seaborn prometheus-client


2. Setup Credentials

Set environment variables untuk autentikasi DagsHub:

MLFLOW_TRACKING_USERNAME

MLFLOW_TRACKING_PASSWORD

3. Menjalankan Training & Monitoring

Jalankan script utama. Script ini akan melakukan training, logging ke DagsHub, dan membuka port 8000 untuk monitoring.

python modelling.py


Tunggu hingga muncul pesan "Script is now STANDBY to serve metrics".

4. Menjalankan Dashboard Monitoring

Buka 2 terminal baru untuk menjalankan service berikut:

Terminal Prometheus:

./prometheus.exe --config.file=prometheus.yml


Terminal Grafana:

./grafana-server.exe


Akses Grafana di http://localhost:3000.

📊 Hasil Monitoring

Experiment Tracking: Tersedia di DagsHub Repository.

Real-time Metrics: Tersedia di Dashboard Grafana (Akurasi Model per Depth).

Submission MLOps Dicoding - Fikri Mandala