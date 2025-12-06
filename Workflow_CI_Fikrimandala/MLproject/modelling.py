import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Import library monitoring
from prometheus_client import start_http_server, Gauge

# --- 1. DEFINISI METRIC MONITORING ---
ACCURACY_METRIC = Gauge('titanic_model_accuracy', 'Accuracy of the Random Forest Model', ['depth'])

def train_model():
    """
    Train Random Forest models with hyperparameter tuning and log results to DagsHub via MLflow.
    Target: Titanic Survival Prediction.
    """
    print("[INFO] Initializing DagsHub and MLflow configuration...")
    
    # Setup MLflow ke DagsHub
    mlflow.set_tracking_uri("https://dagshub.com/fikrimandalaa/Submission_MLOps_Fikrimandala.mlflow")
    mlflow.set_experiment("Titanic_Model_CI_Pipeline")

    # Load Data
    dataset_path = 'titanic_clean.csv'
    print(f"[INFO] Loading dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset '{dataset_path}' not found. Process aborted.")
        return

    df = pd.read_csv(dataset_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Rencana Training
    params_list = [
        {'n_estimators': 50, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 10}
    ]

    for i, params in enumerate(params_list):
        run_name = f"CI_Run_{i+1}_Depth_{params['max_depth']}"
        print(f"[INFO] Executing training run: {run_name}")

        with mlflow.start_run(run_name=run_name, nested=True):
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], 
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Hitung Metrics
            acc = accuracy_score(y_test, y_pred)
            metrics = {
                'accuracy': acc,
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred)
            }

            print(f"   Run Completed. Accuracy: {acc:.4f}")

            # --- 2. UPDATE KE PROMETHEUS ---
            # Kirim data akurasi ke server monitoring
            ACCURACY_METRIC.labels(depth=params['max_depth']).set(acc)
            print(f"   [MONITORING] Prometheus metric updated -> Acc {acc:.4f}")

            # Logging MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            # Simpan Gambar Artifacts
            plt.figure(figsize=(6,5))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {run_name}")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()

            feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            plt.figure(figsize=(8,5))
            sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()
            
            # Kasih jeda 3 detik
            time.sleep(3) 

    print("\n[SUCCESS] Workflow completed.")

if __name__ == "__main__":
    # --- 3. NYALAKAN SERVER MONITORING (PORT 8000) ---
    print("[MONITORING] Starting Prometheus metrics server on port 8000...")
    
    start_http_server(8000)
    
    # Jalankan training
    train_model()
    

    print("[INFO] Script is now STANDBY to serve metrics.")
    print("[INFO] DO NOT CLOSE THIS TERMINAL. Press Ctrl+C to stop.")
    
    while True:
        time.sleep(1)