import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def train_model():
    """
    Train Random Forest models with hyperparameter tuning and log results to DagsHub via MLflow.
    Target: Titanic Survival Prediction.
    """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def train_model():
    """
    Train Random Forest models with hyperparameter tuning and log results to DagsHub via MLflow.
    Target: Titanic Survival Prediction.
    """
    print("[INFO] Initializing DagsHub and MLflow configuration...")
    
    # 1. Initialize DagsHub and MLflow Configuration
    mlflow.set_tracking_uri("https://dagshub.com/fikrimandalaa/Submission_MLOps_Fikrimandala.mlflow")
    mlflow.set_experiment("Titanic_Model_CI_Pipeline")

    # 2. Data Loading and Preparation
    dataset_path = 'titanic_clean.csv'
    print(f"[INFO] Loading dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset '{dataset_path}' not found. Process aborted.")
        return

    df = pd.read_csv(dataset_path)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Split dataset: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Hyperparameter Tuning Execution
    params_list = [
        {'n_estimators': 50, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 10}
    ]

    for i, params in enumerate(params_list):
        run_name = f"CI_Run_{i+1}_Depth_{params['max_depth']}"
        print(f"[INFO] Executing training run: {run_name}")

        with mlflow.start_run(run_name=run_name, nested=True):
            # A. Model Training
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], 
                random_state=42
            )
            model.fit(X_train, y_train)

            # B. Prediction
            y_pred = model.predict(X_test)

            # C. Metrics Calculation
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred)
            }

            print(f"   Run Completed. Accuracy: {metrics['accuracy']:.4f}")

            # D. Logging to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            # E. Artifact Generation
            # 1. Confusion Matrix
            plt.figure(figsize=(6,5))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {run_name}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()

            # 2. Feature Importance
            feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            plt.figure(figsize=(8,5))
            sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()

    print("\n[SUCCESS] CI Workflow completed. Results verified and uploaded to DagsHub.")

if __name__ == "__main__":
    train_model()