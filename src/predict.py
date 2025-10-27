"""
predict.py
-------------------------------------
Loads the trained credit risk model from MLflow Production stage
and performs predictions. Works for batch inference or CLI usage.
"""
import os, joblib
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import HTTPException
import pandas as pd
import pathlib

# ============================================================
# 1️⃣ Load configuration
# ============================================================
def load_config(path: str = "src/config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Config file not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 2️⃣ Load Production model dynamically (cross-platform safe)
# ============================================================


def load_production_model(model_name="Credit_Risk_Model"):
    """
    Loads the latest 'Production' stage model from MLflow registry.
    Falls back to 'Staging' or latest local run for CI/CD pipelines.
    """
    try:
        import os
        import pathlib
        import mlflow
        from mlflow.tracking import MlflowClient
        from fastapi import HTTPException

        # ✅ Choose tracking URI dynamically
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        if not MLFLOW_TRACKING_URI:
            if os.path.exists("mlruns"):
                MLFLOW_TRACKING_URI = pathlib.Path("mlruns").resolve().as_uri()
            else:
                MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")

        client = MlflowClient()

        # ✅ Try Production
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if prod_versions:
            model_uri = f"models:/{model_name}/Production"
            print(f"✅ Found Production model version {prod_versions[0].version}")

        else:
            # ✅ Try Staging
            staging_versions = [v for v in versions if v.current_stage == "Staging"]
            if staging_versions:
                model_uri = f"models:/{model_name}/Staging"
                print(f"⚠️ Using Staging model version {staging_versions[0].version}")
            else:
                # ✅ Fallback: load latest run artifact directly (CI-safe)
                print("⚠️ No registered model found. Loading latest run artifact...")
                experiments = client.list_experiments()
                if not experiments:
                    raise HTTPException(status_code=503, detail="No MLflow experiments found.")
                last_exp = experiments[-1]
                runs = client.search_runs(last_exp.experiment_id, order_by=["attributes.start_time DESC"])
                if not runs:
                    raise HTTPException(status_code=503, detail="No model run found in MLflow.")
                last_run_id = runs[0].info.run_id
                model_uri = f"runs:/{last_run_id}/model"
                print(f"✅ Loaded model from latest run: {last_run_id}")

        # ✅ Load model
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Loaded model from {model_uri}")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")


# ============================================================
# 3️⃣ Predict function
# ============================================================
def predict(model, input_data: pd.DataFrame) -> pd.DataFrame:
    expected_cols = [
        'Loan Amount',
        'Debt-to-Income Ratio',
        'Credit Score',
        'Assets Value',
        'Age',
        'Income',
        'Number of Dependents',
        'Education Level',
        'Payment History',
        'Marital Status',
        'Gender',
        'Employment Status',
        'Loan Purpose'
    ]

    missing_cols = set(expected_cols) - set(input_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    results = input_data.copy()
    results["Predicted_Risk"] = predictions
    results["Probability_HighRisk"] = probabilities.round(4)
    return results

# ============================================================
# 4️⃣ Example CLI usage
# ============================================================
if __name__ == "__main__":
    # Load Production model
    model = load_production_model()

    # Example input
    example_data = pd.DataFrame([{
        "Loan Amount": 50000,
        "Debt-to-Income Ratio": 0.35,
        "Credit Score": 720,
        "Assets Value": 100000,
        "Age": 35,
        "Income": 80000,
        "Number of Dependents": 2,
        "Education Level": "Bachelor's",
        "Payment History": "Good",
        "Marital Status": "Married",
        "Gender": "Male",
        "Employment Status": "Employed",
        "Loan Purpose": "Home Improvement"
    }])

    output = predict(model, example_data)
    print("\n📊 Prediction Result:")
    print(output)
