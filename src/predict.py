"""
predict.py
-------------------------------------
Loads the trained credit risk model from MLflow Production stage
and performs predictions. Works for batch inference or CLI usage.
"""

import pandas as pd
import yaml
import os
import mlflow
from mlflow.tracking import MlflowClient

# ============================================================
# 1️⃣ Load configuration
# ============================================================
def load_config(path: str = "src/config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Config file not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ============================================================
# 2️⃣ Load Production model dynamically
# ============================================================
def load_production_model(model_name="Credit_Risk_Model"):
    client = MlflowClient()
    prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
    if not prod_versions:
        raise ValueError(f"❌ No Production model found for {model_name}")
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Loaded Production model {model_name} version {prod_versions[0].version}")
    return model

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
