"""
main.py
-------------------------------------
FastAPI app for Credit Risk Scoring API
with SHAP explainability and OpenAPI documentation.
Dynamically loads the latest Production model from MLflow Model Registry.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import yaml
import os
import json
import traceback
from typing import Dict
import shap
import mlflow
from mlflow.tracking import MlflowClient
import joblib

# ============================================================
# 0️⃣ Set MLflow tracking URI (supports local + CI/CD)
# ============================================================
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"✅ MLflow tracking URI: {mlflow_tracking_uri}")

# ============================================================
# 1️⃣ Load configuration
# ============================================================
CONFIG_PATH = "src/config.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load config.yaml: {e}")

PREDICTION_LOG = os.getenv("PREDICTION_LOG", "reports/predictions.log")
SHAP_DATA_PATH = "reports/shap_data.pkl"
MODEL_NAME = "Credit_Risk_Model"

# ============================================================
# 2️⃣ Load Production model dynamically from MLflow
# ============================================================
def load_production_model(model_name=MODEL_NAME):
    client = MlflowClient()
    prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
    if not prod_versions:
        raise RuntimeError(f"❌ No Production model found for {model_name}")

    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Loaded Production model {model_name} version {prod_versions[0].version}")
    return model

model = load_production_model()

# ============================================================
# 3️⃣ Load precomputed SHAP explainer (optional)
# ============================================================
explainer = None
if os.path.exists(SHAP_DATA_PATH):
    try:
        shap_data = joblib.load(SHAP_DATA_PATH)
        explainer = shap_data.get("explainer")
        print("✅ Loaded SHAP explainer from saved data.")
    except Exception as e:
        print(f"⚠️ Could not load SHAP explainer: {e}")

# ============================================================
# 4️⃣ FastAPI initialization
# ============================================================
app = FastAPI(
    title="💳 Credit Risk Scoring API",
    description="""
    ### 🚀 Predict Loan Default Risk using Explainable AI (SHAP)
    Endpoints:
    - **/predict** → Predict credit risk score
    - **/explain** → SHAP feature importance explanations
    """,
    version="1.2.0",
)

# ============================================================
# 5️⃣ Input Schema
# ============================================================

class CreditInput(BaseModel):
    Loan_Amount: float = Field(..., alias="Loan Amount", json_schema_extra={"example": 50000})
    Debt_to_Income_Ratio: float = Field(..., alias="Debt-to-Income Ratio", json_schema_extra={"example": 0.35})
    Credit_Score: float = Field(..., alias="Credit Score", json_schema_extra={"example": 720})
    Assets_Value: float = Field(..., alias="Assets Value", json_schema_extra={"example": 150000})
    Age: float = Field(..., json_schema_extra={"example": 32})
    Income: float = Field(..., json_schema_extra={"example": 80000})
    Number_of_Dependents: int = Field(..., alias="Number of Dependents", json_schema_extra={"example": 2})
    Education_Level: str = Field(..., alias="Education Level", json_schema_extra={"example": "Bachelor's"})
    Payment_History: str = Field(..., alias="Payment History", json_schema_extra={"example": "Good"})
    Marital_Status: str = Field(..., alias="Marital Status", json_schema_extra={"example": "Single"})
    Gender: str = Field(..., json_schema_extra={"example": "Female"})
    Employment_Status: str = Field(..., alias="Employment Status", json_schema_extra={"example": "Employed"})
    Loan_Purpose: str = Field(..., alias="Loan Purpose", json_schema_extra={"example": "Home Improvement"})


# ============================================================
# 6️⃣ Utility to log predictions
# ============================================================
def log_prediction(entry: Dict):
    os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ============================================================
# 7️⃣ Routes
# ============================================================
@app.get("/", tags=["General"])
def root():
    return {"message": "Credit Risk Scoring API is running 🚀"}

@app.get("/health", tags=["General"])
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", tags=["Prediction"])
def predict(input_data: CreditInput):
    try:
        df = pd.DataFrame([input_data.model_dump(by_alias=True)])
        expected_cols = [
            'Loan Amount', 'Debt-to-Income Ratio', 'Credit Score', 'Assets Value',
            'Age', 'Income', 'Number of Dependents', 'Education Level',
            'Payment History', 'Marital Status', 'Gender', 'Employment Status', 'Loan Purpose'
        ]
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        pred_prob = model.predict_proba(df)[0, 1]
        pred_label = int(pred_prob > 0.5)

        entry = {
            "input": input_data.model_dump(by_alias=True),
            "prediction": pred_label,
            "probability": round(float(pred_prob), 4),
        }
        log_prediction(entry)
        return entry
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain", tags=["Explainability"])
def explain(input_data: CreditInput):
    try:
        df = pd.DataFrame([input_data.dict(by_alias=True)])
        expected_cols = [
            'Loan Amount', 'Debt-to-Income Ratio', 'Credit Score', 'Assets Value',
            'Age', 'Income', 'Number of Dependents', 'Education Level',
            'Payment History', 'Marital Status', 'Gender', 'Employment Status', 'Loan Purpose'
        ]
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        transformed = model.named_steps["preprocessor"].transform(df)

        if explainer is not None:
            shap_values = explainer.shap_values(transformed)
        else:
            explainer_local = shap.TreeExplainer(model.named_steps["model"])
            shap_values = explainer_local.shap_values(transformed)

        if isinstance(shap_values, list):
            shap_array = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_array = shap_values[0]

        shap_dict = dict(zip(df.columns, shap_array.tolist()))
        entry = {
            "input": input_data.dict(by_alias=True),
            "explanation": shap_dict,
        }
        log_prediction(entry)
        return entry

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

# ============================================================
# 8️⃣ Run locally
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
