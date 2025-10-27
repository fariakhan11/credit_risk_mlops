import sys
import os
sys.path.append(os.path.abspath("."))

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Credit Risk Scoring" in response.json()["message"]


def test_predict_endpoint():
    sample = {
        "Loan Amount": 50000,
        "Debt-to-Income Ratio": 0.35,
        "Credit Score": 720,
        "Assets Value": 150000,
        "Age": 32,
        "Income": 80000,
        "Number of Dependents": 2,
        "Education Level": "Bachelor's",
        "Payment History": "Good",
        "Marital Status": "Single",
        "Gender": "Female",
        "Employment Status": "Employed",
        "Loan Purpose": "Home Improvement"
    }

    response = client.post("/predict", json=sample)
    assert response.status_code == 200, f"Response: {response.text}"
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
