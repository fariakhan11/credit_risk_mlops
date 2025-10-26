from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Credit Risk Scoring" in response.json()["message"]

def test_predict_endpoint():
    sample = {"data": {"Age": 35, "Income": 50000, "LoanAmount": 12000, "LoanDuration": 24, "Gender": "Male"}}
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
