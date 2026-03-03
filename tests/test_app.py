import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Mock the model BEFORE app.py is imported (so it never calls MLflow)
mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

with patch("mlflow.sklearn.load_model", return_value=mock_model):
    from fastapi.testclient import TestClient
    from service.app import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 140.70
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_valid_response():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    assert "churn_probability" in response.json()
    assert "churn_prediction" in response.json()

def test_predict_churn_yes():
    # mock returns 0.7 probability → should predict "Yes"
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.json()["churn_prediction"] == "Yes"
    assert response.json()["churn_probability"] == 0.7
