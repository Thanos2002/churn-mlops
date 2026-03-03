import mlflow.sklearn
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Churn Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-champion")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerData):
    df = pd.DataFrame([customer.model_dump()])
    proba_value = float(model.predict_proba(df)[:, 1][0])
    label = "Yes" if proba_value >= 0.5 else "No"
    return {
        "churn_probability": round(proba_value, 4),
        "churn_prediction": label,
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}"
    }
