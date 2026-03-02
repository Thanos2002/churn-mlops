import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# Load model once at startup
model = joblib.load("model.joblib")

# Define the input schema (matches dataset columns)
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
    proba = model.predict_proba(df)[:, 1][0]
    label = "Yes" if proba >= 0.5 else "No"
    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": label
    }
