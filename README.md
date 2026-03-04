# 📊 Churn Prediction MLOps

**An end-to-end production-grade ML service**

 >   Predicting customer churn is vital for retention. This project isn't just about the model. It's about the lifecycle. 
 >   It features a robust pipeline for experiment tracking, automated deployment, and scalable serving.
---

## 🛠️ Tech Stack

| Tool           | Purpose                              |
| -------------- | ------------------------------------ |
| scikit-learn   | ML pipeline + model training         |
| MLflow         | Experiment tracking + Model Registry |
| FastAPI        | REST API for model serving           |
| Docker         | Containerization                     |
| GitHub Actions | CI/CD (lint + test + build)          |



---
## 🚀 Getting Started
### Option 1: Run with Docker (Recommended)
```
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Option 2: Local Development

#### 1. Clone the repository
```
git clone https://github.com/Thanos2002/churn-mlops.git
cd churn-mlops
```
#### 2. Set up a virtual environment
```
python -m venv .venv
```
#### 3. Activate it
- Windows
```
.venv\Scripts\Activate.ps1 
```
- Mac/Linux
```
source .venv/bin/activate
```
#### 4. Install dependencies
```
pip install -r requirements.txt
```
#### 5. Train Model & Track with MLflow
```
python src/train.py
mlflow ui  # View results at http://localhost:5000
```
#### 6. Start the API
```
uvicorn service.app:app --reload
```

