# Navigate to project
cd churn-mlops

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Train model + log to MLflow
python src/train.py

# Open MLflow UI → http://localhost:5000
mlflow ui

# Start API → http://localhost:8000/docs
uvicorn service.app:app --reload

# Build image
docker build -t churn-api .

# Run container
docker run -p 8000:8000 churn-api

# Save and push changes to GitHub
git add .
git commit -m "your message here"
git push
