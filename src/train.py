import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from mlflow.tracking import MlflowClient
import yaml

# --- Load config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Load data ---
df = pd.read_csv(config["data"]["path"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns=["customerID"])
df["Churn"] = (df[config["data"]["target_column"]] == "Yes").astype(int)

X = df.drop(columns=["Churn"])
y = df["Churn"]


# --- Feature groups ---
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

# --- Preprocessing ---
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])



MODEL_MAP = {
    "RF": RandomForestClassifier,
    "GBC": GradientBoostingClassifier
}

pipelines = {}
for model_name, model_params in config["models"].items():
    # GBC doesn't support class_weight — remove it if present
    params = {k: v for k, v in model_params.items()
              if not (model_name == "GBC" and k == "class_weight")}
    pipelines[model_name] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", MODEL_MAP[model_name](**params))
    ])


# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    stratify=y,
    random_state=42
)


# --- MLflow tracking ---
mlflow.set_experiment(config["mlflow"]["experiment_name"])
mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=False)


with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
    mlflow.log_artifact("config.yaml")  # ← move here, once for the whole experiment

    for name, pipe in pipelines.items():
        with mlflow.start_run(run_name=name, nested=True):

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)

            print(f"\n--- {name} ---")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(classification_report(y_test, y_pred))


# --- Register best model to Model Registry ---
best_model_name = "RF"  # change this if GBC scores better
registered_model_name = "churn-champion"

# Get the last run ID for the best model
client = MlflowClient()
experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{best_model_name}'",
    order_by=["metrics.roc_auc DESC"],
    max_results=1
)

best_run_id = runs[0].info.run_id

# Register model and set champion alias directly
result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="churn-champion"
)

client.set_registered_model_alias("churn-champion", "champion", result.version)
print(f"✅ Model v{result.version} registered and tagged as champion")
