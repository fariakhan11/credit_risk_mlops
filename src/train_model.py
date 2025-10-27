"""
train_model.py
-------------------------------------
Trains a binary classification model for credit risk scoring
with preprocessing, class balance handling, and SHAP explainability.
Logs parameters, metrics, and model artifacts to MLflow,
and registers the model to MLflow Model Registry (Staging stage).
"""

import os, pathlib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import yaml
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from mlflow.tracking import MlflowClient

# ==============================
# 0️⃣ MLflow tracking URI
# ==============================

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow_cfg = config["mlflow"]

# Dynamically choose tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    if "GITHUB_ACTIONS" in os.environ:
        # ✅ Use local file path in CI/CD (no MLflow server available)
        MLFLOW_TRACKING_URI = f"file://{os.path.abspath('mlruns')}"
    else:
        # ✅ Local dev: use mlruns folder or MLflow UI if running
        if os.path.exists("mlruns"):
            MLFLOW_TRACKING_URI = pathlib.Path("mlruns").resolve().as_uri()
        else:
            MLFLOW_TRACKING_URI = mlflow_cfg.get("tracking_uri", "http://127.0.0.1:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(mlflow_cfg["experiment_name"])

print(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# ==============================
# 1️⃣ Load configuration
# ==============================
with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["paths"]["data"]
MODEL_PATH = config["paths"]["model"]
MLFLOW_TRACKING_URI = config["mlflow"]["tracking_uri"]
EXPERIMENT_NAME = config["mlflow"]["experiment_name"]
MODEL_REGISTRY_NAME = "Credit_Risk_Model"

# ==============================
# 2️⃣ Load dataset
# ==============================
df = pd.read_excel(DATA_PATH)
print(f"✅ Raw dataset loaded: {df.shape}")

df = df.drop_duplicates().reset_index(drop=True)

# ==============================
# 3️⃣ Clean numeric columns
# ==============================
numeric_like_cols = [
    'Loan Amount', 'Debt-to-Income Ratio', 'Credit Score',
    'Assets Value', 'Age', 'Income', 'Number of Dependents'
]

for col in numeric_like_cols:
    if df[col].dtype == 'object':
        df[col] = (
            df[col]
            .astype(str)
            .replace('[\$,]', '', regex=True)
            .replace('None', np.nan)
        )
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("✅ Cleaned numeric columns (removed $, commas, and coerced types).")

# ==============================
# 4️⃣ Handle missing values
# ==============================
df = df.fillna({
    'Education Level': 'Unknown',
    'Marital Status': 'Unknown',
    'Gender': 'Unknown',
    'Employment Status': 'Unknown',
    'Payment History': 'Unknown',
    'Loan Purpose': 'Unknown'
})

for col in numeric_like_cols:
    df[col] = df[col].fillna(df[col].median())

print("✅ Missing values handled for numeric and categorical columns.")

# ==============================
# 5️⃣ Define feature sets
# ==============================
target_col = "Risk Rating"

num_cols = numeric_like_cols
cat_cols = [
    'Education Level', 'Payment History', 'Marital Status',
    'Gender', 'Employment Status', 'Loan Purpose'
]

X = df[num_cols + cat_cols]
y = df[target_col]

# ==============================
# 6️⃣ Split data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Train/Test split: {X_train.shape}, {X_test.shape}")

# ==============================
# 7️⃣ Preprocessing
# ==============================
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# ==============================
# 8️⃣ Build pipeline
# ==============================
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# ==============================
# 9️⃣ Train model
# ==============================
clf.fit(X_train, y_train)
print("✅ Model trained successfully.")

# ==============================
# 🔟 Evaluate
# ==============================
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
}

print("\n📊 Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ==============================
# 11️⃣ MLflow Logging & Registry
# ==============================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 250)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.sklearn.log_model(clf, "model")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

    # ---------------- Register Model to Staging ----------------
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_REGISTRY_NAME)
    except Exception:
        pass  # already exists

    model_version = client.create_model_version(
        name=MODEL_REGISTRY_NAME,
        source=f"runs:/{run.info.run_id}/model",
        run_id=run.info.run_id
    )

    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=model_version.version,
        stage="Staging"
    )

    print(f"✅ Model registered as {MODEL_REGISTRY_NAME} version {model_version.version} in Staging")

# ==============================
# 12️⃣ SHAP Explainability
# ==============================
try:
    rf_model = clf.named_steps['model']
    X_test_processed = clf.named_steps['preprocessor'].transform(X_test)

    X_test_processed = np.array(X_test_processed, dtype=float)
    feature_names = clf.named_steps['preprocessor'].get_feature_names_out()

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_processed)

    os.makedirs("reports", exist_ok=True)
    joblib.dump({
        "explainer": explainer,
        "shap_values": shap_values,
        "X_test_processed": X_test_processed,
        "feature_names": feature_names,
        "y_test": y_test
    }, "reports/shap_data.pkl")

    print("✅ SHAP values computed and saved successfully.")

except Exception as e:
    print(f"⚠️ SHAP computation failed: {e}")

print("🎯 Training pipeline completed successfully.")
