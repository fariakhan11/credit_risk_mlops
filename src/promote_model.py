"""
promote_model.py
-------------------------------------
Promotes the latest Staging model version to Production.
"""

import os
import pathlib
import mlflow
from mlflow.tracking import MlflowClient

# --- Step 1: Choose MLflow tracking URI ---
# Default to local MLflow file-based store or server
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# If running in GitHub Actions, prefer the file-based mlruns directory
if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
    mlflow_tracking_uri = pathlib.Path("mlruns").resolve().as_uri()

mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"✅ Using MLflow tracking URI: {mlflow_tracking_uri}")

# --- Step 2: Define model name ---
MODEL_NAME = "Credit_Risk_Model"
client = MlflowClient(tracking_uri=mlflow_tracking_uri)

# --- Step 3: Locate latest model in Staging ---
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
staging_versions = sorted(
    [v for v in versions if v.current_stage == "Staging"],
    key=lambda v: int(v.version),
    reverse=True,
)

if not staging_versions:
    raise SystemExit("❌ No Staging model found to promote.")

latest = staging_versions[0]
print(f"ℹ️ Found Staging model: {MODEL_NAME} (version {latest.version})")

# --- Step 4: Promote to Production ---
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest.version,
    stage="Production",
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=latest.version,
    key="stage",
    value="Production",
)

print(f"✅ Model {MODEL_NAME} version {latest.version} successfully promoted to PRODUCTION.")
