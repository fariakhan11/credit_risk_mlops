"""
promote_model.py
-------------------------------------
Promotes the latest Staging model version to Production.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

# --- Step 1: Choose MLflow tracking URI ---
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# If running in GitHub Actions, use local file-based tracking
if "GITHUB_ACTIONS" in os.environ:
    mlflow_tracking_uri = f"file://{os.path.abspath('mlruns')}"

mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"✅ MLflow tracking URI: {mlflow_tracking_uri}")

# --- Step 2: Define model name ---
MODEL_NAME = "Credit_Risk_Model"
client = MlflowClient()

# --- Step 3: Get latest model in Staging ---
staging_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
staging_versions = [v for v in staging_versions if v.current_stage == "Staging"]

if not staging_versions:
    raise ValueError("❌ No model in Staging found to promote.")

latest_staging_version = staging_versions[0]

# --- Step 4: Promote to Production ---
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_staging_version.version,
    stage="Production"
)

print(f"✅ Model {MODEL_NAME} version {latest_staging_version.version} is now in Production.")
