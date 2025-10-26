"""
promote_model.py
-------------------------------------
Promotes the latest Staging model version to Production.
"""

from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # adjust if your server runs elsewhere

MODEL_NAME = "Credit_Risk_Model"

client = MlflowClient()

# Fetch the latest version in Staging
staging_versions = client.get_latest_versions(name=MODEL_NAME, stages=["Staging"])
if not staging_versions:
    raise ValueError("❌ No model in Staging found to promote.")

latest_staging_version = staging_versions[0]

# Transition to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_staging_version.version,
    stage="Production"
)

print(f"✅ Model {MODEL_NAME} version {latest_staging_version.version} is now in Production.")
