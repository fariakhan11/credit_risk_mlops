"""
tests/test_model_metrics.py
-------------------------------------
Checks model performance metrics before promotion.
Fails pipeline if metrics below threshold.
"""

import os
import mlflow

# ===============================
# 0️⃣ Set MLflow tracking URI
# ===============================
# Use env variable if set (for GitHub Actions CI), otherwise default to local server
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

EXPERIMENT_NAME = "Credit_Risk_Scoring"
MIN_ACCURACY = 0.70  # minimum acceptable accuracy
MIN_PRECISION = 0.60
MIN_RECALL = 0.40

def test_model_metrics():
    client = mlflow.tracking.MlflowClient()
    
    # ✅ Check experiment exists
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None, f"❌ Experiment '{EXPERIMENT_NAME}' not found in MLflow."

    # ✅ Fetch latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    assert runs, "❌ No runs found in experiment."

    latest_run = runs[0]
    metrics = latest_run.data.metrics

    accuracy = metrics.get("accuracy", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)

    print(f"✅ Latest run metrics: accuracy={accuracy}, precision={precision}, recall={recall}")

    # ✅ Threshold checks
    assert accuracy >= MIN_ACCURACY, f"❌ Accuracy {accuracy} below threshold {MIN_ACCURACY}"
    assert precision >= MIN_PRECISION, f"❌ Precision {precision} below threshold {MIN_PRECISION}"
    assert recall >= MIN_RECALL, f"❌ Recall {recall} below threshold {MIN_RECALL}"
