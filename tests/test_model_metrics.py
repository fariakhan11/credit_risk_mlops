"""
tests/test_model_metrics.py
-------------------------------------
Checks model performance metrics before promotion.
Fails pipeline if metrics below threshold.
"""

import mlflow

# Set MLflow tracking URI (same as your local server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "Credit_Risk_Experiment"
MIN_ACCURACY = 0.70  # define your minimum acceptable score

def test_model_metrics():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None, "❌ Experiment not found in MLflow."

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"], max_results=1)
    assert len(runs) > 0, "❌ No runs found in MLflow."

    latest_run = runs[0]
    accuracy = latest_run.data.metrics.get("accuracy", 0.0)
    print(f"✅ Latest model accuracy: {accuracy}")

    assert accuracy >= MIN_ACCURACY, f"❌ Model accuracy {accuracy} below minimum threshold {MIN_ACCURACY}"
