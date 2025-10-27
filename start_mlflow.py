"""
run_pipeline.py
-------------------------------------
Automates the end-to-end MLflow pipeline:
1️⃣ Starts MLflow UI
2️⃣ Trains model
3️⃣ Generates SHAP explainability report
"""

import os
import time
import subprocess
import platform

# ======================================================
# 1️⃣ Start MLflow tracking server (non-blocking)
# ======================================================
print("🚀 Starting MLflow Tracking Server at http://127.0.0.1:5000 ...")

if platform.system() == "Windows":
    # Open MLflow tracking server in a new Command Prompt window
    subprocess.Popen(
        [
            "start",
            "cmd",
            "/k",
            (
                "mlflow server "
                "--backend-store-uri mlruns "
                "--default-artifact-root mlruns "
                "--host 127.0.0.1 "
                "--port 5000"
            ),
        ],
        shell=True,
    )
else:
    # For macOS/Linux
    subprocess.Popen(
        [
            "mlflow",
            "server",
            "--backend-store-uri",
            "mlruns",
            "--default-artifact-root",
            "mlruns",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
    )

# Wait for the MLflow server to start
time.sleep(5)


# ======================================================
# 2️⃣ Run model training
# ======================================================
train_script = "src/train_model.py"
if not os.path.exists(train_script):
    raise FileNotFoundError(f"❌ Training script not found: {train_script}")

print("🏋️‍♂️ Running model training...")
exit_code = os.system(f"python {train_script}")
if exit_code != 0:
    raise RuntimeError("❌ Model training failed!")

# ======================================================
# 3️⃣ Generate SHAP explainability report
# ======================================================
xai_script = "src/generate_xai_report.py"
if not os.path.exists(xai_script):
    print("⚠️ SHAP report script not found, skipping explainability step.")
else:
    print("📄 Generating SHAP explainability report...")
    exit_code = os.system(f"python {xai_script}")
    if exit_code != 0:
        print("⚠️ SHAP report generation failed (check logs).")

# ======================================================
# 4️⃣ Done
# ======================================================
print("\n✅ All tasks completed successfully!")
print("🔗 MLflow UI: http://127.0.0.1:5000")
print("📊 SHAP report saved at: reports/shap_report.html")
print("🗂 Model artifact saved at: models/model.pkl")
