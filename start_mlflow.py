import os
import time
import subprocess

# ==============================
# 1️⃣ Start MLflow UI
# ==============================
print("🚀 Starting MLflow UI at http://127.0.0.1:5000 ...")

# Windows: open in new cmd window
subprocess.Popen(
    ["start", "cmd", "/k", "mlflow ui --backend-store-uri mlruns --port 5000"],
    shell=True
)

# Linux/Mac users can use:
# subprocess.Popen(["mlflow", "ui", "--backend-store-uri", "mlruns", "--port", "5000"])

# Wait a few seconds to allow MLflow UI to start
time.sleep(5)

# ==============================
# 2️⃣ Run model training
# ==============================
print("🏋️‍♂️ Running model training...")
train_script = "src/train_model.py"
os.system(f"python {train_script}")

# ==============================
# 3️⃣ Generate SHAP explainability report
# ==============================
print("📄 Generating SHAP explainability report...")
xai_script = "src/generate_xai_report.py"
os.system(f"python {xai_script}")

# ==============================
# 4️⃣ Done
# ==============================
print("✅ All tasks completed!")
print("🔗 MLflow UI: http://127.0.0.1:5000")
print("📊 SHAP report saved at: reports/shap_report.html")
print("🗂 Model artifact saved at: models/model.pkl")
