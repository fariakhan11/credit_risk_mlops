# === BASE ===
FROM python:3.10-slim

# === SETUP ===
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# === EXPOSE PORT ===
EXPOSE 8080

# === RUN ===
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
