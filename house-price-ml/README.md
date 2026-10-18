# 🏡 House Price Prediction (Simple MLOps)

A simple, beginner-friendly MLOps project for predicting house prices using **PostgreSQL**, **DVC**, **MLflow**, and **FastAPI**.

---

## 📂 Project Structure

```text
house-price-ml/
│
├── data/
│   └── house-prices.csv      # Dataset (tracked by DVC)
│
├── src/
│   ├── config.py             # Database and project settings
│   ├── ingest.py             # Step 1: Read from PostgreSQL -> Save to CSV
│   ├── train.py              # Step 2: Train model & log to MLflow
│   └── app.py                # Step 3: FastAPI prediction server
│
├── dvc.yaml                  # DVC pipeline (ingest -> train)
├── params.yaml               # Hyperparameters tracked by DVC
├── docker-compose.yml        # PostgreSQL & MLflow containers
├── Dockerfile.api            # Docker container for FastAPI
├── .env                      # Database password & configuration
└── requirements.txt          # Python dependencies
```

---

## 🚀 Step-by-Step Guide

### Step 1: Start PostgreSQL and MLflow
```powershell
docker compose up -d
```
- PostgreSQL runs at `localhost:5433`
- MLflow UI runs at [http://localhost:5000](http://localhost:5000)

---

### Step 2: Run the DVC Pipeline
Run the entire automated pipeline:
```powershell
dvc repro
```
This automatically runs:
1. `src/ingest.py`: Fetches the latest data from PostgreSQL into `data/house-prices.csv`.
2. `src/train.py`: Trains the Linear Regression model, logs experiments to MLflow, and exports `metrics.json`.

---

### Step 3: Run the FastAPI Prediction API
Start the prediction server:
```powershell
uvicorn src.app:app --reload
```
Open your browser at: [http://localhost:8000/docs](http://localhost:8000/docs) to test predictions interactively!

Or send a POST request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Home": 1,
    "SqFt": 1800,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Offers": 2,
    "Brick": "No",
    "Neighborhood": "East"
  }'
```
Response:
```json
{
  "predicted_price": 128450.00
}
```

---

## 📊 View Model Metrics & DAG

```powershell
# View DVC pipeline flowchart
dvc dag
```

Open the MLflow dashboard at **http://localhost:5000** to see:
- Model Accuracy ($R^2$, MAE, RMSE)
- Residual Error charts
- Model Registry (`@champion` model)
