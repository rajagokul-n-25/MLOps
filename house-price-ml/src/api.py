import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Works whether you run from project root or inside src folder
try:
    from src.config import (
        MLFLOW_REGISTERED_MODEL,
        MLFLOW_TRACKING_URI,
        MODEL_CHAMPION_ALIAS,
    )
except ModuleNotFoundError:
    from config import (
        MLFLOW_REGISTERED_MODEL,
        MLFLOW_TRACKING_URI,
        MODEL_CHAMPION_ALIAS,
    )

app = FastAPI(title="House Price Prediction API")

# Global variable to store our loaded model in memory
model = None


# 1. Define the input data format expected from user
class HouseData(BaseModel):
    Home: int
    SqFt: int
    Bedrooms: int
    Bathrooms: int
    Offers: int
    Brick: str
    Neighborhood: str


# 2. When API starts, load the @champion model from MLflow
@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_REGISTERED_MODEL}@{MODEL_CHAMPION_ALIAS}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f" Loaded model successfully: {model_uri}")
    except Exception as error:
        print(f" Could not load model: {error}")


# 3. Simple health check
@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_ready",
        "model": MLFLOW_REGISTERED_MODEL,
        "alias": MODEL_CHAMPION_ALIAS,
    }


# 4. Predict house price
@app.post("/predict")
def predict(house: HouseData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # Convert input JSON to DataFrame (same columns model was trained on)
    input_data = pd.DataFrame([house.model_dump()])
    # Predict price
    predicted_price = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(predicted_price), 2),
        "model_used": MLFLOW_REGISTERED_MODEL,
    }
