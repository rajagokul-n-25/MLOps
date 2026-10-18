import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import MLFLOW_URI, MODEL_NAME

app = FastAPI(title="House Price Predictor API")
model = None


class HouseInput(BaseModel):
    Home: int = 1
    SqFt: int = 1800
    Bedrooms: int = 3
    Bathrooms: int = 2
    Offers: int = 2
    Brick: str = "No"
    Neighborhood: str = "East"


@app.on_event("startup")
def load_champion_model():
    """Load the current @champion model from MLflow."""
    global model
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
        print("Loaded Champion model from MLflow successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


@app.get("/health")
def health():
    return {"status": "ready" if model is not None else "model_not_loaded"}


@app.post("/predict")
def predict(data: HouseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    # Convert incoming JSON into a pandas DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Predict price
    price = model.predict(df)[0]
    return {"predicted_price": round(float(price), 2)}
