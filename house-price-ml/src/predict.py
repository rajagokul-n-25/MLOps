import mlflow
import mlflow.sklearn
import pandas as pd

# Support running from root or inside src folder
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


def main():
    # 1. Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 2. Load the current @champion model from registry
    model_uri = f"models:/{MLFLOW_REGISTERED_MODEL}@{MODEL_CHAMPION_ALIAS}"
    print(f"Loading champion model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error: Could not load champion model: {e}")
        print("Tip: Run 'python src/train.py' first to train and register a champion model.")
        return

    # 3. Sample house features for prediction
    sample_house = pd.DataFrame(
        [
            {
                "Home": 1,
                "SqFt": 1500,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Offers": 2,
                "Brick": "No",
                "Neighborhood": "East",
            }
        ]
    )

    # 4. Predict price
    predicted_price = model.predict(sample_house)[0]

    print("\n--- Prediction Result ---")
    print(f"Model:           {MLFLOW_REGISTERED_MODEL} (@{MODEL_CHAMPION_ALIAS})")
    print(f"Predicted Price: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()