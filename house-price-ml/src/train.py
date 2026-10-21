import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PATH,
    EXPERIMENT_NAME,
    MAE_THRESHOLD,
    MLFLOW_URI,
    MODEL_NAME,
    R2_THRESHOLD,
    RANDOM_STATE,
    TEST_SIZE,
)


def train():
    # 1. Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}. Run 'python src/ingest.py' first.")
        return

    # 2. Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # 3. Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    X = df.drop(columns=["Price"])
    y = df["Price"]

    # Convert integer columns to float64 to avoid MLflow schema enforcement warnings with missing values
    int_cols = X.select_dtypes(include=["int64", "int32"]).columns
    X[int_cols] = X[int_cols].astype("float64")

    # 4. Preprocessing: scale numbers, encode categories
    numeric_cols = ["SqFt", "Bedrooms", "Bathrooms", "Offers"]
    categorical_cols = ["Brick", "Neighborhood"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # 5. Build model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # 6. Split data using parameters from params.yaml
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 7. Train model & log with MLflow
    with mlflow.start_run(run_name="Linear Regression") as run:
        # Log input dataset to MLflow
        abs_data_path = os.path.abspath(DATA_PATH)
        dataset = mlflow.data.from_pandas(
            df,
            source=f"file://{abs_data_path}",
            targets="Price",
            name="house_prices",
        )
        mlflow.log_input(dataset, context="training")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        # Log parameters and metrics
        mlflow.log_params({"test_size": TEST_SIZE, "random_state": RANDOM_STATE})
        mlflow.log_metrics({"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)})

        # Save residual error plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(predictions, y_test - predictions, alpha=0.6, color="steelblue")
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Prediction Errors")
        plt.tight_layout()
        mlflow.log_figure(fig, "plots/residuals.png")
        plt.close(fig)

        # Register model in MLflow Registry
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="house_price_model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME,
        )

        # Quality Gate: Check if good enough to be @champion
        version = model_info.registered_model_version
        if r2 >= R2_THRESHOLD and mae <= MAE_THRESHOLD:
            alias = "champion"
            print(f" Quality Gate PASSED! Model promoted to @{alias}")
        else:
            alias = "challenger"
            print(f" Quality Gate FAILED. Model tagged as @{alias}")

        client.set_registered_model_alias(name=MODEL_NAME, alias=alias, version=version)

        print(f"\nTraining Complete!")
        print(f"Accuracy (R2):  {r2:.4f}")
        print(f"Avg Error (MAE): ${mae:,.2f}")
        print(f"Model Version:  v{version} (@{alias})")
        print(f"Tracked in MLflow: {MLFLOW_URI}")


if __name__ == "__main__":
    train()
