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

# Import configuration settings
try:
    from src.config import (
        DATASET_URL,
        MAE_THRESHOLD,
        MLFLOW_EXPERIMENT,
        MLFLOW_REGISTERED_MODEL,
        MLFLOW_TRACKING_URI,
        MODEL_CHALLENGER_ALIAS,
        MODEL_CHAMPION_ALIAS,
        R2_THRESHOLD,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
    )
except ModuleNotFoundError:
    from config import (
        DATASET_URL,
        MAE_THRESHOLD,
        MLFLOW_EXPERIMENT,
        MLFLOW_REGISTERED_MODEL,
        MLFLOW_TRACKING_URI,
        MODEL_CHALLENGER_ALIAS,
        MODEL_CHAMPION_ALIAS,
        R2_THRESHOLD,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
    )


def log_evaluation_plots(model, y_test, predictions):
    """Create and save helpful visual charts to MLflow."""
    # 1. Residuals plot (Difference between actual and predicted price)
    residuals = y_test - predictions
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(predictions, residuals, alpha=0.6, color="steelblue")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuals vs Predicted Price")
    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Error / Residual ($)")
    plt.tight_layout()
    mlflow.log_figure(fig, "plots/residuals.png")
    plt.close(fig)

    # 2. Feature coefficients plot (Impact of each feature)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["regressor"].coef_
    coef_series = pd.Series(coefficients, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    coef_series.plot(kind="barh", ax=ax, color="forestgreen")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Feature Coefficients (Price Impact)")
    plt.tight_layout()
    mlflow.log_figure(fig, "plots/feature_coefficients.png")
    plt.close(fig)


def check_quality_gate_and_promote(model_info, r2, mae):
    """Check if model meets accuracy thresholds to become the new Champion."""
    client = MlflowClient()
    version = model_info.registered_model_version

    # Model must have high R2 and low MAE to pass
    passed = (r2 >= R2_THRESHOLD) and (mae <= MAE_THRESHOLD)

    if passed:
        alias = MODEL_CHAMPION_ALIAS
        print(f" Quality Gate PASSED! (R2={r2:.3f}, MAE={mae:.0f})")
        print(f"   Model v{version} is now promoted to @{alias}")
    else:
        alias = MODEL_CHALLENGER_ALIAS
        print(f" Quality Gate FAILED (R2={r2:.3f}, MAE={mae:.0f})")
        print(f"   Model v{version} marked as @{alias} (will not be in production)")

    # Assign alias in MLflow registry
    client.set_registered_model_alias(
        name=MLFLOW_REGISTERED_MODEL,
        alias=alias,
        version=version,
    )
    return alias


def main():
    print("Starting model training pipeline...")

    # Connect to MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # 1. Load data
    df = pd.read_csv(DATASET_URL)
    print(f"Loaded dataset: {len(df)} rows")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 2. Build preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Train and track run in MLflow
    with mlflow.start_run(run_name="Linear Regression") as run:

        # Track dataset lineage
        dataset = mlflow.data.from_pandas(
            df, targets=TARGET_COLUMN, name="house_prices", source=DATASET_URL
        )
        mlflow.log_input(dataset, context="training")

        # Set tags for tracking
        mlflow.set_tags(
            {
                "model_type": "Linear Regression",
                "framework": "scikit-learn",
                "stage": "training",
            }
        )

        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        # Log parameters and metrics
        mlflow.log_params(
            {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "dataset_rows": len(df),
            }
        )
        mlflow.log_metrics(
            {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
            }
        )

        # Log visual diagnostic plots
        log_evaluation_plots(model, y_test, predictions)

        # Log model artifact and register it
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="house_price_model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=MLFLOW_REGISTERED_MODEL,
        )

        # Check quality gate and assign alias (@champion or @challenger)
        assigned_alias = check_quality_gate_and_promote(model_info, r2, mae)

        print("\nTraining completed successfully!")
        print(f"MAE:            ${mae:,.2f}")
        print(f"R2 Score:       {r2:.4f}")
        print(f"Run ID:         {run.info.run_id}")
        print(f"Model Version:  v{model_info.registered_model_version}")
        print(f"Status:         @{assigned_alias}")


if __name__ == "__main__":
    main()