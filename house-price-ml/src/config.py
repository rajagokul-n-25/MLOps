import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Read params.yaml if present
_params_path = BASE_DIR / "params.yaml"
_params = {}
if _params_path.exists():
    try:
        with open(_params_path, "r") as f:
            _params = yaml.safe_load(f) or {}
    except Exception:
        _params = {}

_ingest_params = _params.get("ingest", {})
_train_params = _params.get("train", {})

# Database Connection (from .env)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5433))
DB_NAME = os.getenv("DB_NAME", "mlflow")
DB_USER = os.getenv("DB_USER", "mlflow")
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "Password2026"))
DB_TABLE = _ingest_params.get("table_name", "house_prices")

# File Paths
DATA_PATH = str(BASE_DIR / "data" / "house-prices.csv")

# Training Hyperparameters
TEST_SIZE = float(_train_params.get("test_size", 0.2))
RANDOM_STATE = int(_train_params.get("random_state", 42))

# MLflow Settings
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "House Price Prediction"
MODEL_NAME = "house-price-predictor"

# Quality Gate Thresholds
R2_THRESHOLD = float(_train_params.get("r2_threshold", 0.70))
MAE_THRESHOLD = float(_train_params.get("mae_threshold", 20000.0))
