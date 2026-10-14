import os

from dotenv import load_dotenv

load_dotenv()

DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "ywchiu/riii/master/data/house-prices.csv"
)

TARGET_COLUMN = "Price"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = "House Price Prediction Production"
MLFLOW_REGISTERED_MODEL = "house-price-predictor"
MODEL_CHAMPION_ALIAS = "champion"
MODEL_CHALLENGER_ALIAS = "challenger"

# Quality gate thresholds — model must pass both to become champion:
R2_THRESHOLD = 0.70
MAE_THRESHOLD = 20000.0
