import warnings
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
from src.config import (
    DATA_PATH,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_TABLE,
    DB_USER,
)


def ingest():
    print(f"Connecting to PostgreSQL ({DB_HOST}:{DB_PORT}/{DB_NAME})...")

    # 1. Connect to PostgreSQL
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    # 2. Fetch all rows
    query = f'SELECT * FROM "{DB_TABLE}";'
    df = pd.read_sql(query, conn)
    conn.close()

    # 3. Save to data/house-prices.csv
    df.to_csv(DATA_PATH, index=False)
    print(f"Success! Saved {len(df)} rows to {DATA_PATH}")


if __name__ == "__main__":
    ingest()
