"""
Spaceship Titanic – Step 1: Data Ingestion
Reads raw train.csv and returns a DataFrame.
"""

from pathlib import Path
import pandas as pd

BASE_DIR   = Path(__file__).parent
INPUT_FILE = BASE_DIR / "train.csv"


def ingest_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"train.csv not found at {INPUT_FILE}. "
            "Please download it from the Kaggle competition page."
        )

    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "Dataset is empty"
    print(f"Data ingested : {INPUT_FILE}  |  shape={df.shape}")
    return df


if __name__ == "__main__":
    ingest_data()
