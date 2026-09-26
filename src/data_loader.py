"""
data_loader.py
Loads the KAUST Cyber Attack Detection dataset (train.csv / test.csv) and
returns raw, untouched DataFrames. No cleaning or feature engineering here.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TARGET_COL = "Attack Type"
ID_COL = "ID"
TIMESTAMP_COL = "Timestamp"


def load_raw(raw_dir: Path = RAW_DIR):
    """Load train/test CSVs exactly as provided by Kaggle."""
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    return train, test


def get_feature_columns(df: pd.DataFrame):
    """Return the actual numeric feature columns present in the dataframe
    (excludes ID, Timestamp, and target if present)."""
    exclude = {ID_COL, TIMESTAMP_COL, TARGET_COL}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    train, test = load_raw()
    print("train:", train.shape, "test:", test.shape)
    print("feature columns:", get_feature_columns(train))
