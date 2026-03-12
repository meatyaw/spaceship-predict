import os
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ARTIFACTS_DIR = Path("artifacts")
RANDOM_STATE  = 42

CATEGORICAL_FEATURES = [
    "HomePlanet", "CryoSleep", "Destination", "VIP",
    "Deck", "Side", "Age_group",
]
NUMERICAL_FEATURES = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Cabin_num", "Group_size", "Solo", "Family_size", "TotalSpending",
    "HasSpending", "NoSpending", "Age_missing", "CryoSleep_missing",
]
SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Derive new features from raw columns."""
    df = df.copy()

    df["Deck"]      = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else "Unknown")
    df["Cabin_num"] = df["Cabin"].apply(lambda x: x.split("/")[1] if pd.notna(x) else -1).astype(float)
    df["Side"]      = df["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else "Unknown")
    df["Group"]      = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Group_size"] = df.groupby("Group")["Group"].transform("count")
    df["Solo"]       = (df["Group_size"] == 1).astype(int)
    df["LastName"]    = df["Name"].apply(lambda x: x.split()[-1] if pd.notna(x) else "Unknown")
    df["Family_size"] = df.groupby("LastName")["LastName"].transform("count")
    df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
    df["HasSpending"]   = (df["TotalSpending"] > 0).astype(int)
    df["NoSpending"]    = (df["TotalSpending"] == 0).astype(int)
    for col in SPENDING_COLS:
        df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)
    df["Age_group"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 30, 50, 100],
        labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"],
    ).astype(str)
    df["Age_missing"]       = df["Age"].isna().astype(int)
    df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

    return df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Impute, label-encode categoricals, return (X, encoders, feature_columns).
    Pass fit=False + pre-fitted encoders for transform-only mode (test set).
    """
    df = df.copy()

    ratio_cols      = [c for c in df.columns if "_ratio" in c]
    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ratio_cols

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")
    for col in NUMERICAL_FEATURES + ratio_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    X = df[feature_columns]
    return X, encoders, feature_columns

def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.
    Returns (X_train, X_val, y_train, y_val, feature_columns).
    Saves LabelEncoders to artifacts/preprocessor.pkl.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = feature_engineering(df)

    y = df["Transported"].astype(int)
    X, encoders, feature_columns = encode_features(df, fit=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    joblib.dump(encoders, ARTIFACTS_DIR / "preprocessor.pkl")
    print(f"Preprocessing done. Encoders saved → artifacts/preprocessor.pkl")
    print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Features : {len(feature_columns)}")

    return X_train, X_val, y_train, y_val, feature_columns


if __name__ == "__main__":
    from data_ingestion import ingest_data
    df = ingest_data()
    preprocess(df)
