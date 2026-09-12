import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json

def load_data(filepath):
    print("── LOADING DATA ─────────────────────────────────────")
    df = pd.read_csv(filepath)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def remove_unnecessary_columns(df):
    print("\n── REMOVING UNNECESSARY COLUMNS ─────────────────────")
    cols_to_drop = ["CustomerID"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped: {cols_to_drop}")
    return df

def handle_missing_values(df):
    print("\n── HANDLING MISSING VALUES ──────────────────────────")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: filled with median={median_val:.2f}")

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  {col}: filled with mode='{mode_val}'")

    print(f"  Total missing after imputation: {df.isnull().sum().sum()}")
    return df

def handle_outliers(df):
    print("\n── HANDLING OUTLIERS (IQR CAPPING) ──────────────────")
    skip_cols = ["ProdTaken", "Passport", "OwnCar"]
    num_cols  = [c for c in df.select_dtypes(include=["int64","float64"]).columns
                 if c not in skip_cols]
    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        if before > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"  {col}: {before} outliers capped")
    return df

def encode_categorical(df):
    print("\n── ENCODING CATEGORICAL VARIABLES ───────────────────")
    cat_cols     = df.select_dtypes(include=["object"]).columns.tolist()
    encoding_map = {}
    le           = LabelEncoder()

    for col in cat_cols:
        original = df[col].unique().tolist()
        df[col]  = le.fit_transform(df[col].astype(str))
        encoding_map[col] = {
            str(v): int(le.transform([str(v)])[0])
            for v in original
        }
        print(f"  Encoded: {col}")

    os.makedirs("models", exist_ok=True)
    with open("models/encoding_map.json", "w") as f:
        json.dump(encoding_map, f, indent=2)
    print("  Encoding map saved to models/encoding_map.json")
    return df

def split_and_save(df):
    print("\n── SPLITTING DATASET ─────────────────────────────────")
    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = X_train.copy(); train_df["ProdTaken"] = y_train
    test_df  = X_test.copy();  test_df["ProdTaken"]  = y_test

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv",   index=False)

    print(f"  Train: {train_df.shape[0]} rows → data/train.csv")
    print(f"  Test:  {test_df.shape[0]} rows  → data/test.csv")

def main():
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    df = load_data("data/tourism_data.csv")
    df = remove_unnecessary_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = encode_categorical(df)
    split_and_save(df)
    print("\n✓ DATA PREPARATION COMPLETE")

if __name__ == "__main__":
    main()
