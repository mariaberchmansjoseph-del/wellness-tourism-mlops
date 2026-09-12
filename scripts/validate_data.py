import pandas as pd
import sys
import os

def validate_dataset(filepath):
    print("=" * 60)
    print("DATA REGISTRATION AND VALIDATION")
    print("=" * 60)

    # ── CHECK FILE EXISTS ──────────────────────────────────────
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        sys.exit(1)
    print(f"✓ File found: {filepath}")

    # ── LOAD DATASET ───────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── EXPECTED COLUMNS ───────────────────────────────────────
    expected = [
        "CustomerID", "ProdTaken", "Age", "TypeofContact",
        "CityTier", "Occupation", "Gender",
        "NumberOfPersonVisiting", "PreferredPropertyStar",
        "MaritalStatus", "NumberOfTrips", "Passport",
        "OwnCar", "NumberOfChildrenVisiting", "Designation",
        "MonthlyIncome", "PitchSatisfactionScore",
        "ProductPitched", "NumberOfFollowups", "DurationOfPitch"
    ]

    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    else:
        print(f"✓ All {len(expected)} expected columns present")

    # ── SUMMARY ────────────────────────────────────────────────
    print(f"\nDataset Summary:")
    print(f"  Rows:           {df.shape[0]:,}")
    print(f"  Columns:        {df.shape[1]}")
    print(f"  Duplicates:     {df.duplicated().sum()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    print(f"\nTarget Variable Distribution:")
    counts = df["ProdTaken"].value_counts()
    for val, count in counts.items():
        label = "Purchased" if val == 1 else "Not Purchased"
        print(f"  {label} ({val}): {count:,} ({count/len(df)*100:.1f}%)")

    print(f"\nMissing Values Per Column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  No missing values found.")
    else:
        for col, cnt in missing.items():
            print(f"  {col}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("✓ VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    validate_dataset("data/tourism_data.csv")
