"""Dataset preparation and customer demand profile.

Pipeline:
    CSV -> clean -> demand statistics (category / gender / quantity)
        -> results/demand_profile.json (consumed by the store simulation)

The simulation samples its customers from the real transactions, so the
mix of sections people want to visit follows the retailer's sales data.

Run:  python -m data.prepare_data            (from task_a_pathfinding/)
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

import config

EXPECTED_COLUMNS = [
    "Transaction ID", "Date", "Customer ID", "Gender", "Age",
    "Product Category", "Quantity", "Price per Unit", "Total Amount",
]
PROFILE_COLUMNS = ["Gender", "Age", "Product Category", "Quantity"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def generate_synthetic_dataset(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a dataset with the same schema and distributions as the Kaggle
    'Retail Sales Dataset', used only when the real CSV is not available."""
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, n_rows), unit="D")
    quantity = rng.integers(1, 5, n_rows)
    price = rng.choice([25, 30, 50, 300, 500], n_rows)
    return pd.DataFrame({
        "Transaction ID": np.arange(1, n_rows + 1),
        "Date": dates.strftime("%Y-%m-%d"),
        "Customer ID": [f"CUST{i:03d}" for i in range(1, n_rows + 1)],
        "Gender": rng.choice(["Male", "Female"], n_rows),
        "Age": rng.integers(18, 65, n_rows),
        "Product Category": rng.choice(["Beauty", "Clothing", "Electronics"], n_rows),
        "Quantity": quantity,
        "Price per Unit": price,
        "Total Amount": quantity * price,
    })


def load_dataset(path=config.DATASET_PATH) -> tuple[pd.DataFrame, str]:
    """Load the retail CSV, falling back to a synthetic dataset if missing."""
    if path.exists():
        return pd.read_csv(path), "kaggle"
    warnings.warn(
        f"{path} not found - using a synthetic dataset with the same schema. "
        "Download the Kaggle 'Retail Sales Dataset' into data/ for real results."
    )
    return generate_synthetic_dataset(), "synthetic"


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise column names, fix types and drop invalid rows."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")

    report = {"rows_raw": len(df)}
    for col in ["Age", "Quantity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    df["Product Category"] = df["Product Category"].astype(str).str.strip().str.title()

    df = df.dropna(subset=PROFILE_COLUMNS)
    df = df[(df["Quantity"] > 0) & (df["Age"].between(0, 120))]
    df = df.drop_duplicates(subset="Transaction ID")
    df[["Age", "Quantity"]] = df[["Age", "Quantity"]].astype(int)
    report["rows_clean"] = len(df)
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Demand profile
# --------------------------------------------------------------------------
def build_demand_profile(df: pd.DataFrame, source: str) -> dict:
    """Shares that drive the simulated shopping behaviour."""
    category_share = df["Product Category"].value_counts(normalize=True).sort_index()
    gender_share = (df.groupby("Product Category")["Gender"]
                    .value_counts(normalize=True).unstack(fill_value=0.0))
    quantity_share = df["Quantity"].value_counts(normalize=True).sort_index()
    return {
        "source": source,
        "n_transactions": int(len(df)),
        "category_share": category_share.round(4).to_dict(),
        "gender_share_by_category": gender_share.round(4).to_dict(orient="index"),
        "quantity_distribution": {str(q): round(p, 4) for q, p in quantity_share.items()},
    }


@lru_cache(maxsize=1)
def _prepared() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(raw, clean, profile), computed once per process: the dataset is tiny and static."""
    raw, source = load_dataset()
    clean, report = clean_dataset(raw)
    profile = build_demand_profile(clean, source)
    profile["cleaning_report"] = report
    return raw, clean, profile


def prepare(save: bool = True) -> dict:
    """Compute the demand profile and (optionally) write it to results/."""
    raw, _, profile = _prepared()
    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        config.DEMAND_PROFILE_PATH.write_text(json.dumps(profile, indent=2))
        if profile["source"] == "synthetic":
            raw.to_csv(config.SYNTHETIC_DATASET_PATH, index=False)
    return profile


def load_demand_profile() -> dict:
    """Demand profile for the simulation (recomputed from the CSV, nothing written)."""
    return prepare(save=False)


def load_customer_profiles() -> pd.DataFrame:
    """One row per real transaction (Gender, Age, Product Category, Quantity);
    the simulation samples customers from these rows."""
    _, clean, _ = _prepared()
    return clean[PROFILE_COLUMNS].copy()


if __name__ == "__main__":
    p = prepare()
    print(f"Source: {p['source']}  |  transactions: {p['n_transactions']}  |  cleaning: {p['cleaning_report']}")
    print(f"Category share: {p['category_share']}")
    for category, shares in p["gender_share_by_category"].items():
        print(f"  {category:<12} gender share: {shares}")
    print(f"Quantity distribution: {p['quantity_distribution']}")
    print(f"Saved -> {config.DEMAND_PROFILE_PATH}")
