"""Dataset preparation and market calibration.

Pipeline:
    CSV -> clean -> filter category -> customer segments -> demand statistics
        -> results/market_params.json (consumed by the Gymnasium environment)

Run:  python -m data.prepare_data            (from task_c_dynamic_pricing/)
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import config

EXPECTED_COLUMNS = [
    "Transaction ID", "Date", "Customer ID", "Gender", "Age",
    "Product Category", "Quantity", "Price per Unit", "Total Amount",
]


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
    """Normalise column names, fix types and drop invalid rows.

    Rows with missing values are dropped rather than imputed: the dataset is
    small and has only a few discrete price points, so imputed prices or
    quantities would create artificial observations.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")

    report = {"rows_raw": len(df)}
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Age", "Quantity", "Price per Unit", "Total Amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    df["Product Category"] = df["Product Category"].astype(str).str.strip().str.title()

    report["rows_with_missing"] = int(df[EXPECTED_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=EXPECTED_COLUMNS)
    df = df[(df["Quantity"] > 0) & (df["Price per Unit"] > 0) & (df["Age"].between(0, 120))]
    df = df.drop_duplicates(subset="Transaction ID")

    # Consistency check: Total Amount must equal Price x Quantity
    recomputed = df["Price per Unit"] * df["Quantity"]
    report["inconsistent_totals_fixed"] = int((recomputed != df["Total Amount"]).sum())
    df["Total Amount"] = recomputed
    report["rows_clean"] = len(df)
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Customer segment from age bands (Young / Adult / Senior)."""
    df = df.copy()
    df["Segment"] = None
    for name, lo, hi in config.SEGMENTS:
        df.loc[df["Age"].between(lo, hi), "Segment"] = name
    return df.dropna(subset=["Segment"])


# --------------------------------------------------------------------------
# Market calibration
# --------------------------------------------------------------------------
def calibrate_market(df: pd.DataFrame, category: str = config.PRODUCT_CATEGORY) -> dict:
    """Derive the demand parameters of the simulator from the transactions."""
    cat = df[df["Product Category"] == category]
    if cat.empty:
        raise ValueError(f"No rows for category '{category}'")

    n_days = max((cat["Date"].max() - cat["Date"].min()).days + 1, 1)
    reference_price = float(cat["Price per Unit"].mean())

    # Day-of-week demand trend: units sold per weekday relative to the average
    units_by_dow = cat.groupby(cat["Date"].dt.dayofweek)["Quantity"].sum()
    units_by_dow = units_by_dow.reindex(range(7), fill_value=0)
    weekday_trend = (units_by_dow / units_by_dow.mean()).clip(0.5, 1.5)

    segments = {}
    total_tx = len(cat)
    for name, _, _ in config.SEGMENTS:
        seg = cat[cat["Segment"] == name]
        units_per_day = seg["Quantity"].sum() / n_days
        segments[name] = {
            "share": len(seg) / total_tx,
            # Willingness-to-pay anchor: what this segment paid on average
            "reference_price": float(seg["Price per Unit"].mean()),
            # Units per day if this segment were the whole daily traffic, scaled
            # to the volume of one store (see config.DEMAND_SCALE)
            "base_demand": float(units_per_day / max(len(seg) / total_tx, 1e-6) * config.DEMAND_SCALE),
            "price_sensitivity": config.PRICE_SENSITIVITY[name],
            "mean_quantity": float(seg["Quantity"].mean()),
            "female_ratio": float((seg["Gender"] == "Female").mean()),
            "n_transactions": int(len(seg)),
        }

    return {
        "category": category,
        "reference_price": reference_price,
        "price_points": sorted(cat["Price per Unit"].unique().tolist()),
        "n_days": int(n_days),
        "weekday_trend": weekday_trend.round(4).tolist(),
        "segments": segments,
    }


def prepare(category: str = config.PRODUCT_CATEGORY, save: bool = True) -> dict:
    raw, source = load_dataset()
    clean, report = clean_dataset(raw)
    clean = assign_segments(clean)
    params = calibrate_market(clean, category)
    params["source"] = source
    params["cleaning_report"] = report
    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        config.MARKET_PARAMS_PATH.write_text(json.dumps(params, indent=2))
        if source == "synthetic":
            raw.to_csv(config.SYNTHETIC_DATASET_PATH, index=False)
    return params


def load_market_params() -> dict:
    """Market parameters for the environment.

    Recomputed from the CSV every time (it takes milliseconds), so changes to
    config.py or to the dataset are always picked up; nothing is written.
    """
    return prepare(save=False)


if __name__ == "__main__":
    p = prepare()
    print(f"Source: {p['source']}  |  cleaning: {p['cleaning_report']}")
    print(f"Category: {p['category']}  reference price: {p['reference_price']:.2f}")
    print(f"Weekday trend: {p['weekday_trend']}")
    for name, s in p["segments"].items():
        print(f"  {name:<7} share={s['share']:.2f} ref_price={s['reference_price']:.1f} "
              f"base_demand={s['base_demand']:.1f}/day beta={s['price_sensitivity']}")
    print(f"Saved -> {config.MARKET_PARAMS_PATH}")
