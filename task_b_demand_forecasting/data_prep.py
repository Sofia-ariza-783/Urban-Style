"""Carga del CSV de ventas y conversión a series de tiempo semanales."""
import pandas as pd

import config

REQUIRED_COLUMNS = ["Date", "Gender", "Age", "Product Category", "Quantity", "Total Amount"]


def load_transactions(path=config.DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}.\n"
            "Descárgalo de Kaggle (retail-sales-dataset) y colócalo ahí, "
            "o genera datos de prueba con: python make_sample_data.py"
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_series_key(df, segmentation=config.SEGMENTATION):
    """Agrega la columna 'Series' que define cada serie a pronosticar."""
    df = df.copy()
    if segmentation == "category":
        df["Series"] = df["Product Category"]
    elif segmentation == "category_gender":
        df["Series"] = df["Product Category"] + " - " + df["Gender"]
    elif segmentation == "category_age":
        age_group = pd.cut(df["Age"], bins=config.AGE_BINS, labels=config.AGE_LABELS)
        df["Series"] = df["Product Category"] + " - " + age_group.astype(str)
    else:
        raise ValueError(f"Segmentación no válida: {segmentation}")
    return df


def build_weekly_series(df, target=config.TARGET_COLUMN, freq=config.WEEK_FREQ):
    """Devuelve un DataFrame con índice semanal y una columna por serie.

    Las semanas sin ventas quedan en 0 para que la serie sea continua.
    """
    df = add_series_key(df)
    weekly = (
        df.groupby(["Series", pd.Grouper(key="Date", freq=freq)])[target]
        .sum()
        .unstack("Series")
    )
    full_index = pd.date_range(weekly.index.min(), weekly.index.max(), freq=freq)
    weekly = weekly.reindex(full_index).fillna(0.0)
    weekly.index.name = "Week"
    return weekly.sort_index(axis=1)


def load_weekly_series(path=config.DATA_PATH):
    return build_weekly_series(load_transactions(path))


if __name__ == "__main__":
    weekly = load_weekly_series()
    print(f"Semanas: {len(weekly)}  |  Series: {weekly.shape[1]}")
    print(weekly.describe().round(1).T)
