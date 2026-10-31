"""Genera un CSV SINTÉTICO con las mismas columnas del dataset de Kaggle.

Solo sirve para probar el código mientras se descarga el dataset real.
NO usar estos datos en el informe final.

Uso:
    python make_sample_data.py
"""
import numpy as np
import pandas as pd

import config

CATEGORIES = ["Beauty", "Clothing", "Electronics"]
PRICES = [25, 30, 50, 300, 500]


def main(n_rows=1000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    # Estacionalidad suave: más ventas a fin de año
    weights = 1 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, len(days)) - np.pi / 2) ** 2
    weights[-45:] *= 1.6
    weights /= weights.sum()

    dates = rng.choice(days, size=n_rows, p=weights)
    qty = rng.integers(1, 5, size=n_rows)
    price = rng.choice(PRICES, size=n_rows)
    df = pd.DataFrame({
        "Transaction ID": np.arange(1, n_rows + 1),
        "Date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "Customer ID": [f"CUST{i:03d}" for i in range(1, n_rows + 1)],
        "Gender": rng.choice(["Male", "Female"], size=n_rows),
        "Age": rng.integers(18, 65, size=n_rows),
        "Product Category": rng.choice(CATEGORIES, size=n_rows),
        "Quantity": qty,
        "Price per Unit": price,
        "Total Amount": qty * price,
    }).sort_values("Date")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DATA_DIR / "sample_retail_sales.csv"
    df.to_csv(out, index=False)
    print(f"CSV sintético guardado en {out}")
    print(f"Pruébalo con:  python train.py --data {out.relative_to(config.BASE_DIR)}")
    print(f"           o:  python app.py --data {out.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
