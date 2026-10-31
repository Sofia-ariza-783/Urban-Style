"""Entrena y evalúa todas las series desde la terminal (útil para el informe).

Uso:
    python train.py
    python train.py --horizon 4 --lookback 6 --patience 30
    python train.py --data data/sample_retail_sales.csv
    python train.py --explain        # además genera la explicación escrita de cada serie
"""
import argparse
from pathlib import Path

import pandas as pd

import config
from data_prep import load_weekly_series
from forecasting import export_predictions, train_and_forecast


def safe_name(series_name):
    return series_name.replace(" - ", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Tarea B - Pronóstico semanal con encoder-decoder LSTM")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH,
                        help="Ruta al CSV de ventas")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Máximo de épocas")
    parser.add_argument("--patience", type=int, default=config.PATIENCE,
                        help="Épocas sin mejora en validación antes de detener")
    parser.add_argument("--lookback", type=int, default=config.LOOKBACK)
    parser.add_argument("--horizon", type=int, default=config.HORIZON,
                        help=f"Semanas a pronosticar (máx. {config.VAL_WEEKS})")
    parser.add_argument("--test-weeks", type=int, default=config.TEST_WEEKS)
    parser.add_argument("--history-weeks", type=int, default=None,
                        help="Usar solo las últimas N semanas (por defecto, todas)")
    parser.add_argument("--explain", action="store_true",
                        help="Generar explicación escrita (Claude o reglas) por serie")
    args = parser.parse_args()

    weekly = load_weekly_series(args.data)
    print(f"Datos: {len(weekly)} semanas, {weekly.shape[1]} series "
          f"({weekly.index.min():%Y-%m-%d} a {weekly.index.max():%Y-%m-%d})")
    print(f"Horizonte: {args.horizon} semanas | lookback: {args.lookback} | "
          f"máx. épocas: {args.epochs} | paciencia: {args.patience}\n")

    results, rows, horizon_rows = [], [], []
    for name in weekly.columns:
        r = train_and_forecast(
            weekly[name],
            lookback=args.lookback,
            horizon=args.horizon,
            test_weeks=args.test_weeks,
            epochs=args.epochs,
            patience=args.patience,
            history_weeks=args.history_weeks,
        )
        results.append(r)
        rows.append({
            "Serie": name,
            "MAE LSTM": r.mae,
            "RMSE LSTM": r.rmse,
            "MAE Media móvil": r.baseline_mae,
            "RMSE Media móvil": r.baseline_rmse,
            "Mejor época": r.best_epoch,
            "Épocas": r.epochs_run,
            f"Total próximas {r.horizon} sem": r.future_pred.sum(),
        })
        for m in r.horizon_metrics:
            horizon_rows.append({"Serie": name, "Semanas adelante": m["h"],
                                 "MAE LSTM": m["mae"], "MAE Media móvil": m["baseline_mae"]})

    summary = pd.DataFrame(rows).set_index("Serie")
    pd.set_option("display.width", 220)
    print(summary.round(1).to_string())
    print("\nPromedio:")
    print(summary.iloc[:, :4].mean().round(1).to_string())

    by_horizon = (pd.DataFrame(horizon_rows)
                  .groupby("Semanas adelante")[["MAE LSTM", "MAE Media móvil"]].mean())
    print("\nMAE promedio según cuántas semanas adelante se pronostica:")
    print(by_horizon.round(1).to_string())

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.OUTPUT_DIR / "metrics_summary.csv", float_format="%.2f")
    by_horizon.to_csv(config.OUTPUT_DIR / "metrics_by_horizon.csv", float_format="%.2f")
    path = export_predictions(results, config.OUTPUT_DIR / "predictions_all.csv")
    print(f"\nGuardado: {config.OUTPUT_DIR / 'metrics_summary.csv'}")
    print(f"Guardado: {config.OUTPUT_DIR / 'metrics_by_horizon.csv'}")
    print(f"Guardado: {path}")

    if args.explain:
        from narrator import write_narrative
        print()
        for r in results:
            out, engine = write_narrative(
                r, config.OUTPUT_DIR / f"explicacion_{safe_name(r.series_name)}.md")
            print(f"Explicación ({engine}): {out}")


if __name__ == "__main__":
    main()
