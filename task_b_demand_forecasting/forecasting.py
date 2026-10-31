"""Entrenamiento con early stopping, evaluación (MAE/RMSE), baseline y pronóstico multi-semana."""
import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

import config
from model import Seq2SeqLSTM


@dataclass
class ForecastResult:
    series_name: str
    lookback: int
    horizon: int
    weeks: pd.DatetimeIndex        # semanas usadas (después de recortar historia)
    actual: np.ndarray             # valores reales de todas esas semanas
    val_start: int                 # índice donde empieza validación
    test_start: int                # índice donde empieza prueba
    test_weeks: pd.DatetimeIndex
    test_actual: np.ndarray
    test_pred: np.ndarray          # pronóstico LSTM en prueba (bloques de `horizon` semanas)
    test_baseline: np.ndarray      # pronóstico de media móvil en prueba (mismos bloques)
    future_weeks: pd.DatetimeIndex
    future_pred: np.ndarray        # próximas `horizon` semanas
    future_baseline: np.ndarray
    mae: float
    rmse: float
    baseline_mae: float
    baseline_rmse: float
    horizon_metrics: list          # MAE/RMSE por semana adelante (1..horizon)
    train_loss: list
    val_loss: list
    best_epoch: int
    epochs_run: int
    early_stopped: bool
    params: dict = field(default_factory=dict)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def set_seed(seed=config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_windows(scaled, lookback, horizon, origins):
    """Para cada origen t: entrada = semanas [t-lookback, t), objetivo = [t, t+horizon)."""
    X = np.stack([scaled[t - lookback:t] for t in origins])
    Y = np.stack([scaled[t:t + horizon] for t in origins])
    return (
        torch.tensor(X, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(Y, dtype=torch.float32),
    )


def min_history_weeks(lookback, horizon, test_weeks=config.TEST_WEEKS,
                      val_weeks=config.VAL_WEEKS, min_train_samples=8):
    """Semanas mínimas para tener `min_train_samples` ventanas de entrenamiento."""
    return lookback + horizon + val_weeks + test_weeks + min_train_samples - 1


def train_and_forecast(
    series: pd.Series,
    lookback=config.LOOKBACK,
    horizon=config.HORIZON,
    test_weeks=config.TEST_WEEKS,
    val_weeks=config.VAL_WEEKS,
    epochs=config.EPOCHS,
    patience=config.PATIENCE,
    min_epochs=config.MIN_EPOCHS,
    history_weeks=None,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    weight_decay=config.WEIGHT_DECAY,
    teacher_forcing=config.TEACHER_FORCING,
    lr=config.LEARNING_RATE,
    ma_window=config.MOVING_AVG_WINDOW,
    seed=config.SEED,
) -> ForecastResult:
    """Entrena un encoder-decoder sobre una serie semanal y lo evalúa en las últimas `test_weeks`.

    - Validación: las `val_weeks` semanas previas a la prueba; se usan para early stopping.
    - Prueba: se pronostica en bloques de `horizon` semanas, como se usaría en la práctica.
    - history_weeks: si se indica, solo se usan las últimas N semanas de la serie.
    """
    if horizon > val_weeks:
        raise ValueError(f"El horizonte ({horizon}) no puede ser mayor que VAL_WEEKS ({val_weeks}).")
    set_seed(seed)
    if history_weeks:
        series = series.iloc[-history_weeks:]

    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    test_start = n - test_weeks
    val_start = test_start - val_weeks

    train_origins = np.arange(lookback, val_start - horizon + 1)
    val_origins = np.arange(val_start, test_start - horizon + 1)
    if len(train_origins) < 4:
        raise ValueError(
            f"Muy pocas semanas ({n}) para lookback={lookback}, horizonte={horizon}. "
            f"Se necesitan al menos {min_history_weeks(lookback, horizon, test_weeks, val_weeks, 4)}."
        )

    # Escalado min-max ajustado SOLO con el periodo de entrenamiento (evita fuga de datos)
    lo, hi = values[:val_start].min(), values[:val_start].max()
    scale = hi - lo if hi > lo else 1.0
    scaled = (values - lo) / scale

    X_train, Y_train = make_windows(scaled, lookback, horizon, train_origins)
    X_val, Y_val = make_windows(scaled, lookback, horizon, val_origins)

    model = Seq2SeqLSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loss, val_loss = [], []
    best_val, best_epoch, best_state = float("inf"), 0, None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train, horizon, Y_train, teacher_forcing), Y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_val, horizon), Y_val).item()
        val_loss.append(v)

        # Las primeras épocas no cuentan: con tan pocas semanas de validación su
        # error puede ser engañosamente bajo y el modelo quedaría sin entrenar
        if epoch < min(min_epochs, epochs):
            continue
        if v < best_val:
            best_val, best_epoch = v, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    epochs_run = len(train_loss)
    model.load_state_dict(best_state)  # nos quedamos con la mejor época en validación
    model.eval()

    def predict(origins):
        X, _ = make_windows(np.append(scaled, np.zeros(horizon)), lookback, horizon, origins)
        with torch.no_grad():
            out = model(X, horizon).numpy()
        return np.clip(out * scale + lo, 0, None)  # las ventas no pueden ser negativas

    # Prueba en bloques: desde cada origen se pronostican `horizon` semanas seguidas
    block_origins = np.arange(test_start, n, horizon)
    block_preds = predict(block_origins)
    test_pred, test_baseline = [], []
    for t, pred in zip(block_origins, block_preds):
        steps = min(horizon, n - t)
        test_pred.extend(pred[:steps])
        test_baseline.extend([values[t - ma_window:t].mean()] * steps)
    test_pred, test_baseline = np.array(test_pred), np.array(test_baseline)
    test_actual = values[test_start:]

    # Error por semana adelante (h=1..horizon), con un origen en cada semana de prueba
    rolling_origins = np.arange(test_start, n)
    rolling_preds = predict(rolling_origins)
    horizon_metrics = []
    for h in range(1, horizon + 1):
        idx = [i for i, t in enumerate(rolling_origins) if t + h - 1 < n]
        if not idx:
            break
        y = np.array([values[rolling_origins[i] + h - 1] for i in idx])
        p = rolling_preds[idx, h - 1]
        b = np.array([values[rolling_origins[i] - ma_window:rolling_origins[i]].mean() for i in idx])
        horizon_metrics.append({
            "h": h, "n": len(idx),
            "mae": mae(y, p), "rmse": rmse(y, p),
            "baseline_mae": mae(y, b), "baseline_rmse": rmse(y, b),
        })

    # Pronóstico de las próximas `horizon` semanas a partir de lo último observado
    last_window = torch.tensor(scaled[n - lookback:], dtype=torch.float32).view(1, lookback, 1)
    with torch.no_grad():
        future_pred = np.clip(model(last_window, horizon).numpy()[0] * scale + lo, 0, None)
    future_baseline = np.full(horizon, values[-ma_window:].mean())

    weeks = series.index
    offset = pd.tseries.frequencies.to_offset(weeks.freq or pd.infer_freq(weeks) or config.WEEK_FREQ)
    future_weeks = pd.date_range(weeks[-1] + offset, periods=horizon, freq=offset)

    return ForecastResult(
        series_name=str(series.name),
        lookback=lookback,
        horizon=horizon,
        weeks=weeks,
        actual=values,
        val_start=val_start,
        test_start=test_start,
        test_weeks=weeks[test_start:],
        test_actual=test_actual,
        test_pred=test_pred,
        test_baseline=test_baseline,
        future_weeks=future_weeks,
        future_pred=future_pred,
        future_baseline=future_baseline,
        mae=mae(test_actual, test_pred),
        rmse=rmse(test_actual, test_pred),
        baseline_mae=mae(test_actual, test_baseline),
        baseline_rmse=rmse(test_actual, test_baseline),
        horizon_metrics=horizon_metrics,
        train_loss=train_loss,
        val_loss=val_loss,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        early_stopped=epochs_run < epochs,
        params={
            "lookback": lookback, "horizon": horizon, "max_epochs": epochs,
            "patience": patience, "min_epochs": min_epochs,
            "dropout": dropout, "weight_decay": weight_decay,
            "history_weeks": n, "test_weeks": test_weeks, "val_weeks": val_weeks,
        },
    )


def result_to_frame(result: ForecastResult) -> pd.DataFrame:
    """Tabla con el pronóstico de prueba + el de las próximas semanas (para exportar a CSV)."""
    test = pd.DataFrame({
        "series": result.series_name,
        "week": result.test_weeks,
        "type": "test",
        "step_ahead": [(i % result.horizon) + 1 for i in range(len(result.test_weeks))],
        "actual": result.test_actual,
        "lstm_pred": result.test_pred,
        "moving_avg_pred": result.test_baseline,
    })
    future = pd.DataFrame({
        "series": result.series_name,
        "week": result.future_weeks,
        "type": "forecast",
        "step_ahead": np.arange(1, result.horizon + 1),
        "actual": np.nan,
        "lstm_pred": result.future_pred,
        "moving_avg_pred": result.future_baseline,
    })
    return pd.concat([test, future], ignore_index=True)


def export_predictions(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([result_to_frame(r) for r in results], ignore_index=True)
    frame.to_csv(path, index=False, float_format="%.2f")
    return path
