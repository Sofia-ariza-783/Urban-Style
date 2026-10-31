"""Convierte un pronóstico en una explicación escrita para el negocio.

Flujo:
    1. build_facts(): extrae del resultado los datos clave (tendencia, cambios %, errores...).
       Todo lo numérico se calcula aquí, en Python, para que el texto no invente cifras.
    2. Un modelo de lenguaje (Claude) redacta la explicación a partir de esos datos.
       Si no hay API key, o falla la conexión, se usa un redactor por reglas.
    3. write_narrative(): guarda el texto en outputs/ junto al CSV exportado.
"""
import json
import os
from datetime import datetime

import numpy as np

import config

SYSTEM_PROMPT = """Eres un analista de demanda de UrbanStyle, un retailer de moda. Recibes en JSON \
el resultado de un modelo de pronóstico (encoder-decoder LSTM) para una línea de producto, y \
redactas para el equipo de compras e inventario una explicación clara en español.

Reglas:
- Usa solo las cifras del JSON. No inventes datos, eventos ni causas; si propones una posible \
causa (p. ej. temporada navideña), preséntala como hipótesis.
- Las cifras son ventas semanales en dólares (columna Total Amount).
- Sé honesto con la confiabilidad: compara el error del modelo (MAE) con el promedio semanal \
y con el baseline de media móvil. Si el baseline fue mejor, dilo.
- Escribe en Markdown, entre 250 y 450 palabras, con exactamente estas secciones:
## Resumen
## Comportamiento esperado en las próximas semanas
## Cómo se llegó a estas conclusiones
## Qué tan confiable es este pronóstico
## Recomendación para inventario
- En "Cómo se llegó a estas conclusiones" explica el razonamiento paso a paso citando las \
cifras concretas que lo sustentan (historia reciente, tendencia, comparación con el promedio, \
métricas)."""


def _money(v):
    return f"${v:,.0f}"


def _pct(a, b):
    return float((a - b) / b * 100) if b else 0.0


def build_facts(result):
    """Resume el resultado del modelo en un diccionario simple y verificable."""
    recent_n = min(8, len(result.actual))
    recent_avg = float(result.actual[-4:].mean())
    hist_avg = float(result.actual.mean())
    future = result.future_pred
    slope = float(np.polyfit(np.arange(len(future)), future, 1)[0]) if len(future) > 1 else 0.0
    rel_slope = slope / recent_avg if recent_avg else 0.0
    trend = "al alza" if rel_slope > 0.05 else "a la baja" if rel_slope < -0.05 else "estable"
    future_avg = float(future.mean())

    return {
        "serie": result.series_name,
        "variable": f"{config.TARGET_COLUMN} semanal (USD)",
        "horizonte_semanas": result.horizon,
        "historia": {
            "semanas_usadas": len(result.actual),
            "promedio_semanal_historico": round(hist_avg, 2),
            "promedio_ultimas_4_semanas": round(recent_avg, 2),
            "ultimas_semanas": [
                {"semana": f"{w:%Y-%m-%d}", "venta": round(float(v), 2)}
                for w, v in zip(result.weeks[-recent_n:], result.actual[-recent_n:])
            ],
        },
        "pronostico": [
            {
                "semana": f"{w:%Y-%m-%d}",
                "lstm": round(float(p), 2),
                "media_movil": round(float(b), 2),
                "cambio_vs_promedio_ultimas_4_pct": round(_pct(p, recent_avg), 1),
            }
            for w, p, b in zip(result.future_weeks, future, result.future_baseline)
        ],
        "resumen_pronostico": {
            "total_horizonte": round(float(future.sum()), 2),
            "promedio_semanal": round(future_avg, 2),
            "cambio_promedio_vs_ultimas_4_pct": round(_pct(future_avg, recent_avg), 1),
            "tendencia": trend,
            "pendiente_por_semana": round(slope, 2),
            "semana_mas_alta": f"{result.future_weeks[int(future.argmax())]:%Y-%m-%d}",
            "semana_mas_baja": f"{result.future_weeks[int(future.argmin())]:%Y-%m-%d}",
        },
        "evaluacion_en_prueba": {
            "semanas_prueba": len(result.test_actual),
            "mae_lstm": round(result.mae, 2),
            "rmse_lstm": round(result.rmse, 2),
            "mae_media_movil": round(result.baseline_mae, 2),
            "rmse_media_movil": round(result.baseline_rmse, 2),
            "mae_como_pct_del_promedio_semanal": round(result.mae / hist_avg * 100, 1) if hist_avg else None,
            "mejor_modelo": "LSTM" if result.mae < result.baseline_mae else "media movil",
            "error_por_semana_adelante": [
                {"semanas_adelante": m["h"], "mae_lstm": round(m["mae"], 2),
                 "mae_media_movil": round(m["baseline_mae"], 2)}
                for m in result.horizon_metrics
            ],
        },
        "entrenamiento": {
            "modelo": "Encoder-decoder LSTM",
            "lookback_semanas": result.lookback,
            "mejor_epoca": result.best_epoch,
            "epocas_ejecutadas": result.epochs_run,
            "early_stopping_activado": result.early_stopped,
        },
    }


# ----------------------------------------------------------------- redactor con Claude
def _llm_available():
    if config.NARRATIVE_ENGINE == "rules":
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if config.NARRATIVE_ENGINE == "llm":
        return True
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def llm_narrative(facts):
    import anthropic

    client = anthropic.Anthropic(timeout=120)
    response = client.beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=16000,
        output_config={"effort": config.LLM_EFFORT},
        # Si el modelo declina la solicitud, la API reintenta con otro modelo automáticamente
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": "Datos del pronóstico:\n```json\n"
                       + json.dumps(facts, ensure_ascii=False, indent=2) + "\n```",
        }],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("El modelo declinó generar la explicación.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("Respuesta vacía del modelo.")
    return text


# ----------------------------------------------------------------- redactor por reglas
def rule_based_narrative(facts):
    s, ev, hist = facts["resumen_pronostico"], facts["evaluacion_en_prueba"], facts["historia"]
    fc = facts["pronostico"]
    h = facts["horizonte_semanas"]
    change = s["cambio_promedio_vs_ultimas_4_pct"]
    direction = "por encima" if change > 0 else "por debajo"
    lstm_wins = ev["mejor_modelo"] == "LSTM"
    err_pct = ev["mae_como_pct_del_promedio_semanal"]
    reliability = ("alta" if err_pct is not None and err_pct < 25 else
                   "moderada" if err_pct is not None and err_pct < 50 else "baja")

    lines = [
        "## Resumen",
        f"Para **{facts['serie']}** se esperan ventas de **{_money(s['total_horizonte'])}** en las "
        f"próximas {h} semanas (≈ {_money(s['promedio_semanal'])} por semana), un "
        f"{abs(change):.1f}% {direction} del promedio de las últimas 4 semanas. "
        f"Dentro de ese horizonte, la trayectoria semana a semana es **{s['tendencia']}**.",
        "",
        "## Comportamiento esperado en las próximas semanas",
    ]
    for f in fc:
        lines.append(f"- Semana {f['semana']}: {_money(f['lstm'])} "
                     f"({f['cambio_vs_promedio_ultimas_4_pct']:+.1f}% vs. promedio reciente).")
    lines += [
        f"\nLa semana más alta sería la del {s['semana_mas_alta']} y la más baja la del "
        f"{s['semana_mas_baja']}.",
        "",
        "## Cómo se llegó a estas conclusiones",
        f"1. Se tomaron las últimas {facts['entrenamiento']['lookback_semanas']} semanas de ventas "
        f"como entrada del modelo encoder-decoder, que genera {h} semanas hacia adelante.",
        f"2. El promedio de las últimas 4 semanas fue {_money(hist['promedio_ultimas_4_semanas'])} "
        f"y el histórico {_money(hist['promedio_semanal_historico'])}; el pronóstico se compara "
        "contra esa referencia.",
        f"3. La pendiente del pronóstico es {_money(s['pendiente_por_semana'])} por semana, por lo "
        f"que la trayectoria se clasifica como {s['tendencia']} (umbral: ±5% del promedio reciente "
        "por semana). Esto es distinto del nivel: el pronóstico puede ser estable pero estar "
        "por encima o por debajo de lo vendido recientemente.",
        f"4. El modelo se entrenó hasta la época {facts['entrenamiento']['epocas_ejecutadas']} y se "
        f"conservó la época {facts['entrenamiento']['mejor_epoca']}, la de menor error en validación "
        "(early stopping).",
        "",
        "## Qué tan confiable es este pronóstico",
        f"En las {ev['semanas_prueba']} semanas de prueba el LSTM tuvo un MAE de "
        f"{_money(ev['mae_lstm'])} (≈ {err_pct}% del promedio semanal), frente a "
        f"{_money(ev['mae_media_movil'])} de la media móvil. La confiabilidad se considera "
        f"**{reliability}**"
        + (" y el LSTM superó al baseline." if lstm_wins else
           ", y la media móvil fue más precisa, así que conviene usar ambos como referencia."),
    ]
    if ev["error_por_semana_adelante"]:
        first, last = ev["error_por_semana_adelante"][0], ev["error_por_semana_adelante"][-1]
        lines.append(f"El error pasa de {_money(first['mae_lstm'])} a 1 semana a "
                     f"{_money(last['mae_lstm'])} a {last['semanas_adelante']} semanas.")
    lines += [
        "",
        "## Recomendación para inventario",
        ("Ajustar las compras al alza de forma gradual" if s["tendencia"] == "al alza" else
         "Moderar las compras y vigilar el inventario" if s["tendencia"] == "a la baja" else
         "Mantener el nivel de compras actual")
        + f", con un margen de seguridad cercano al MAE ({_money(ev['mae_lstm'])} por semana) "
        "para cubrir la incertidumbre del pronóstico.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------- orquestación
def generate_narrative(result):
    """Devuelve (texto_markdown, motor_usado, datos_usados)."""
    facts = build_facts(result)
    note = ""
    if _llm_available():
        try:
            return llm_narrative(facts), f"Claude ({config.LLM_MODEL})", facts
        except Exception as exc:  # sin internet, key inválida, límite de uso, etc.
            note = f" — no se pudo usar Claude: {type(exc).__name__}: {exc}"
    return rule_based_narrative(facts), "Reglas (sin LLM)" + note, facts


def write_narrative(result, path):
    text, engine, facts = generate_narrative(result)
    header = (
        f"# Explicación del pronóstico — {result.series_name}\n\n"
        f"*Generado: {datetime.now():%Y-%m-%d %H:%M} · Motor: {engine}*\n\n"
    )
    appendix = (
        "\n\n---\n\n<details><summary>Datos usados para esta explicación</summary>\n\n"
        "```json\n" + json.dumps(facts, ensure_ascii=False, indent=2) + "\n```\n</details>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + text + appendix, encoding="utf-8")
    return path, engine
