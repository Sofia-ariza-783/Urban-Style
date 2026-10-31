# Tarea B — Pronóstico de demanda (encoder-decoder LSTM)

Parte del proyecto UrbanStyle. Pronostica las ventas de las **próximas semanas** por
categoría con un **encoder-decoder LSTM** que usa **early stopping y regularización**. Lo
evalúa con MAE y RMSE, lo compara con un baseline de media móvil y, al exportar, genera una
**explicación escrita del pronóstico** con un modelo de lenguaje (Claude). Incluye una
interfaz en Pygame.

## Estructura

```
task_b_forecasting/
├── config.py            # Rutas e hiperparámetros
├── data_prep.py         # CSV -> series semanales por categoría
├── model.py             # Encoder-decoder LSTM (PyTorch)
├── forecasting.py       # Entrenamiento con early stopping, MAE/RMSE, media móvil, CSV
├── narrator.py          # Explicación escrita del pronóstico (Claude o reglas)
├── train.py             # Corre todas las series desde la terminal (tablas para el informe)
├── app.py               # Interfaz Pygame
├── make_sample_data.py  # Datos sintéticos para probar sin el dataset real
├── requirements.txt
├── data/                # Aquí va el CSV de Kaggle
└── outputs/             # Predicciones, métricas y explicaciones exportadas
```

## Instalación

```bash
cd task_b_forecasting
python -m venv .venv
# Windows: .venv\Scripts\activate     |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

1. Descargar el dataset: https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset
2. Copiar `retail_sales_dataset.csv` en `task_b_forecasting/data/`.

Para probar antes de tener el dataset: `python make_sample_data.py` y luego agregar
`--data data/sample_retail_sales.csv` a los comandos. Esos datos son aleatorios y no
sirven para el informe.

## Explicación con IA (opcional)

Para que la explicación la redacte Claude, define tu API key **antes** de correr la app.
Las keys se crean en https://platform.claude.com/.

```bash
# Windows PowerShell (solo para esa terminal)
$env:ANTHROPIC_API_KEY="sk-ant-..."
# Mac/Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

Si no hay key o no hay internet, se usa automáticamente un **redactor por reglas**, así
que la app nunca falla. El encabezado de cada explicación indica qué motor la generó. Para
forzar un motor, cambia `NARRATIVE_ENGINE` en `config.py` a `"llm"` o `"rules"`.

⚠️ Nunca escribas la key dentro del código ni la subas a GitHub.

## Uso

```bash
python train.py                # entrena las 6 series, imprime métricas, exporta a outputs/
python train.py --explain      # además genera una explicación por serie
python app.py                  # interfaz Pygame
```

Opciones de `train.py`: `--horizon`, `--lookback`, `--epochs`, `--patience`,
`--test-weeks`, `--history-weeks`, `--data`, `--explain`.

### Controles de la app

| Control | Función |
|---|---|
| Botones de serie | Elegir la categoría (reentrena automáticamente) |
| Slider Máx. épocas | Tope de épocas; el early stopping suele cortar antes |
| Slider Lookback | Cuántas semanas anteriores ve el encoder (2–8) |
| Slider Horizonte | Cuántas semanas pronostica el decoder (1–6) |
| Slider Semanas de historia | Cuánta historia se usa para entrenar |
| Entrenar | Aplica los valores de los sliders (aparece `*` si hay cambios pendientes) |
| Media móvil | Mostrar u ocultar el baseline |
| Exportar CSV + explicación | Guarda `predictions_<serie>.csv` y `explicacion_<serie>.md` en `outputs/` |
| Esc | Salir |

## Archivos que se generan en `outputs/`

| Archivo | Lo genera | Contenido |
|---|---|---|
| `metrics_summary.csv` | `train.py` | MAE y RMSE por serie, mejor época, total pronosticado |
| `metrics_by_horizon.csv` | `train.py` | MAE promedio a 1, 2, ..., H semanas adelante |
| `predictions_all.csv` | `train.py` | Predicciones de prueba y de las próximas semanas, todas las series |
| `predictions_<serie>.csv` | App | Lo mismo, para la serie exportada |
| `explicacion_<serie>.md` | App o `train.py --explain` | Explicación escrita del pronóstico |

## Metodología

1. **Series:** el dataset trae solo 3 categorías (Beauty, Clothing, Electronics). Para
   cumplir con las 5–8 categorías se cruza **categoría × género**, lo que da 6 series.
   Se puede cambiar en `config.SEGMENTATION`.
2. **Preprocesamiento:** se suma `Total Amount` por semana (semanas que terminan en lunes).
   Las semanas sin ventas quedan en 0.
3. **División temporal:** `[ entrenamiento | validación (6 sem) | prueba (8 sem) ]`. El
   escalado min-max se ajusta solo con el periodo de entrenamiento.
4. **Modelo encoder-decoder:**
   - El encoder LSTM lee las últimas `lookback` semanas.
   - El decoder LSTM genera `horizon` semanas, una a la vez, usando su propia predicción
     anterior como entrada.
   - Durante el entrenamiento se usa *teacher forcing* (50%): a veces el decoder recibe el
     valor real en lugar de su predicción.
5. **Regularización y early stopping:**
   - Dropout de 0.2 antes de la capa de salida y *weight decay* (L2) de 1e-4 en Adam.
   - Después de un mínimo de 20 épocas se vigila el error en validación. Si no mejora en
     30 épocas seguidas, el entrenamiento se detiene y se recupera **la mejor época**.
6. **Evaluación:**
   - En prueba se pronostica en **bloques de `horizon` semanas**, como se haría en la
     práctica, y se calculan MAE y RMSE.
   - También se reporta el MAE **por semana adelante** (1, 2, ..., H). Normalmente el
     error crece cuanto más lejos se pronostica.
7. **Baseline:** media móvil de 4 semanas, con la misma lógica de bloques.
8. **Explicación (`narrator.py`):**
   - Python calcula los datos clave: tendencia, cambio porcentual frente al promedio
     reciente, semanas pico y confiabilidad.
   - Claude redacta el texto a partir de esos datos, con instrucciones de no inventar
     cifras.
   - Los datos usados se adjuntan al final del archivo para poder verificarlos.

## Limitaciones

- Hay unas 53 semanas por serie y pocas transacciones por semana, así que las series son
  ruidosas. Es normal que la media móvil compita con el LSTM o le gane.
- La validación tiene pocas ventanas, así que la mejor época puede variar con la semilla.
- La primera semana es parcial, porque los datos empiezan a mitad de esa semana.
