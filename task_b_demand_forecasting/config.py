"""Configuración central de la Tarea B (pronóstico de demanda)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# CSV de Kaggle: https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset
# Descárgalo y déjalo en data/retail_sales_dataset.csv
DATA_PATH = DATA_DIR / "retail_sales_dataset.csv"

# El dataset solo trae 3 categorías (Beauty, Clothing, Electronics).
# Para cumplir con "5-8 categorías" se cruza categoría x género -> 6 series.
# Opciones: "category" (3 series), "category_gender" (6), "category_age" (9).
SEGMENTATION = "category_gender"

# Grupos de edad usados cuando SEGMENTATION = "category_age"
AGE_BINS = [0, 30, 45, 120]
AGE_LABELS = ["18-30", "31-45", "46+"]

# Variable a pronosticar: "Total Amount" (ingresos) o "Quantity" (unidades)
TARGET_COLUMN = "Total Amount"

# Frecuencia semanal; W-MON = semanas que terminan en lunes
WEEK_FREQ = "W-MON"

# División temporal: [ entrenamiento | validación | prueba ]
TEST_WEEKS = 8        # últimas semanas reservadas para evaluar
VAL_WEEKS = 6         # semanas antes de la prueba, usadas para early stopping

# Encoder-decoder
LOOKBACK = 4          # semanas de historia que ve el encoder
HORIZON = 4           # semanas que pronostica el decoder (debe ser <= VAL_WEEKS)
HIDDEN_SIZE = 32
NUM_LAYERS = 1
TEACHER_FORCING = 0.5  # prob. de alimentar al decoder con el valor real durante el entrenamiento

# Entrenamiento y regularización
EPOCHS = 500          # máximo de épocas (early stopping suele cortar antes)
PATIENCE = 30         # épocas sin mejorar en validación antes de detenerse
MIN_EPOCHS = 20       # épocas mínimas antes de empezar a vigilar la validación
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4   # regularización L2 en Adam
LEARNING_RATE = 0.01

MOVING_AVG_WINDOW = 4  # ventana del baseline de media móvil
SEED = 42

# Explicación automática del pronóstico (narrator.py)
# "auto": usa Claude si hay ANTHROPIC_API_KEY, si no, reglas. "llm" o "rules" para forzar.
NARRATIVE_ENGINE = "auto"
LLM_MODEL = "claude-opus-5-5"
LLM_EFFORT = "medium"
