from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "fashion_mnist"
RESULTS_ROOT = ROOT / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"
MODEL_ROOT = ROOT / "models"
SAMPLES_ROOT = FIGURES_ROOT
CHECKPOINT_ROOT = MODEL_ROOT

IMAGE_SIZE = 28
LATENT_DIM = 64
CHANNELS = 1
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 64

for directory in (DATA_ROOT, RESULTS_ROOT, FIGURES_ROOT, MODEL_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
