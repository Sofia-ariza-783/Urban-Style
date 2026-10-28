"""Central configuration for Task C - Dynamic Pricing with RL.

Every script imports its paths, environment constants and hyperparameters
from here so that experiments are reproducible and easy to tweak.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# The dataset is shared by all UrbanStyle tasks: look in this task's data/
# folder first, then in the repository-level data/ folder.
DATASET_PATH = next(
    (p for p in (DATA_DIR / "retail_sales_dataset.csv",
                 ROOT.parent / "data" / "retail_sales_dataset.csv") if p.exists()),
    DATA_DIR / "retail_sales_dataset.csv",
)
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_retail_sales_dataset.csv"
MARKET_PARAMS_PATH = RESULTS_DIR / "market_params.json"

# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
PRODUCT_CATEGORY = "Clothing"  # fashion retailer -> clothing is the core category

# Age bands used to build customer segments: (name, min_age, max_age) inclusive
SEGMENTS = [
    ("Young", 18, 29),
    ("Adult", 30, 49),
    ("Senior", 50, 120),
]

# Price sensitivity per segment (beta in the demand model). The dataset only
# contains a handful of fixed price points, so elasticity cannot be estimated
# from it; these values are explicit modelling assumptions (see docs).
PRICE_SENSITIVITY = {"Young": 2.0, "Adult": 1.1, "Senior": 0.65}

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
EPISODE_LENGTH = 30                                 # days per episode
PRICE_MULTIPLIERS = [0.90, 0.95, 1.00, 1.05, 1.10]  # action -> price change
MIN_PRICE_FACTOR = 0.5                              # price bounds relative to reference price
MAX_PRICE_FACTOR = 2.0
INITIAL_PRICE_RANGE = (0.7, 1.3)                    # random start price (x reference) during training
SALES_HISTORY = 3                                   # recent days of sales in the state
SEGMENT_STICKINESS = 0.85                           # P(same segment tomorrow)

# Daily customer traffic is scaled up from the dataset (1000 transactions over
# a year is ~1 unit/day for a whole category) to a single store's daily volume.
DEMAND_SCALE = 20.0

USE_INVENTORY = True
INVENTORY_BUFFER = 1.3      # initial stock = buffer * expected episode demand
DAILY_RESTOCK_FRACTION = 0.02  # fraction of initial stock restocked every day

# Shaped reward (optional, not used for the main results)
HOLDING_COST_PER_UNIT = 0.5   # per unit of leftover stock per day
STOCKOUT_PENALTY_PER_UNIT = 5.0  # per unit of unmet demand

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
TOTAL_TIMESTEPS = 200_000
N_ENVS = 4
SEEDS = [0, 1, 2]

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2,
)

DQN_KWARGS = dict(
    learning_rate=5e-4,
    buffer_size=50_000,
    learning_starts=2_000,
    batch_size=128,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.02,
)

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
EVAL_EPISODES = 100
EVAL_SEED_OFFSET = 10_000   # evaluation seeds never overlap training seeds
FIXED_PRICE_GRID = [round(0.5 + 0.05 * i, 2) for i in range(31)]  # 0.50 .. 2.00 x ref
