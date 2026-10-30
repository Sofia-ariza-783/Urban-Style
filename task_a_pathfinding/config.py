"""Central configuration for Task A - Customer Pathfinding & Store Layout.

Every script imports its paths, simulation constants and store layouts from
here so that experiments are reproducible and easy to tweak.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# The dataset is shared by all UrbanStyle tasks: look in this task's data/
# folder first, then in the repository-level data/ folder.
DATASET_PATH = next(
    (p for p in (DATA_DIR / "retail_sales_dataset.csv",
                 ROOT.parent / "data" / "retail_sales_dataset.csv") if p.exists()),
    DATA_DIR / "retail_sales_dataset.csv",
)
SYNTHETIC_DATASET_PATH = DATA_DIR / "synthetic_retail_sales_dataset.csv"
DEMAND_PROFILE_PATH = RESULTS_DIR / "demand_profile.json"

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
DEFAULT_LAYOUT = "Compact 8x6"
DEFAULT_ALGORITHM = "A*"
DEFAULT_CUSTOMERS = 60      # customers that enter the store in one run
DEFAULT_SEED = 0
MAX_TICKS = 20_000          # safety limit for one run

SPAWN_INTERVAL = 4          # a new customer enters every N ticks
DWELL_TICKS = (3, 10)       # time spent at a section / checkout (inclusive range)
FITTING_PROB = 0.30         # P(visit the fitting rooms after a clothing section)
MAX_EXTRA_STOPS = 2         # cap on additional browsing stops per customer

# Congestion-aware search: step cost = 1 + CONGESTION_ALPHA * normalised traffic.
# "Traffic" is recent occupancy: every tick  traffic = traffic * CONGESTION_DECAY
# + customers per cell, so old crowds fade away (unlike the cumulative heatmap,
# which only grows and is dominated by cells where people dwell).
CONGESTION_ALPHA = 3.0
CONGESTION_DECAY = 0.9

# Hotspots: cells whose visit count is in the top (100 - percentile)% of
# the visited cells
HOTSPOT_PERCENTILE = 90
TOP_K_HOTSPOTS = 5

# --------------------------------------------------------------------------
# Store sections
# --------------------------------------------------------------------------
# Shelf letters used in the ASCII layouts -> display name and RGB colour
SECTIONS = {
    "W": {"name": "Women's Wear",       "color": (214, 86, 142)},
    "M": {"name": "Men's Wear",         "color": (66, 115, 196)},
    "B": {"name": "Beauty & Cosmetics", "color": (230, 150, 60)},
    "T": {"name": "Accessories & Tech", "color": (60, 170, 150)},
    "F": {"name": "Fitting Rooms",      "color": (140, 100, 190)},
    "C": {"name": "Checkout",           "color": (110, 110, 120)},
}

# Dataset product category -> section where that category is sold.
# Clothing is split by gender (the dataset has a Gender column).
CATEGORY_TO_SECTION = {"Beauty": "B", "Electronics": "T"}
CLOTHING_SECTION_BY_GENDER = {"Female": "W", "Male": "M"}
CLOTHING_SECTIONS = ("W", "M")   # a visit here may be followed by the fitting rooms

# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------
# ASCII legend (row 0 = top):
#   #  wall                    .  walkable floor
#   E  entrance (floor)        X  exit (floor)
#   C  checkout counter        W M B T F  section shelves
# Everything except . E X is blocked; shelves are reached from adjacent floor.
LAYOUTS = {
    # The 8 x 6 example from the brief
    "Compact 8x6": [
        "WWW..BBB",
        "........",
        "MM.TT.FF",
        "........",
        ".....CC.",
        "E......X",
    ],
    # Parallel shelf aisles: customers move along long straight corridors
    "Grid aisles": [
        "################",
        "#..............#",
        "#.WWWWWW.MMMMM.#",
        "#..............#",
        "#.BBBBBB.TTTTT.#",
        "#..............#",
        "#.FFFF.........#",
        "#..............#",
        "#.......CCCC...#",
        "####E########X##",
    ],
    # Central island surrounded by a loop corridor
    "Racetrack loop": [
        "################",
        "#WWWWW...MMMMMM#",
        "#..............#",
        "#..BBBBBTTTTT..#",
        "#..##########..#",
        "#..##########..#",
        "#..FFFFF#####..#",
        "#..............#",
        "#........CCCC..#",
        "#E############X#",
    ],
}
