"""Compare layouts and search algorithms over several random seeds.

Run:  python -m evaluation.experiments [--customers 60] [--seeds 0 1 2]     (from task_a_pathfinding/)
Outputs:
    results/comparison.csv      mean and std of every metric, one row per layout x algorithm
    results/metrics.json        the same metrics + top hotspots (seed 0) with section context
    results/figures/*.png       heatmaps.png, nodes_expanded.png, layout_comparison.png

Every layout x algorithm x seed is one full simulation. For a given seed all
algorithms see the same customers, so differences come from the algorithms.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from search.algorithms import ALGORITHMS  # noqa: E402
from simulation.store_sim import StoreSimulation  # noqa: E402

# Metric key -> (label used in tables and plots)
METRICS = {
    "avg_nodes_per_search": "Avg nodes expanded / search",
    "total_nodes": "Total nodes expanded",
    "avg_path_length": "Avg path length (steps)",
    "avg_trip_ticks": "Avg trip time (ticks)",
    "ticks_to_empty": "Ticks until store empty",
    "peak_heat": "Peak cell heat",
    "mean_heat": "Mean cell heat",
    "hotspot_share": "Hotspot share of floor",
}

# Fixed colour per algorithm (same palette as Task C), shared by all figures
ALGO_COLORS = {"BFS": "#2a78d6", "A*": "#eb6834", "Congestion-aware": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
WALL_COLOR = "#3b3b3f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
})


# --------------------------------------------------------------------------
# Running the experiments
# --------------------------------------------------------------------------
def run_once(layout: str, algorithm: str, n_customers: int, seed: int) -> tuple[dict, StoreSimulation]:
    sim = StoreSimulation(layout, algorithm, n_customers=n_customers, seed=seed)
    if not sim.run():
        raise RuntimeError(f"{layout} / {algorithm} / seed {seed} did not finish")
    s = sim.stats()
    metrics = {
        "avg_nodes_per_search": s["avg_nodes_expanded_per_search"],
        "total_nodes": s["total_nodes_expanded"],
        "avg_path_length": s["avg_path_length"],
        "avg_trip_ticks": s["avg_trip_ticks"],
        "ticks_to_empty": s["ticks"],
        "peak_heat": s["peak_heat"],
        "mean_heat": s["mean_heat"],
        "hotspot_share": float(sim.hotspot_mask().sum() / sim.layout.walkable_mask().sum()),
    }
    return metrics, sim


def run_experiments(n_customers: int, seeds: list[int]) -> tuple[dict, dict, dict]:
    """Returns (raw[layout][algo] = list of per-seed metrics,
                hotspots[layout][algo] = top hotspots of the first seed,
                sims[layout][algo] = the first seed's finished simulation)."""
    raw: dict = {}
    hotspots: dict = {}
    sims: dict = {}
    for layout in config.LAYOUTS:
        for algorithm in ALGORITHMS:
            runs = []
            for i, seed in enumerate(seeds):
                metrics, sim = run_once(layout, algorithm, n_customers, seed)
                runs.append(metrics)
                if i == 0:
                    sims.setdefault(layout, {})[algorithm] = sim
                    hotspots.setdefault(layout, {})[algorithm] = [
                        {"cell": list(cell), "count": count, "context": sim.layout.describe(cell)}
                        for cell, count in sim.hotspots(config.TOP_K_HOTSPOTS)
                    ]
            raw.setdefault(layout, {})[algorithm] = runs
    return raw, hotspots, sims


def aggregate(runs: list[dict]) -> dict:
    """mean and sample standard deviation over seeds, per metric."""
    out = {}
    for key in METRICS:
        values = np.array([r[key] for r in runs], dtype=float)
        out[key] = {"mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def comparison_table(summary: dict) -> pd.DataFrame:
    rows = []
    for layout, by_algo in summary.items():
        for algorithm, metrics in by_algo.items():
            row = {"layout": layout, "algorithm": algorithm}
            for key in METRICS:
                row[f"{key}_mean"] = round(metrics[key]["mean"], 4)
                row[f"{key}_std"] = round(metrics[key]["std"], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary: dict, hotspots: dict) -> None:
    def cell(m, key, fmt):
        return f"{m[key]['mean']:{fmt}} +/-{m[key]['std']:{fmt}}"

    header = (f"{'Layout':<15}{'Algorithm':<18}{'nodes/search':>16}{'path len':>14}"
              f"{'trip ticks':>16}{'ticks':>16}{'peak heat':>16}{'hotspot %':>14}")
    print(header)
    print("-" * len(header))
    for layout, by_algo in summary.items():
        for algorithm, m in by_algo.items():
            share = {"mean": 100 * m["hotspot_share"]["mean"], "std": 100 * m["hotspot_share"]["std"]}
            print(f"{layout:<15}{algorithm:<18}{cell(m, 'avg_nodes_per_search', '.1f'):>16}"
                  f"{cell(m, 'avg_path_length', '.2f'):>14}{cell(m, 'avg_trip_ticks', '.1f'):>16}"
                  f"{cell(m, 'ticks_to_empty', '.0f'):>16}{cell(m, 'peak_heat', '.0f'):>16}"
                  f"{share['mean']:>8.1f} +/-{share['std']:.1f}")
    print("\nTop-3 hotspots (seed 0, A*):")
    for layout, by_algo in hotspots.items():
        top = ", ".join(f"{tuple(h['cell'])} {h['count']} [{h['context']}]" for h in by_algo["A*"][:3])
        print(f"  {layout:<15}{top}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def draw_store(ax, sim: StoreSimulation, cmap, outline_hotspots: bool = True) -> None:
    """Floor coloured by heat (share of the peak), shelves in section colours."""
    layout = sim.layout
    peak = max(sim.heat.max(), 1)
    image = np.zeros((layout.height, layout.width, 3))
    for r in range(layout.height):
        for c in range(layout.width):
            ch = layout.rows[r][c]
            if layout.walkable((r, c)):
                image[r, c] = cmap(sim.heat[r, c] / peak)[:3]
            elif ch in config.SECTIONS:
                image[r, c] = np.array(config.SECTIONS[ch]["color"]) / 255
            else:
                image[r, c] = matplotlib.colors.to_rgb(WALL_COLOR)
    ax.imshow(image, interpolation="nearest")

    for code, shelves in layout.sections.items():
        for r, c in shelves:
            ax.text(c, r, code, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    for label, (r, c) in (("E", layout.entrance), ("X", layout.exit)):
        ax.text(c, r, label, ha="center", va="center", color=INK, fontsize=8, fontweight="bold")
    if outline_hotspots:
        for r, c in np.argwhere(sim.hotspot_mask()):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_heatmaps(sims: dict, path) -> None:
    layouts, algos = list(config.LAYOUTS), list(ALGORITHMS)
    cmap = plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(len(layouts), len(algos), figsize=(14, 3.6 * len(layouts)),
                             constrained_layout=True)
    for i, layout in enumerate(layouts):
        for j, algorithm in enumerate(algos):
            ax = axes[i, j]
            draw_store(ax, sims[layout][algorithm], cmap)
            if i == 0:
                ax.set_title(algorithm)
            if j == 0:
                ax.set_ylabel(layout, fontsize=11, fontweight="bold", rotation=90)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes, shrink=0.5, pad=0.02,
                 label="Visits relative to the busiest cell of each panel")
    fig.suptitle("Customer heatmaps (seed 0). Black outline = hotspots (top 10% of visited cells)",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grouped_bars(ax, summary: dict, key: str, ylabel: str) -> None:
    layouts, algos = list(summary), list(ALGORITHMS)
    width = 0.8 / len(algos)
    x = np.arange(len(layouts))
    for k, algorithm in enumerate(algos):
        means = [summary[layout][algorithm][key]["mean"] for layout in layouts]
        stds = [summary[layout][algorithm][key]["std"] for layout in layouts]
        ax.bar(x + (k - (len(algos) - 1) / 2) * width, means, width, yerr=stds, capsize=3,
               color=ALGO_COLORS[algorithm], label=algorithm, error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_xticks(x, layouts)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


def plot_nodes_expanded(summary: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    grouped_bars(ax, summary, "avg_nodes_per_search", "Average nodes expanded per search")
    ax.set_title("Search effort: nodes expanded per search (mean ± std over seeds)")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_layout_comparison(summary: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    grouped_bars(axes[0], summary, "avg_trip_ticks", "Average trip time (ticks)")
    axes[0].set_title("Time spent in the store")
    grouped_bars(axes[1], summary, "peak_heat", "Visits to the busiest cell")
    axes[1].set_title("Crowding at the busiest cell")
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=config.DEFAULT_CUSTOMERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    raw, hotspots, sims = run_experiments(args.customers, args.seeds)
    summary = {layout: {algo: aggregate(runs) for algo, runs in by_algo.items()}
               for layout, by_algo in raw.items()}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison_table(summary).to_csv(config.RESULTS_DIR / "comparison.csv", index=False)
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps({
        "customers": args.customers,
        "seeds": args.seeds,
        "metrics": summary,
        "hotspots_seed_" + str(args.seeds[0]): hotspots,
    }, indent=2))
    plot_heatmaps(sims, config.FIGURES_DIR / "heatmaps.png")
    plot_nodes_expanded(summary, config.FIGURES_DIR / "nodes_expanded.png")
    plot_layout_comparison(summary, config.FIGURES_DIR / "layout_comparison.png")

    print(f"{args.customers} customers, seeds {args.seeds}\n")
    print_summary(summary, hotspots)
    print(f"\nSaved -> {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
