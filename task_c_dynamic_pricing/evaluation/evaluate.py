"""Evaluate RL agents against static pricing and produce metrics + figures.

Run (after training):  python -m evaluation.evaluate
Outputs:
    results/metrics.json        all metrics
    results/comparison.csv      one row per strategy
    results/episodes.csv        revenue of every evaluation episode
    results/figures/*.png       learning curves, revenue comparison, price trajectory
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from stable_baselines3.common.monitor import load_results  # noqa: E402

import config  # noqa: E402
from env.pricing_env import UrbanStylePricingEnv  # noqa: E402

# Validated categorical palette (fixed order, colour follows the entity)
COLORS = {
    "PPO": "#2a78d6", "DQN": "#eb6834", "Best fixed price": "#1baf7a",
    "Dataset price": "#eda100", "Random": "#8a8984",
}
SEGMENT_COLORS = {"Young": "#2a78d6", "Adult": "#eb6834", "Senior": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "font.size": 10, "lines.linewidth": 2,
})


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------
def run_episodes(policy, env: UrbanStylePricingEnv, n_episodes: int,
                 seed_offset: int = config.EVAL_SEED_OFFSET, record: bool = False) -> dict:
    """Roll out a policy; seeds are shared across policies (paired comparison)."""
    revenues, trajectories = [], []
    for ep in range(n_episodes):
        options = policy.reset_options(env) if hasattr(policy, "reset_options") else {"initial_price": env.ref_price}
        obs, _ = env.reset(seed=seed_offset + ep, options=options)
        done, total, traj = False, 0.0, []
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
            total += info["revenue"]
            if record:
                traj.append(info)
        revenues.append(total)
        trajectories.append(traj)
    return {"revenue": np.array(revenues), "trajectories": trajectories}


def summarize(revenues: np.ndarray) -> dict:
    return {
        "total_revenue": float(revenues.sum()),
        "mean_revenue": float(revenues.mean()),
        "std_revenue": float(revenues.std(ddof=1)),
        "min_revenue": float(revenues.min()),
        "max_revenue": float(revenues.max()),
    }


def improvement(a: np.ndarray, b: np.ndarray) -> dict:
    """Revenue growth of strategy a over b on the same (paired) episodes."""
    diff = a - b
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return {
        "improvement_pct": float(100 * (a.sum() / b.sum() - 1)),
        "mean_diff": float(diff.mean()),
        "ci95_low": float(diff.mean() - 1.96 * se),
        "ci95_high": float(diff.mean() + 1.96 * se),
        "win_rate": float((diff > 0).mean()),
    }


# --------------------------------------------------------------------------
# Training curves
# --------------------------------------------------------------------------
def training_curves(algo: str, window: int = 50) -> pd.DataFrame | None:
    """Episode reward vs timesteps for each seed, interpolated on a common grid."""
    curves = {}
    for seed in config.SEEDS:
        log_dir = config.LOGS_DIR / f"{algo}_seed{seed}"
        if not log_dir.exists():
            continue
        df = load_results(str(log_dir)).sort_values("t")
        if df.empty:
            continue
        df["timesteps"] = df["l"].cumsum()
        curves[seed] = df[["timesteps", "r"]]
    if not curves:
        return None
    max_t = min(c["timesteps"].max() for c in curves.values())
    grid = np.linspace(window * config.EPISODE_LENGTH, max_t, 200)
    out = pd.DataFrame({"timesteps": grid})
    for seed, c in curves.items():
        smooth = c["r"].rolling(window, min_periods=1).mean()
        out[f"seed{seed}"] = np.interp(grid, c["timesteps"], smooth)
    return out


def stability(curves: pd.DataFrame) -> dict:
    seeds = curves.drop(columns="timesteps")
    tail = seeds.iloc[int(0.8 * len(seeds)):]
    return {
        "final_mean_reward": float(tail.mean().mean()),
        # Variation of the reward within the last 20% of training (per seed, averaged)
        "tail_cv_pct": float(100 * (tail.std() / tail.mean()).mean()),
        # Disagreement between seeds at the end of training
        "between_seed_std": float(tail.mean().std(ddof=0)),
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_learning_curves(all_curves: dict, baselines: dict):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for algo, curves in all_curves.items():
        seeds = curves.drop(columns="timesteps")
        mean, std = seeds.mean(axis=1), seeds.std(axis=1, ddof=0)
        ax.plot(curves["timesteps"], mean, color=COLORS[algo], label=f"{algo} (mean of {seeds.shape[1]} seeds)")
        ax.fill_between(curves["timesteps"], mean - std, mean + std, color=COLORS[algo], alpha=0.15, linewidth=0)
    for name, value in baselines.items():
        ax.axhline(value, color=COLORS[name], linestyle="--", linewidth=1.5, label=f"{name} (static)")
    ax.set_title("Training: revenue per episode")
    ax.set_xlabel("Training timesteps (days simulated)")
    ax.set_ylabel("Episode revenue (rolling mean of 50)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.xaxis.set_major_formatter(THOUSANDS)
    fig.savefig(config.FIGURES_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_revenue_comparison(results: dict, base_name: str):
    names = list(results)
    means = [results[n]["revenue"].mean() for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    bars = ax1.bar(names, means, color=[COLORS[n] for n in names], width=0.6,
                   edgecolor=SURFACE, linewidth=2)
    base = results[base_name]["revenue"].mean()
    for bar, m, n in zip(bars, means, names):
        label = f"{m:,.0f}" if n == base_name else f"{m:,.0f}\n({100 * (m / base - 1):+.1f}%)"
        ax1.text(bar.get_x() + bar.get_width() / 2, m, label, ha="center", va="bottom", fontsize=9, color=INK)
    lo = min(means) * 0.9
    ax1.set_ylim(lo, max(means) * 1.04)
    ax1.set_title(f"Mean revenue per episode (% vs {base_name.lower()})")
    ax1.set_ylabel("Revenue per 30-day episode")
    ax1.grid(axis="x", visible=False)

    box = ax2.boxplot([results[n]["revenue"] for n in names], patch_artist=True, widths=0.55,
                      medianprops={"color": INK, "linewidth": 1.5})
    for patch, n in zip(box["boxes"], names):
        patch.set_facecolor(COLORS[n])
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK_2)
    ax2.set_xticks(range(1, len(names) + 1), names)
    ax2.set_title(f"Episode revenue distribution ({config.EVAL_EPISODES} test episodes)")
    ax2.set_ylabel("Revenue per 30-day episode")
    ax2.grid(axis="x", visible=False)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=15)
        ax.yaxis.set_major_formatter(THOUSANDS)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "revenue_comparison.png", dpi=150)
    plt.close(fig)


def plot_price_trajectory(env: UrbanStylePricingEnv, traj_rl: list, traj_static: list, algo: str):
    days = [i["day"] for i in traj_rl]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, height_ratios=[3, 2])

    # Shade the active customer segment behind the price line
    for i in traj_rl:
        ax1.axvspan(i["day"] - 1, i["day"], color=SEGMENT_COLORS[i["segment"]], alpha=0.12, linewidth=0)
    ax1.step(days, [i["price"] for i in traj_rl], where="pre", color=COLORS[algo], label=f"{algo} price")
    ax1.step(days, [i["price"] for i in traj_static], where="pre", color=COLORS["Dataset price"],
             linestyle="--", label="Static (dataset) price")
    for s, name in enumerate(env.segment_names):
        ax1.axhline(env.optimal_price(s), color=SEGMENT_COLORS[name], linewidth=1, linestyle=":",
                    label=f"Optimal price, {name} (shaded days)")
    optima = [env.optimal_price(s) for s in range(env.n_segments)]
    prices = [i["price"] for i in traj_rl + traj_static]
    ax1.set_ylim(0.9 * min(optima + prices), 1.08 * max(optima + prices))
    ax1.set_ylabel("Price")
    ax1.set_title(f"Sample test episode: {algo} adapts the price to the active segment")
    ax1.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))

    ax2.plot(days, np.cumsum([i["revenue"] for i in traj_rl]), color=COLORS[algo], label=algo)
    ax2.plot(days, np.cumsum([i["revenue"] for i in traj_static]), color=COLORS["Dataset price"],
             linestyle="--", label="Static (dataset) price")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Cumulative revenue")
    ax2.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax2.yaxis.set_major_formatter(THOUSANDS)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "price_trajectory.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    from evaluation.baselines import RandomPolicy, StaticPricePolicy, find_best_fixed_price
    from training.train import load_model, model_path

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    env = UrbanStylePricingEnv()
    n = config.EVAL_EPISODES

    best_factor, grid_scores = find_best_fixed_price()
    policies = {}
    for algo in ("ppo", "dqn"):
        seeds = [s for s in config.SEEDS if (config.MODELS_DIR / f"{algo}_urbanstyle_seed{s}.zip").exists()]
        if seeds:
            policies[algo.upper()] = [load_model(algo, s) for s in seeds]
    if not policies:
        raise SystemExit(f"No trained models found (expected e.g. {model_path('ppo', 0)}). Run training first.")
    policies["Best fixed price"] = [StaticPricePolicy(best_factor, "Best fixed price")]
    policies["Dataset price"] = [StaticPricePolicy(1.0, "Dataset price")]
    policies["Random"] = [RandomPolicy(seed=0)]

    # RL agents: evaluate every seed; report pooled episodes across seeds
    results = {}
    per_seed = {}
    for name, models in policies.items():
        runs = [run_episodes(m, env, n)["revenue"] for m in models]
        per_seed[name] = [float(r.mean()) for r in runs]
        results[name] = {"revenue": np.mean(runs, axis=0)}  # mean over seeds, per episode

    metrics = {"best_fixed_price_factor": best_factor,
               "best_fixed_price": best_factor * env.ref_price,
               "dataset_price": env.ref_price,
               "eval_episodes": n,
               "strategies": {}}
    rows = []
    for name, r in results.items():
        s = summarize(r["revenue"])
        s["per_seed_mean_revenue"] = per_seed[name]
        for base in ("Dataset price", "Best fixed price"):
            if name != base:
                s[f"vs_{base.lower().replace(' ', '_')}"] = improvement(r["revenue"], results[base]["revenue"])
        metrics["strategies"][name] = s
        rows.append({
            "strategy": name, **{k: v for k, v in s.items() if not isinstance(v, (dict, list))},
            "improvement_vs_dataset_pct": s.get("vs_dataset_price", {}).get("improvement_pct", 0.0),
            "improvement_vs_best_fixed_pct": s.get("vs_best_fixed_price", {}).get("improvement_pct", 0.0),
        })

    all_curves = {}
    for algo in ("PPO", "DQN"):
        curves = training_curves(algo.lower())
        if curves is not None:
            all_curves[algo] = curves
            metrics["strategies"][algo]["training_stability"] = stability(curves)

    plot_learning_curves(all_curves, {
        "Best fixed price": results["Best fixed price"]["revenue"].mean(),
        "Dataset price": results["Dataset price"]["revenue"].mean(),
    })
    plot_revenue_comparison(results, "Dataset price")
    main_algo = "PPO" if "PPO" in policies else next(iter(policies))
    sample_seed = config.EVAL_SEED_OFFSET + 3
    traj_rl = run_episodes(policies[main_algo][0], env, 1, seed_offset=sample_seed, record=True)["trajectories"][0]
    traj_st = run_episodes(policies["Dataset price"][0], env, 1, seed_offset=sample_seed, record=True)["trajectories"][0]
    plot_price_trajectory(env, traj_rl, traj_st, main_algo)

    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "comparison.csv", index=False)
    pd.DataFrame({k: v["revenue"] for k, v in results.items()}).to_csv(config.RESULTS_DIR / "episodes.csv", index_label="episode")
    metrics["fixed_price_grid"] = {str(k): v for k, v in grid_scores.items()}
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- Console summary --------------------------------------------------
    print(f"\nEvaluation over {n} paired test episodes "
          f"(best fixed price = {best_factor:.2f} x {env.ref_price:.2f} = {best_factor * env.ref_price:.2f})\n")
    print(f"{'Strategy':<18}{'Total':>14}{'Mean':>12}{'Std':>10}{'vs dataset':>12}{'vs best fixed':>15}")
    for name, s in metrics["strategies"].items():
        vd = s.get("vs_dataset_price", {}).get("improvement_pct")
        vb = s.get("vs_best_fixed_price", {}).get("improvement_pct")
        fmt = lambda v: "-" if v is None else f"{v:+.2f}%"  # noqa: E731
        print(f"{name:<18}{s['total_revenue']:>14,.0f}{s['mean_revenue']:>12,.0f}{s['std_revenue']:>10,.0f}"
              f"{fmt(vd):>12}{fmt(vb):>15}")
    for algo in all_curves:
        st = metrics["strategies"][algo]["training_stability"]
        print(f"{algo} stability: final reward {st['final_mean_reward']:,.0f}, "
              f"tail CV {st['tail_cv_pct']:.2f}%, between-seed std {st['between_seed_std']:,.0f}")
    print(f"\nSaved metrics and figures to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
