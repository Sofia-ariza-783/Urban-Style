# UrbanStyle, Task C: Dynamic Pricing with Reinforcement Learning

An RL agent (Stable-Baselines3 **PPO**, with **DQN** for comparison) learns to adjust the price of the *Clothing* category day by day to maximize revenue. It acts inside a custom **Gymnasium** environment calibrated on the UrbanStyle retail sales dataset, and is compared against static pricing. A **Pygame** dashboard lets you run episodes interactively.

- Full technical explanation (sections 1–11 of the brief): [`docs/TECHNICAL_GUIDE.md`](docs/TECHNICAL_GUIDE.md)
- Academic report (section 12): [`docs/REPORT.md`](docs/REPORT.md)

## Architecture

```
 retail_sales_dataset.csv
          │  data/prepare_data.py   clean → filter category → age segments → demand statistics
          ▼
 market parameters (ref. price, segment shares, base demand, weekday trend, price sensitivity)
          │
          ▼
 env/pricing_env.py   UrbanStylePricingEnv (gymnasium.Env)
   state:  price, segment, recent sales, trend, inventory, time left
   action: −10% −5% 0 +5% +10%
   reward: price × units sold
          │                                     │
          ▼                                     ▼
 training/train.py                     evaluation/evaluate.py
   make_vec_env (4 envs) + VecNormalize   RL vs static (dataset price, best fixed price, random)
   PPO / DQN, 3 seeds, EvalCallback       100 paired test episodes → metrics.json, figures/
   models/*.zip                                 │
          └──────────────┬──────────────────────┘
                         ▼
              pygame_ui/dashboard.py   slider · reset · step · run episode · live charts
```

## Quick start

Run all commands from this folder (`task_c_dynamic_pricing/`):

```bash
pip install -r requirements.txt

python -m data.prepare_data              # calibrate the market (uses data/retail_sales_dataset.csv)
python -m pytest tests                   # environment sanity checks
python -m training.train --algo ppo      # ~2 min per seed on CPU, 3 seeds
python -m training.train --algo dqn
python -m evaluation.evaluate            # metrics + figures in results/
python -m pygame_ui.dashboard            # interactive UI
```

For a quick smoke test, run `python -m training.train --algo ppo --timesteps 20000 --seeds 0`.

## Layout

| Path | Purpose |
|---|---|
| `config.py` | All paths, environment constants and hyperparameters |
| `data/prepare_data.py` | Loading, cleaning, segmentation, market calibration |
| `env/pricing_env.py` | `UrbanStylePricingEnv` Gymnasium environment |
| `training/train.py` | SB3 training, saving and loading (PPO / DQN) |
| `evaluation/baselines.py` | Static and random pricing policies |
| `evaluation/evaluate.py` | Metrics, comparison tables and plots |
| `pygame_ui/` | Dashboard and minimal widgets |
| `tests/test_env.py` | Environment tests (`check_env`, reward, bounds, seeding) |
| `models/` | Trained agents (`ppo_urbanstyle_seed0.zip`, …) |
| `results/` | `metrics.json`, `comparison.csv`, `episodes.csv`, `figures/`, training logs |
