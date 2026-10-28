# Technical Guide: Dynamic Pricing with RL (Gymnasium + Stable-Baselines3)

This guide explains every design decision in Task C, following the structure of the assignment brief. The code lives in `task_c_dynamic_pricing/`, and every command below is run from that folder.

---

## 1. Project architecture

```
┌──────────────────────────┐
│ retail_sales_dataset.csv │  Transaction ID, Date, Customer ID, Gender, Age,
└────────────┬─────────────┘  Product Category, Quantity, Price per Unit, Total Amount
             │ data/prepare_data.py
             ▼
┌──────────────────────────────────────────────────────────────┐
│ PREPROCESSING                                                 │
│ clean → filter "Clothing" → age segments → demand statistics   │
│ output: reference price, segment shares, base demand per      │
│         segment, weekday trend, price sensitivity (assumed)   │
└────────────┬─────────────────────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────────────────┐
│ ENVIRONMENT  env/pricing_env.py  (gymnasium.Env)              │
│                                                              │
│   observation ──► agent ──► action (−10 % … +10 %)            │
│        ▲                          │                           │
│        │          new price = clip(price × multiplier)        │
│        │          demand ~ Poisson(λ(price, segment, weekday))│
│        │          sold = min(demand, inventory)               │
│        └───────── reward = price × sold ◄─────────────────────┘
└────────────┬─────────────────────────────────────────────────┘
             │
    ┌────────┴─────────┐
    ▼                  ▼
┌────────────────┐  ┌──────────────────────────────────────────┐
│ TRAINING       │  │ EVALUATION                               │
│ train.py       │  │ evaluate.py + baselines.py               │
│ 4 vec envs     │  │ 100 paired test episodes                 │
│ VecNormalize   │  │ PPO, DQN vs dataset price, best fixed    │
│ PPO / DQN      │  │ price, random → metrics.json, figures/   │
│ 3 seeds        │  └──────────────────────────────────────────┘
│ models/*.zip ──┼──► pygame_ui/dashboard.py (interactive demo)
└────────────────┘
```

**Workflows**

| Workflow | Steps |
|---|---|
| Environment | `reset()` samples a market (start segment, weekday, initial price) → for 30 days: `step(action)` adjusts the price, simulates demand, returns revenue as the reward and the next observation |
| Training | check the env with `check_env` → build 4 parallel envs → normalize rewards → `model.learn()` with periodic evaluation → `model.save()` → reload check |
| Evaluation | find the best fixed price on training seeds → run every policy on the same 100 held-out seeds → total and mean revenue, improvement %, confidence intervals, win rate → learning-curve stability → plots |

---

## 2. Dataset preparation (`data/prepare_data.py`)

| Step | What | Why |
|---|---|---|
| Load | `pd.read_csv(...)` from `task_c_dynamic_pricing/data/` or the repository-level `data/` folder shared by all tasks. If the file is missing, a synthetic dataset with the same schema is generated | The pipeline always runs, even before the Kaggle file is downloaded |
| Clean | trim column names, parse `Date`, coerce numeric columns, normalize the case of categorical text | Prevents silent type errors (for example prices read as strings) |
| Missing values | **dropped**, not imputed | Only about 1000 rows with a handful of discrete price points: an imputed price would be a fake price point and would bias the calibration |
| Validity | drop rows with Quantity ≤ 0 or Price ≤ 0, or with an out-of-range age; drop duplicate transaction IDs; recompute `Total Amount = Price × Quantity` | Guarantees the identity that revenue (our reward) is built on |
| Category | keep `Clothing` (configurable in `config.PRODUCT_CATEGORY`) | UrbanStyle is a fashion retailer. One category means one product line with its own demand curve |
| Segments | age bands **Young 18–29, Adult 30–49, Senior 50+** | Age is the strongest available proxy for price sensitivity (income and life stage). Gender is reported per segment but not used to split, because 6 segments would leave too few rows each |
| Demand | per segment: reference price (mean price paid), base demand (units per day scaled by `DEMAND_SCALE` to one store's volume), market share. Overall: weekday trend (units by day of week ÷ mean) | These values come directly from the transactions and anchor the simulator to the data |
| Price sensitivity | assumed β per segment: Young 2.0, Adult 1.1, Senior 0.65 | See below |

**Why price sensitivity is assumed.** In the dataset each product sells at one of five fixed price points (25, 30, 50, 300, 500), and price and quantity are essentially independent. There is no price variation for the *same* product, so a price elasticity cannot be estimated statistically. This is the usual situation for dynamic pricing projects. The standard approach is a *calibrated simulator*: the volume, mix and seasonality come from data, and the price response is an explicit, documented assumption. Younger customers are modeled as more price-sensitive, which is a common finding in fashion retail. `tests/test_env.py` checks that the assumptions imply a different optimal price per segment.

Run `python -m data.prepare_data` to print the calibrated values and save `results/market_params.json`.

---

## 3. Gymnasium environment (`env/pricing_env.py`)

`UrbanStylePricingEnv(gym.Env)` implements:

| Member | Definition |
|---|---|
| `observation_space` | `Box(0, 3, shape=(10,), float32)` |
| `action_space` | `Discrete(5)` |
| `reset(seed, options)` | random starting segment and weekday, full inventory, initial price from `options["initial_price"]` (used by the Pygame slider) or uniform in [0.7, 1.3] × the reference price |
| `step(action)` | applies the multiplier, clips the price to [0.5, 2.0] × the reference price, draws demand, sells up to the inventory, returns `(obs, reward, terminated, truncated, info)` |
| `render()` | `"human"` prints one line per day, `"ansi"` returns it as a string |

**Demand model.** For the segment *s* that is shopping today and weekday *d*:

```
λ = base_demand[s] · trend[d] · exp(−β[s] · (price / ref_price[s] − 1))
units demanded ~ Poisson(λ)          units sold = min(demand, inventory)
```

- The exponential form means demand always falls as the price rises, never goes negative, and equals `base_demand` at the reference price.
- Expected revenue `p·λ(p)` has a single interior maximum at `p* = ref_price[s] / β[s]`. With the Kaggle data (category reference price 174.29) that is ≈0.57× the reference price for Young, 0.87× for Adult and 1.49× for Senior. **No single fixed price is optimal for all segments**, and that is exactly what a dynamic policy can exploit.
- Poisson noise makes daily sales stochastic, as real sales are.
- The customer segment follows a **sticky Markov chain**: it keeps the same segment with probability 0.85, otherwise it draws a new one from the empirical shares. This models runs of days dominated by one group (school holidays, pay days, campaigns). The next day's segment is shown in the observation, so the agent can react.

**Episode.** 30 days. `terminated=True` after day 30. The horizon is part of the state (time remaining), so this is a true terminal state, not a truncation.

The full runnable code is in `env/pricing_env.py`. Minimal usage:

```python
from env.pricing_env import UrbanStylePricingEnv
env = UrbanStylePricingEnv(render_mode="human")
obs, info = env.reset(seed=0, options={"initial_price": 170})
done = False
while not done:
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
```

---

## 4. State representation

| Index | Feature | Scaling | Why it is needed |
|---|---|---|---|
| 0 | Current price | ÷ reference price | The action is *relative* (±%), so the agent must know where the price currently is to know where it will land |
| 1–3 | Customer segment (one-hot) | 0/1 | The optimal price depends on the segment. One-hot avoids implying an order between segments |
| 4–6 | Sales of the last 3 days | ÷ (2 × max base demand) | Gives a direct signal of the demand response to recent prices, and a little memory, because the policy itself is memoryless |
| 7 | Demand trend (weekday factor) | ≈0.5–1.5 | Busy days sell more units at any price. Matters for allocating inventory |
| 8 | Inventory level | ÷ initial stock | Low stock means a higher price is better (scarcity). This is the main link between today's decision and future days |
| 9 | Time remaining | ÷ 30 | Near the end of the episode, leftover stock is worthless, so the agent can sell more aggressively |

All features are roughly in [0, 1]. Neural networks train much better on normalized inputs, and it means no observation-normalization wrapper is needed at inference time.

---

## 5. Action space

| Action | Meaning | Multiplier |
|---|---|---|
| 0 | Decrease price 10% | 0.90 |
| 1 | Decrease price 5% | 0.95 |
| 2 | Keep price | 1.00 |
| 3 | Increase price 5% | 1.05 |
| 4 | Increase price 10% | 1.10 |

Why this action space works:

- **Relative, bounded changes** match how retailers actually reprice: small markdowns or markups, not jumps to arbitrary values. Customers are not confronted with erratic prices.
- **Discrete and small (5 actions).** Both DQN, which needs discrete actions, and PPO can be used. Exploration is easy, and each action is interpretable for the report and the dashboard.
- **Two step sizes** (5% and 10%) allow fast moves toward a new segment's optimum and fine-tuning near it.
- The **[0.5, 2.0] × reference price clip** is a business guardrail against absurd prices.

A continuous alternative would be `Box(-0.1, 0.1)` with PPO or SAC. It is more expressive but harder to explain and is not needed here.

---

## 6. Reward function

The main reward is exactly the daily revenue:

```python
reward = price * quantity_sold
```

This is the business objective. Because SB3 maximizes the *discounted sum* of rewards (γ = 0.99, close to 1 over 30 days), the agent maximizes the episode revenue rather than greedily maximizing each day. It will, for example, keep stock for busy days.

Revenue is in currency units, thousands per day. Training rescales the reward with `VecNormalize(norm_reward=True)`, which leaves the optimal policy unchanged but keeps the value-function targets well scaled.

**Possible improvements** (`reward_mode="shaped"` implements the first two):

| Improvement | Formula idea | Effect |
|---|---|---|
| Inventory holding cost | `− h · leftover_stock` | Discourages overstocking, so the agent clears inventory |
| Stockout / unmet-demand penalty | `− k · (demand − sold)` | Lost sales hurt future loyalty, so the agent avoids pricing so low that it runs out |
| Customer satisfaction / price stability | `− c · \|Δprice\|` or a penalty above a fair-price threshold | Avoids price whiplash that customers perceive as unfair |
| Long-term profitability | `(price − unit_cost) · sold` instead of revenue | Optimizes margin, not turnover. Revenue maximization can mean selling at a loss |
| Brand / markdown limits | constraints, for example at most N markdowns per month | Protects brand positioning for premium lines |

The main results use the pure revenue reward, because the brief defines the objective as revenue and it keeps the comparison with static pricing fair.

---

## 7. RL training (`training/train.py`)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env.pricing_env import UrbanStylePricingEnv

vec_env = make_vec_env(UrbanStylePricingEnv, n_envs=4, seed=0, monitor_dir="results/logs/ppo_seed0")
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)          # reward scaling only
model = PPO("MlpPolicy", vec_env, n_steps=1024, batch_size=256, gamma=0.99, ent_coef=0.01, seed=0)
model.learn(total_timesteps=200_000)                                     # training loop
model.save("models/ppo_urbanstyle_seed0")                                # saving
model = PPO.load("models/ppo_urbanstyle_seed0")                          # loading
action, _ = model.predict(obs, deterministic=True)                       # inference
```

The script adds environment validation (`check_env`), an `EvalCallback` that keeps the best checkpoint in `models/best_<algo>_seed<k>/`, three training seeds, DQN support and a save/load round-trip check:

```bash
python -m training.train --algo ppo --timesteps 200000 --seeds 0 1 2
python -m training.train --algo dqn --timesteps 200000 --seeds 0 1 2
```

`Monitor` (inside `make_vec_env`) writes the raw episode revenue of every training episode to `results/logs/<algo>_seed<k>/`. These logs are the data for the learning curves.

**Recommended algorithm: PPO.**

- It is on-policy and uses a clipped objective, which gives very stable learning on stochastic rewards (Poisson demand).
- It uses parallel environments naturally (4 here), which gives more diverse experience per update.
- It works with discrete or continuous actions, so it supports the continuous-price extension.
- It is robust to hyperparameters. SB3 defaults work almost unchanged.

DQN is a good fit for the small discrete action space and is more sample-efficient through its replay buffer. However, it is more sensitive to exploration settings and reward scale, and its learning curves are noisier. It is trained as a comparison.

---

## 8. Baseline comparison (`evaluation/baselines.py`)

| Strategy | Definition | Role |
|---|---|---|
| **Dataset price** | always the category's mean `Price per Unit` from the dataset (action "keep") | What UrbanStyle does today: the original price |
| **Best fixed price** | grid search of 31 fixed prices (0.50–2.00 × reference), chosen on *training* seeds | The strongest possible static policy: an optimal but non-adaptive price |
| **Random** | uniformly random actions | Sanity floor. Any learned policy must beat it |

Every policy runs on the **same 100 test seeds** (paired comparison: same customers, same demand shocks). RL agents start at the dataset price, the same as the static strategy, so the only difference is the pricing decisions.

---

## 9. Evaluation metrics (`evaluation/evaluate.py`)

| Metric | Definition | Interpretation |
|---|---|---|
| Total revenue | Σ revenue over the 100 test episodes | Overall business outcome |
| Average revenue | mean ± std revenue per 30-day episode | Expected monthly revenue and its risk |
| Revenue improvement % | `100 · (Σ RL / Σ baseline − 1)` | The headline number |
| 95% CI and win rate | paired difference per episode, mean ± 1.96·SE, and the share of episodes where RL wins | Shows whether the gain is statistically reliable and not luck |
| Reward per episode | training-episode revenue from the Monitor logs | Learning progress |
| Training stability | coefficient of variation of the reward in the last 20% of training, and the std of final reward across 3 seeds | Low values mean the agent converged and is reproducible |

```bash
python -m evaluation.evaluate
```

Outputs:

- `results/metrics.json`, `results/comparison.csv` and `results/episodes.csv`
- `results/figures/learning_curves.png`: mean ± std over seeds, with static baselines as dashed lines
- `results/figures/revenue_comparison.png`: bar chart of mean revenue with % improvement, and a box plot of the distribution
- `results/figures/price_trajectory.png`: one test episode showing how the agent moves the price as segments change

---

## 10. Pygame integration (`pygame_ui/`)

```bash
python -m pygame_ui.dashboard
python -m pygame_ui.dashboard --screenshot results/figures/dashboard.png   # headless
```

| Required element | Implementation |
|---|---|
| Slider for initial price | `Slider` in [0.5, 2.0] × the reference price, used on reset |
| Button to reset environment | **Reset Env** (new random market) |
| Button to run an episode | **Run Episode** (one day every 150 ms), plus **Step** for single days |
| Current price, segment, sales, reward | stat panel on the left. The segment is shown as today → tomorrow |
| Revenue history chart | cumulative revenue of the selected policy vs a static-price "shadow" env with the same seed, plus a price-history chart |
| Policy switch | cycles through PPO, DQN and Static |

The widgets (`widgets.py`, about 100 lines) are plain Pygame drawing calls. The UI only visualizes; all logic lives in the environment and the trained model.

---

## 11. Project structure

```
task_c_dynamic_pricing/
├── config.py               # single source of truth for constants & hyperparameters
├── requirements.txt
├── README.md
├── data/                   # retail_sales_dataset.csv + prepare_data.py
├── env/pricing_env.py      # UrbanStylePricingEnv
├── training/train.py
├── evaluation/             # baselines.py, evaluate.py
├── pygame_ui/              # widgets.py, dashboard.py
├── tests/test_env.py
├── models/                 # *.zip agents
├── results/                # metrics, CSVs, figures, logs
└── docs/                   # this guide + REPORT.md
```

Imports are package-relative from this folder (`python -m package.module`), so every script can be run without installing the project.
