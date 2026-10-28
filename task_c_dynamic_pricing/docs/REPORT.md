# Task C: Dynamic Pricing with Reinforcement Learning

*UrbanStyle Project: Artificial Intelligence applied to fashion retail*

> All figures below were produced from the Kaggle *Retail Sales Dataset* (1,000 transactions, 351 in the Clothing category) and can be reproduced with `prepare_data → train → evaluate` (see `README.md`). The raw values are in `results/metrics.json`.

## 1. Objective

The objective of this task is to design, train and evaluate a reinforcement learning (RL) agent that dynamically adjusts the selling price of a product category in order to **maximize total revenue**. We then determine quantitatively whether this adaptive strategy outperforms static pricing. The problem is formulated as a Markov Decision Process (MDP) and implemented as a custom Gymnasium environment calibrated on UrbanStyle's retail transaction data. Agents are trained with the Stable-Baselines3 library.

## 2. Methodology

The work follows a five-stage pipeline:

1. **Data preparation.** The transactional dataset (Transaction ID, Date, Customer ID, Gender, Age, Product Category, Quantity, Price per Unit, Total Amount) is cleaned: types are coerced, incomplete or invalid rows are removed, and the identity *Total = Price × Quantity* is enforced. It is then restricted to the *Clothing* category. Customers are grouped into three age-based segments (Young 18–29, Adult 30–49, Senior 50+).
2. **Market calibration.** For each segment we estimate its market share, its reference (average paid) price and its base daily demand. A day-of-week demand factor is computed for the category. The dataset contains only five fixed price points and shows no price variation within a product, so price elasticity cannot be identified from the data. Segment price sensitivities (β = 2.0, 1.1 and 0.65 for Young, Adult and Senior) are therefore introduced as explicit modeling assumptions, consistent with the higher price sensitivity usually reported for younger fashion consumers.
3. **Environment design.** A stochastic demand simulator is wrapped in the Gymnasium API (Section 3).
4. **Agent training.** Proximal Policy Optimization (PPO) and Deep Q-Network (DQN) agents are each trained for 200,000 environment steps, with three random seeds per algorithm.
5. **Evaluation.** The agents and three non-learning baselines run on the same 100 held-out episodes (a paired design). Revenue, relative improvement, confidence intervals, win rate and training stability are compared.

## 3. Environment Design

**Episode.** One episode is a 30-day selling period. At the start of each day the agent chooses a price adjustment. Customers of the segment active that day then respond, and the generated revenue is returned as the reward.

**Demand model.** For segment *s*, weekday *d* and price *p*:

$$\lambda = D_s \cdot T_d \cdot \exp\!\left(-\beta_s\left(\frac{p}{P_s}-1\right)\right), \qquad q \sim \min\big(\text{Poisson}(\lambda),\ I_t\big)$$

Here $D_s$ is the base demand, $T_d$ the weekday trend, $P_s$ the segment's reference price and $I_t$ the available inventory. Expected revenue $p\lambda(p)$ is maximized at $p^*_s = P_s/\beta_s$. These optimal prices are about 98.5 (0.57×), 151.0 (0.87×) and 258.8 (1.49×), relative to the category's reference price of 174.29, for the Young, Adult and Senior segments respectively. Because the optimal price differs across segments, no constant price is optimal at all times. The active segment evolves as a sticky Markov chain: it stays the same with probability 0.85 and is otherwise resampled from the empirical shares. This represents periods dominated by one customer group.

**State space.** The state is a 10-dimensional normalized vector:

- current price relative to the reference price
- one-hot customer segment
- units sold in each of the last three days
- weekday demand trend
- remaining inventory fraction
- remaining-time fraction

**Action space.** Five discrete actions apply a multiplicative price change of −10%, −5%, 0%, +5% or +10%. Prices are clipped to [0.5, 2.0] × the reference price. Relative, bounded adjustments reflect real markdown and markup practice and avoid erratic price changes.

**Reward.** The reward is the daily revenue, $r_t = p_t \cdot q_t$. With a discount factor γ = 0.99, maximizing the expected return is very close to maximizing episode revenue. Inventory couples the days together: selling too much early leaves less stock for later. An optional shaped reward, with holding and stockout penalties, is implemented but not used in the main experiments.

## 4. RL Agent

**PPO** is the primary agent. It is an on-policy actor-critic method whose clipped surrogate objective limits each policy update, which makes learning stable under the noisy (Poisson) rewards of this environment. Experience is collected from four parallel environments (`n_steps` = 1024, batch size 256, 10 epochs, learning rate 3·10⁻⁴, entropy coefficient 0.01).

**DQN** is the comparison agent. It is an off-policy, value-based method with experience replay (buffer of 50,000 transitions, ε-greedy exploration decayed over 30% of training, target network updated every 1,000 steps).

Both use a two-layer MLP policy (64–64 units). Rewards are rescaled during training with a running normalization (`VecNormalize`). This does not change the optimal policy, but it improves the conditioning of the value-function targets. A periodic evaluation callback keeps the best checkpoint. The final models are saved and reloaded for evaluation and for the interactive Pygame dashboard.

## 5. Results

**Table 1.** Revenue over 100 paired test episodes (30 days each). RL rows show the per-episode mean across three training seeds.

| Strategy | Total revenue | Mean ± SD per episode | vs dataset price | vs best fixed price |
|---|---:|---:|---:|---:|
| **PPO** | **28,407,492** | **284,075 ± 29,995** | **+6.26%** | **+3.78%** |
| DQN | 27,684,446 | 276,844 ± 29,588 | +3.55% | +1.14% |
| Best fixed price (148.14) | 27,371,935 | 273,719 ± 30,100 | +2.38% | — |
| Dataset price (174.29) | 26,734,521 | 267,345 ± 20,471 | — | −2.33% |
| Random actions | 25,898,723 | 258,987 ± 24,105 | −3.13% | −5.38% |

**Table 2.** Paired comparison of the RL agents against the static baselines. For each test episode we take the agent's revenue minus the baseline's revenue on the same seed, then report the mean difference and its 95% confidence interval.

| Comparison | Mean gain per episode | 95% CI | Episodes won |
|---|---:|---:|---:|
| PPO vs dataset price | +16,730 | [13,475; 19,985] | 95% |
| PPO vs best fixed price | +10,356 | [6,841; 13,870] | 79% |
| DQN vs dataset price | +9,499 | [6,818; 12,181] | 79% |
| DQN vs best fixed price | +3,125 | [118; 6,132] | 69% |

**Table 3.** Training stability. The tail CV is the coefficient of variation of the smoothed training reward over the last 20% of training. The standard deviation across seeds is computed on that same tail.

| Agent | Final training reward | Tail CV | Std. across seeds (training) | Test mean revenue per seed |
|---|---:|---:|---:|---|
| PPO | 280,989 | 1.53% | 656 | 284,467 / 284,048 / 283,710 |
| DQN | 277,037 | 1.71% | 140 | 278,474 / 280,026 / 272,034 |

Figures (in `results/figures/`):

- `learning_curves.png`: both agents pass the dataset-price baseline after roughly 15,000–20,000 steps and the best fixed price after roughly 50,000 steps. They plateau after about 75,000 steps, and PPO stays above DQN for most of training.
- `revenue_comparison.png`: mean revenue per strategy and the distribution of episode revenue.
- `price_trajectory.png`: in one test episode, PPO raises the price during Senior-dominated periods and cuts it steadily when price-sensitive Young customers dominate.
- `dashboard.png`: the Pygame interface.

## 6. Comparison with Static Pricing

Both RL agents earn more than every static strategy. For PPO the result is statistically clear: both confidence intervals exclude zero by a wide margin. PPO beats the original price in 95% of test episodes and the best fixed price in 79%. DQN's advantage over the best fixed price is small (+1.14%), and its confidence interval only just excludes zero.

The comparison against the **best fixed price** matters most. That baseline is an idealized manager who knows the demand model and picks the single best constant price (148.14, a 15% markdown), so it is already 2.38% better than UrbanStyle's original price. The extra +3.78% that PPO achieves on top of it comes from *adaptivity*: the agent conditions the price on the observed customer segment, recent sales and remaining stock.

The price trajectory shows interpretable behavior:

- **Markups** toward the high optimal price of the price-insensitive Senior segment.
- **Gradual markdowns** toward the low optimal price of the elastic Young segment.
- **Partial rather than full moves** toward each segment's myopic optimum. This hedges against the segment switching again, given the ±10% per-day limit on price changes.

The static dataset price has the lowest variance among the non-random strategies, because it never reacts. Its average revenue, however, is 5.9% lower than PPO's (equivalently, PPO earns 6.26% more).

PPO was the better algorithm. It outperformed DQN by 2.7 percentage points against the original price, and its three seeds gave almost identical test revenue (a spread of 757). DQN's test revenue varied much more across seeds (a spread of 7,992), even though its training curves looked similarly flat. Value-based methods can converge to noticeably different greedy policies from small differences in their Q-estimates.

## 7. Conclusions

A reinforcement learning formulation of dynamic pricing was implemented end to end:

- a data-calibrated Gymnasium environment
- PPO and DQN agents trained with Stable-Baselines3
- a rigorous paired evaluation
- an interactive Pygame dashboard

The PPO agent increased revenue by **6.3% compared with UrbanStyle's original price** and by **3.8% compared with the best possible fixed price**, with stable and reproducible results across seeds. PPO is therefore the recommended algorithm for this problem.

**Limitations.**

1. Price sensitivities are assumptions, because the historical data has no price variation. The absolute gains depend on these values, though the methodology does not.
2. Only one product category is modeled, and competitors' prices and cross-category effects are ignored.
3. Revenue, not profit, is optimized.

**Future work.**

- Estimate elasticities from A/B price tests or from richer data.
- Optimize margin with unit costs.
- Add price-stability or customer-satisfaction terms to the reward.
- Model several categories jointly.
- Use a continuous action space (PPO or SAC), or a recurrent policy to infer segments that are not directly observed.
