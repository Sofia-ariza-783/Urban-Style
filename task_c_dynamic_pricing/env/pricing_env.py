"""Custom Gymnasium environment for UrbanStyle dynamic pricing.

One episode = EPISODE_LENGTH days of selling one product category. Every day
the agent observes the market state, picks a price adjustment and receives the
revenue generated that day as reward.

Demand model (per day, for the customer segment active that day):

    lambda = base_demand[seg] * weekday_trend[dow] * exp(-beta[seg] * (p / ref[seg] - 1))
    quantity ~ Poisson(lambda), capped by the available inventory

Daily revenue p * lambda is maximised at p* = ref[seg] / beta[seg], so the
revenue-optimal price depends on the segment that is currently shopping.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from data.prepare_data import load_market_params


class UrbanStylePricingEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        market_params: dict | None = None,
        episode_length: int = config.EPISODE_LENGTH,
        use_inventory: bool = config.USE_INVENTORY,
        reward_mode: str = "revenue",
        render_mode: str | None = None,
    ):
        super().__init__()
        if reward_mode not in ("revenue", "shaped"):
            raise ValueError("reward_mode must be 'revenue' or 'shaped'")
        self.params = market_params or load_market_params()
        self.episode_length = episode_length
        self.use_inventory = use_inventory
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        # ---- Market parameters derived from the dataset -------------------
        self.segment_names = list(self.params["segments"].keys())
        seg = [self.params["segments"][n] for n in self.segment_names]
        self.n_segments = len(seg)
        self.ref_price = float(self.params["reference_price"])
        self.seg_share = np.array([s["share"] for s in seg], dtype=np.float64)
        self.seg_share /= self.seg_share.sum()
        self.seg_ref_price = np.array([s["reference_price"] for s in seg])
        self.seg_base_demand = np.array([s["base_demand"] for s in seg])
        self.seg_beta = np.array([s["price_sensitivity"] for s in seg])
        self.weekday_trend = np.array(self.params["weekday_trend"], dtype=np.float64)

        self.min_price = config.MIN_PRICE_FACTOR * self.ref_price
        self.max_price = config.MAX_PRICE_FACTOR * self.ref_price
        self.multipliers = np.array(config.PRICE_MULTIPLIERS)

        # Normalisation constants for the observation
        self.expected_daily_demand = float(self.seg_share @ self.seg_base_demand)
        self.sales_scale = 2.0 * self.seg_base_demand.max()
        self.initial_inventory = config.INVENTORY_BUFFER * self.expected_daily_demand * episode_length
        self.daily_restock = config.DAILY_RESTOCK_FRACTION * self.initial_inventory

        # ---- Spaces -------------------------------------------------------
        # 0: decrease 10% | 1: decrease 5% | 2: keep | 3: increase 5% | 4: increase 10%
        self.action_space = spaces.Discrete(len(self.multipliers))

        # [price/ref, segment one-hot (3), last k sales, demand trend,
        #  inventory fraction, time remaining]
        self.obs_dim = 1 + self.n_segments + config.SALES_HISTORY + 3
        low = np.zeros(self.obs_dim, dtype=np.float32)
        high = np.full(self.obs_dim, 3.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode state (initialised in reset)
        self.price = self.ref_price
        self.segment = 0
        self.day = 0
        self.weekday = 0
        self.inventory = self.initial_inventory
        self.sales_history = np.zeros(config.SALES_HISTORY)
        self.total_revenue = 0.0
        self.last_info: dict = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def expected_demand(self, price: float, segment: int, weekday: int) -> float:
        """Mean units demanded at a given price (before inventory limits)."""
        rel = price / self.seg_ref_price[segment]
        return float(
            self.seg_base_demand[segment]
            * self.weekday_trend[weekday]
            * np.exp(-self.seg_beta[segment] * (rel - 1.0))
        )

    def optimal_price(self, segment: int) -> float:
        """Revenue-maximising price for a segment (ignoring inventory)."""
        p = self.seg_ref_price[segment] / self.seg_beta[segment]
        return float(np.clip(p, self.min_price, self.max_price))

    def _next_segment(self) -> int:
        # Sticky Markov chain: the same customer group tends to dominate several
        # days in a row (campaigns, school holidays, pay days, ...)
        if self.np_random.random() < config.SEGMENT_STICKINESS:
            return self.segment
        return int(self.np_random.choice(self.n_segments, p=self.seg_share))

    def _get_obs(self) -> np.ndarray:
        one_hot = np.zeros(self.n_segments)
        one_hot[self.segment] = 1.0
        inv_frac = self.inventory / self.initial_inventory if self.use_inventory else 1.0
        obs = np.concatenate([
            [self.price / self.ref_price],
            one_hot,
            self.sales_history / self.sales_scale,
            [self.weekday_trend[self.weekday]],
            [inv_frac],
            [(self.episode_length - self.day) / self.episode_length],
        ])
        return np.clip(obs, 0.0, 3.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "initial_price" in options:
            self.price = float(np.clip(options["initial_price"], self.min_price, self.max_price))
        else:
            lo, hi = config.INITIAL_PRICE_RANGE
            self.price = float(self.np_random.uniform(lo, hi) * self.ref_price)
        self.segment = int(self.np_random.choice(self.n_segments, p=self.seg_share))
        self.day = 0
        self.weekday = int(self.np_random.integers(7))
        self.inventory = self.initial_inventory
        self.sales_history = np.full(config.SALES_HISTORY, self.expected_daily_demand)
        self.total_revenue = 0.0
        self.last_info = self._info(quantity=0, demand=0.0, revenue=0.0, reward=0.0)
        return self._get_obs(), self.last_info

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # 1) Apply the price adjustment
        self.price = float(np.clip(self.price * self.multipliers[action], self.min_price, self.max_price))

        # 2) Customers of today's segment react to the price
        mean_demand = self.expected_demand(self.price, self.segment, self.weekday)
        demand = int(self.np_random.poisson(mean_demand))
        quantity = min(demand, int(self.inventory)) if self.use_inventory else demand
        revenue = self.price * quantity

        # 3) Reward
        reward = revenue
        if self.reward_mode == "shaped":
            unmet = demand - quantity
            leftover = self.inventory - quantity
            reward -= config.STOCKOUT_PENALTY_PER_UNIT * unmet
            reward -= config.HOLDING_COST_PER_UNIT * leftover

        # 4) Update state
        if self.use_inventory:
            self.inventory = self.inventory - quantity + self.daily_restock
        self.sales_history = np.roll(self.sales_history, -1)
        self.sales_history[-1] = quantity
        self.total_revenue += revenue
        segment_today = self.segment
        self.day += 1
        self.weekday = (self.weekday + 1) % 7
        self.segment = self._next_segment()

        terminated = self.day >= self.episode_length
        self.last_info = self._info(quantity, mean_demand, revenue, reward, segment_today, demand)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, False, self.last_info

    def _info(self, quantity, demand, revenue, reward, segment=None, raw_demand=None) -> dict:
        segment = self.segment if segment is None else segment
        return {
            "day": self.day,
            "price": self.price,
            "quantity": int(quantity),
            "expected_demand": float(demand),
            "demand": int(raw_demand if raw_demand is not None else quantity),
            "revenue": float(revenue),
            "reward": float(reward),
            "total_revenue": float(self.total_revenue),
            "segment": self.segment_names[segment],
            "segment_id": int(segment),
            "inventory": float(self.inventory),
        }

    def render(self):
        i = self.last_info
        text = (
            f"Day {i['day']:>2} | segment {i['segment']:<6} | price {i['price']:8.2f} | "
            f"sold {i['quantity']:>3} | revenue {i['revenue']:9.2f} | "
            f"total {i['total_revenue']:10.2f} | stock {i['inventory']:7.1f}"
        )
        if self.render_mode == "ansi":
            return text
        print(text)
        return None
