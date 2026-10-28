"""Static (non-learning) pricing strategies used as baselines.

Every policy exposes the same interface as an SB3 model:
    action, _ = policy.predict(obs, deterministic=True)
so that the evaluation loop treats RL agents and baselines identically.
"""
from __future__ import annotations

import numpy as np

import config
from env.pricing_env import UrbanStylePricingEnv

KEEP_PRICE = config.PRICE_MULTIPLIERS.index(1.00)


class StaticPricePolicy:
    """Hold one fixed price for the whole episode.

    The environment starts each episode at `price_factor * reference price`
    (passed through reset options) and the policy always picks "keep price".
    """

    def __init__(self, price_factor: float = 1.0, name: str = "Static"):
        self.price_factor = price_factor
        self.name = name

    def reset_options(self, env: UrbanStylePricingEnv) -> dict:
        return {"initial_price": self.price_factor * env.ref_price}

    def predict(self, obs, deterministic: bool = True):
        return KEEP_PRICE, None


class RandomPolicy:
    """Uniformly random price adjustments (lower bound on performance)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.name = "Random"

    def predict(self, obs, deterministic: bool = True):
        return int(self.rng.integers(len(config.PRICE_MULTIPLIERS))), None


def find_best_fixed_price(n_episodes: int = 30, seed: int = 0) -> tuple[float, dict]:
    """Grid-search the single price that maximises mean episode revenue.

    This is the strongest static strategy: a manager who knows the market and
    sets the best possible *constant* price. Uses training seeds only, so the
    choice is not tuned on the evaluation episodes.
    """
    from evaluation.evaluate import run_episodes  # local import avoids a cycle

    env = UrbanStylePricingEnv()
    scores = {}
    for factor in config.FIXED_PRICE_GRID:
        revenues = run_episodes(StaticPricePolicy(factor), env, n_episodes, seed_offset=seed)["revenue"]
        scores[factor] = float(np.mean(revenues))
    best = max(scores, key=scores.get)
    return best, scores
