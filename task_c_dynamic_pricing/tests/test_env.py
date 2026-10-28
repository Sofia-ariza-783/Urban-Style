"""Sanity tests for the pricing environment.  Run:  python -m pytest tests"""
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import config
from env.pricing_env import UrbanStylePricingEnv


@pytest.fixture
def env():
    return UrbanStylePricingEnv()


def test_gymnasium_api(env):
    check_env(env, warn=True)


def test_reward_is_revenue(env):
    env.reset(seed=0)
    for _ in range(10):
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert reward == pytest.approx(info["price"] * info["quantity"])


def test_price_stays_in_bounds(env):
    env.reset(seed=1, options={"initial_price": env.max_price})
    for _ in range(config.EPISODE_LENGTH):
        _, _, done, _, info = env.step(4)  # keep increasing
        assert env.min_price <= info["price"] <= env.max_price
        if done:
            break
    env.reset(seed=1, options={"initial_price": env.min_price})
    _, _, _, _, info = env.step(0)
    assert info["price"] == pytest.approx(env.min_price)


def test_episode_length_and_observation_bounds(env):
    obs, _ = env.reset(seed=2)
    steps, done = 0, False
    while not done:
        assert env.observation_space.contains(obs)
        obs, _, done, truncated, _ = env.step(2)
        steps += 1
    assert steps == config.EPISODE_LENGTH and not truncated


def test_seed_reproducibility():
    def rollout(seed):
        e = UrbanStylePricingEnv()
        e.reset(seed=seed)
        return [e.step(a)[1] for a in [0, 1, 2, 3, 4] * 6]
    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_segment_optimal_prices_differ(env):
    # The premise of dynamic pricing: the best price depends on the segment
    optima = [env.optimal_price(s) for s in range(env.n_segments)]
    assert max(optima) / min(optima) > 1.5


def test_lower_price_increases_demand(env):
    lo = env.expected_demand(0.8 * env.ref_price, 0, 0)
    hi = env.expected_demand(1.2 * env.ref_price, 0, 0)
    assert lo > hi
