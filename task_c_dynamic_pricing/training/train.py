"""Train PPO / DQN pricing agents with Stable-Baselines3.

Examples (run from task_c_dynamic_pricing/):
    python -m training.train --algo ppo
    python -m training.train --algo dqn --timesteps 200000 --seeds 0 1 2
    python -m training.train --algo ppo --timesteps 20000 --seeds 0      # quick smoke test
"""
from __future__ import annotations

import argparse
import time

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from env.pricing_env import UrbanStylePricingEnv

ALGOS = {"ppo": PPO, "dqn": DQN}


def model_path(algo: str, seed: int) -> str:
    return str(config.MODELS_DIR / f"{algo}_urbanstyle_seed{seed}.zip")


def load_model(algo: str = "ppo", seed: int = 0):
    """Load a trained agent for evaluation or for the Pygame dashboard."""
    return ALGOS[algo].load(model_path(algo, seed), device="cpu")


def make_env(algo: str, seed: int, n_envs: int):
    """Vectorised training environment.

    PPO collects experience from several environments in parallel; DQN learns
    off-policy from a replay buffer, so one environment is enough. Rewards are
    revenues in currency units (thousands per day), which is far from the
    ~unit scale neural networks prefer, so VecNormalize rescales the *reward*
    during training only. Observations are already normalised by the
    environment, which means the saved model needs no wrapper at inference.
    """
    monitor_dir = config.LOGS_DIR / f"{algo}_seed{seed}"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(
        UrbanStylePricingEnv,
        n_envs=n_envs if algo == "ppo" else 1,
        seed=seed,
        monitor_dir=str(monitor_dir),
    )
    return VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=0.99)


def train(algo: str, seed: int, timesteps: int, n_envs: int = config.N_ENVS):
    vec_env = make_env(algo, seed, n_envs)
    kwargs = config.PPO_KWARGS if algo == "ppo" else config.DQN_KWARGS
    model = ALGOS[algo]("MlpPolicy", vec_env, seed=seed, verbose=0, device="cpu", **kwargs)

    # Periodic evaluation on separate episodes; keeps the best checkpoint.
    # Wrapped in a frozen VecNormalize so SB3 can sync it with the training
    # env, while still reporting raw (unnormalised) revenue.
    eval_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(UrbanStylePricingEnv())]),
        training=False, norm_obs=False, norm_reward=False,
    )
    eval_env.seed(config.EVAL_SEED_OFFSET + 999 + seed)
    best_dir = config.MODELS_DIR / f"best_{algo}_seed{seed}"
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(config.LOGS_DIR / f"{algo}_seed{seed}_eval"),
        eval_freq=max(10_000 // (n_envs if algo == "ppo" else 1), 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    start = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    model.save(model_path(algo, seed))
    print(f"[{algo.upper()} seed={seed}] trained {timesteps} steps in {time.time() - start:.0f}s "
          f"-> {model_path(algo, seed)}  (best eval reward {eval_cb.best_mean_reward:.0f})")
    vec_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train dynamic pricing RL agents")
    parser.add_argument("--algo", choices=list(ALGOS), default="ppo")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--n-envs", type=int, default=config.N_ENVS)
    args = parser.parse_args()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    check_env(UrbanStylePricingEnv(), warn=True)  # validates the Gymnasium API contract

    for seed in args.seeds:
        model = train(args.algo, seed, args.timesteps, args.n_envs)

    # Demonstrate save/load round-trip
    reloaded = load_model(args.algo, args.seeds[-1])
    env = UrbanStylePricingEnv()
    obs, _ = env.reset(seed=123)
    a_saved, _ = model.predict(obs, deterministic=True)
    a_loaded, _ = reloaded.predict(obs, deterministic=True)
    assert int(a_saved) == int(a_loaded), "loaded model differs from trained model"
    print("Model reload check passed.")


if __name__ == "__main__":
    main()
