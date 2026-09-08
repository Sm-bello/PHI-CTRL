#!/usr/bin/env python3
"""
PHI-CTRL: Train (or smoke-test) the residual PPO policy against the REAL
JSBSim c172p 6-DOF plant, instead of the linear-ODE approximation the
original phi_ctrl_residual_final.zip checkpoint was trained on.

Usage:
  python train_residual_jsbsim.py --smoke-test       # ~5k steps, sanity check only
  python train_residual_jsbsim.py --timesteps 500000  # a real training run

A real run (500k timesteps, matching the original checkpoint's budget) is
compute-heavy -- JSBSim is not vectorized/GPU-accelerated, so this is CPU-
bound wall-clock time dominated by physics stepping, not network size. Run
it locally with `n_envs` > 1 (SubprocVecEnv) if you have spare cores; do not
expect this to be fast in a constrained sandbox.
"""
import argparse
import io
import contextlib

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_env.jsbsim_phi_ctrl_env import JSBSimPhiCtrlEnv


def make_env():
    def _init():
        with contextlib.redirect_stdout(io.StringIO()):
            env = JSBSimPhiCtrlEnv(max_episode_time_s=20.0)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run only ~5k timesteps to verify the pipeline works end-to-end.")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--out", type=str, default="models/phi_ctrl_residual_jsbsim.zip")
    args = parser.parse_args()

    timesteps = 5_000 if args.smoke_test else args.timesteps

    print(f"Building {args.n_envs} JSBSim env(s) (trim search runs once per env -- may take a bit)...")
    vec_env = make_vec_env(make_env(), n_envs=args.n_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048 // args.n_envs if args.n_envs > 1 else 512,  # smaller for single-env smoke test
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path="models/checkpoints/",
        name_prefix="phi_ctrl_residual_jsbsim",
    )

    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=checkpoint_cb, progress_bar=False)

    model.save(args.out)
    print(f"Saved: {args.out}")

    if args.smoke_test:
        print("\n[SMOKE TEST] Running one evaluation episode with the freshly-trained (undertrained) policy...")
        env = make_env()()
        obs, info = env.reset(seed=0)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"[SMOKE TEST] steps={steps}, terminated={terminated}, total_reward={total_reward:.1f}")
        print("[SMOKE TEST] Pipeline runs end-to-end. This policy is NOT trained enough to be useful --")
        print("[SMOKE TEST] rerun with --timesteps 500000 (or more) for a real result.")


if __name__ == "__main__":
    main()
