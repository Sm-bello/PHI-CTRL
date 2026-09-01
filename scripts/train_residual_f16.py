#!/usr/bin/env python3
"""
PHI-CTRL: Train the residual PPO policy against the REAL F16 JSBSim plant.

Supports a pre-computed altitude×airspeed trim grid (solved once at env
construction) and a simple curriculum that starts on the nominal 15k/400
point and widens to the full grid.

Usage:
  python scripts/train_residual_f16.py --smoke-test
  python scripts/train_residual_f16.py --timesteps 500000
  python scripts/train_residual_f16.py --timesteps 500000 --curriculum
  python scripts/train_residual_f16.py --timesteps 300000 --no-randomize-trim
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(HERE))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_env.jsbsim_phi_ctrl_env_f16 import JSBSimF16PhiCtrlEnv


class CurriculumCallback(BaseCallback):
    """
    Linear curriculum on trim-grid sampling:
      first `warmup_fraction` of training → phase=0 (nominal only)
      then linearly ramp phase → 1.0 by the end of training.

    Calls set_curriculum_phase on every sub-env of the VecEnv.
    """

    def __init__(self, total_timesteps: int, warmup_fraction: float = 0.30, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.warmup_fraction = float(np.clip(warmup_fraction, 0.0, 0.9))
        self._last_logged_phase = -1.0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        if progress <= self.warmup_fraction:
            phase = 0.0
        else:
            # map [warmup, 1] → [0, 1]
            phase = (progress - self.warmup_fraction) / (1.0 - self.warmup_fraction)
            phase = float(np.clip(phase, 0.0, 1.0))

        # Push phase into every env in the vec
        envs = getattr(self.training_env, "envs", None)
        if envs is not None:
            for e in envs:
                # unwrap Monitor / TimeLimit if present
                base = e
                while hasattr(base, "env"):
                    if isinstance(base, JSBSimF16PhiCtrlEnv):
                        break
                    base = base.env
                if isinstance(base, JSBSimF16PhiCtrlEnv):
                    base.set_curriculum_phase(phase)
                elif hasattr(base, "set_curriculum_phase"):
                    base.set_curriculum_phase(phase)

        if self.verbose and abs(phase - self._last_logged_phase) >= 0.1:
            print(f"[CURRICULUM] t={self.num_timesteps}  phase={phase:.2f}")
            self._last_logged_phase = phase
        return True


def make_env_fn(randomize_trim: bool, curriculum_phase: float):
    def _init():
        return JSBSimF16PhiCtrlEnv(
            max_episode_time_s=20.0,
            randomize_trim=randomize_trim,
            curriculum_phase=curriculum_phase,
        )
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--out",
        type=str,
        default=str(HERE / "models" / "phi_ctrl_residual_f16.zip"),
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable trim-grid curriculum (nominal → full grid).",
    )
    parser.add_argument(
        "--no-randomize-trim",
        action="store_true",
        help="Force single nominal trim (old behaviour).",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.30,
        help="Fraction of training spent on nominal-only before widening.",
    )
    args = parser.parse_args()

    timesteps = 5_000 if args.smoke_test else args.timesteps
    randomize = not args.no_randomize_trim
    # Start curriculum at 0 if enabled; otherwise open full grid immediately
    init_phase = 0.0 if args.curriculum else (1.0 if randomize else 0.0)

    print(
        f"Building {args.n_envs} F16 env(s) "
        f"(trim grid solved once per env at startup; "
        f"randomize={randomize}, curriculum={args.curriculum})..."
    )
    vec_env = make_vec_env(
        make_env_fn(randomize_trim=randomize, curriculum_phase=init_phase),
        n_envs=args.n_envs,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048 // args.n_envs if args.n_envs > 1 else 512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    ckpt_dir = HERE / "models" / "checkpoints_f16"
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="phi_ctrl_residual_f16",
    )

    callbacks = [checkpoint_cb]
    if args.curriculum and randomize:
        callbacks.append(
            CurriculumCallback(
                total_timesteps=timesteps,
                warmup_fraction=args.warmup_fraction,
                verbose=1,
            )
        )

    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    model.save(args.out)
    print(f"Saved: {args.out}")

    if args.smoke_test:
        print("\n[SMOKE TEST] One evaluation episode...")
        env = make_env_fn(randomize_trim=randomize, curriculum_phase=1.0 if randomize else 0.0)()
        obs, info = env.reset(seed=0)
        print(f"[SMOKE TEST] reset trim: h={info.get('trim_alt_ft')} Vc={info.get('trim_vc_kts')}")
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"[SMOKE TEST] steps={steps}, terminated={terminated}, total_reward={total_reward:.1f}")
        print("[SMOKE TEST] Pipeline OK. Use --timesteps 500000 (+ --curriculum) for a real run.")


if __name__ == "__main__":
    main()
