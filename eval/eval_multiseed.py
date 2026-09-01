#!/usr/bin/env python3
"""
PHI-CTRL multi-seed evaluation (V2 — aligned with unified plant path)
======================================================================
Uses the same trim → settle → EnergyHold → fault → (optional) residual
law as phi_ctrl_unified_f16.py so numbers are comparable.

Reports mean/std of max|Δh|, θ range, crash rate across seeds.

Usage (Windows one-liners):
  python eval\\eval_multiseed.py --seeds 20 --gamma 1.0 0.8 0.5
  python eval\\eval_multiseed.py --seeds 10 --gamma 0.8 0.5 --mode hybrid
  python eval\\eval_multiseed.py --seeds 10 --gamma 0.5 --mode full --with-residual
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jsbsim

from plant.jsbsim_plant_f16 import (
    DT, native_trim, ownership, force_ic, set_throttle, set_elev, set_pitch_trim,
    flight_state, wing_level, PROP_AIL, PROP_RUD,
)
from controller.energy_hold_f16 import EnergyHold
from detector.mmae_bank import ElevEffectivenessBank

SETTLE_S = 12.0
DURATION_S = 45.0
FAULT_S = 12.0
RESIDUAL_ENABLE_GAMMA_HAT = 0.92
RESIDUAL_MAX = 0.20
RESIDUAL_GAIN = 0.35
RESIDUAL_RATE_MAX = 0.05
BAILOUT_THETA = 60.0
BAILOUT_ALT_FRAC = 0.35


def make_fdm():
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise RuntimeError("Failed to load f16")
    return fdm


def run_once(alt, vc, gamma_remaining, seed, mode="baseline", residual_model=None):
    """
    mode:
      baseline — EnergyHold only, physical γ applied, no compensation knowledge
      hybrid   — EnergyHold + MMAE 1/γ compensation
      full     — hybrid + optional PPO residual (scaled, rate-limited)
    """
    rng = np.random.default_rng(seed)
    alt_j = alt + float(rng.uniform(-40.0, 40.0))
    vc_j = vc + float(rng.uniform(-4.0, 4.0))
    fault_s = float(np.clip(FAULT_S + rng.uniform(-1.0, 1.0), 6.0, DURATION_S - 8.0))

    env = {"alt_ft": alt_j, "vc_kts": vc_j, "theta_seed": 2.5, "desc": "mc"}
    fdm = make_fdm()

    ok, thr, elev, ptrim, theta = native_trim(fdm, env)
    if not ok:
        return {
            "ok": False, "seed": seed, "alt": alt, "vc": vc,
            "gamma_remaining": gamma_remaining, "mode": mode,
            "reason": "trim_fail",
        }

    force_ic(fdm, env)
    set_throttle(fdm, thr)
    set_elev(fdm, elev)
    set_pitch_trim(fdm, ptrim)
    wing_level(fdm)
    ownership(fdm, ptrim, 0.0)
    fdm.run_ic()

    for _ in range(int(1.0 / DT)):
        set_elev(fdm, elev)
        set_pitch_trim(fdm, ptrim)
        set_throttle(fdm, thr)
        wing_level(fdm)
        ownership(fdm, ptrim, 0.0)
        fdm.run()

    ctrl = EnergyHold(thr, elev, ptrim, theta, DT)
    for _ in range(int(SETTLE_S / DT)):
        cmds = ctrl.update(fdm, alt_j, vc_j)
        set_elev(fdm, cmds["elev"])
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

    st0 = flight_state(fdm)
    # Reject bad settle — do not count as scientific sample
    if abs(st0["theta"]) > 12.0 or abs(st0["hdot"]) > 60.0 or abs(st0["q"]) > 15.0:
        return {
            "ok": False, "seed": seed, "alt": alt, "vc": vc,
            "gamma_remaining": gamma_remaining, "mode": mode,
            "reason": "settle_fail",
        }

    ctrl.elev0 = st0["elev_cmd"] if abs(st0["elev_cmd"]) > 1e-6 else elev
    ctrl.thr0 = max(st0["thr"] if st0["thr"] else thr, 0.0)
    ctrl.theta0 = st0["theta"]
    ctrl.prev_elev = ctrl.elev0
    ctrl.prev_thr = ctrl.thr0

    h0 = st0["h"]
    bank = ElevEffectivenessBank(dt=DT)
    bank.reset()

    n = int(DURATION_S / DT)
    fault_step = int(fault_s / DT)
    max_abs_dh = 0.0
    min_theta = 99.0
    max_theta = -99.0
    crash = False
    residual_steps = 0
    prev_rl = 0.0
    prev_elev_cmd = ctrl.elev0

    for i in range(n):
        st = flight_state(fdm)
        fault_active = i >= fault_step
        physical_gamma = float(gamma_remaining) if fault_active else 1.0

        cmds = ctrl.update(fdm, alt_j, vc_j)
        elev_raw = float(cmds["elev"])

        bank_out = bank.update(elev_raw, st["q"])
        gamma_hat = float(bank_out["gamma_hat"]) if mode in ("hybrid", "full") else 1.0

        if mode == "baseline":
            comp = 1.0
            gamma_hat = 1.0
        else:
            g_est = gamma_hat if fault_active else 1.0
            # During pre-fault keep comp=1; post-fault use bank
            if fault_active:
                g_est = min(g_est, 0.99)  # avoid div issues
                comp = min(1.0 / max(g_est, 0.05), 4.0)
            else:
                comp = 1.0

        rl_res = 0.0
        if mode == "full" and residual_model is not None and fault_active:
            if gamma_hat < RESIDUAL_ENABLE_GAMMA_HAT:
                residual_steps += 1
                try:
                    obs = np.array([
                        st["vc"] * 1.68781,
                        0.0,
                        math.radians(st["q"]),
                        math.radians(st["theta"]),
                        st["h"],
                        alt_j,
                        prev_elev_cmd,
                        gamma_hat,
                    ], dtype=np.float32)
                    action, _ = residual_model.predict(obs, deterministic=True)
                    raw_a = float(np.clip(float(action[0]), -0.5, 0.5))
                    deficit = float(np.clip(1.0 - gamma_hat, 0.0, 1.0))
                    desired = raw_a * RESIDUAL_GAIN * max(deficit, 0.15)
                    desired = float(np.clip(desired, -RESIDUAL_MAX, RESIDUAL_MAX))
                    du = float(np.clip(desired - prev_rl, -RESIDUAL_RATE_MAX, RESIDUAL_RATE_MAX))
                    rl_res = prev_rl + du
                except Exception:
                    rl_res = prev_rl * 0.9
            else:
                rl_res = prev_rl * 0.85
            prev_rl = rl_res
        else:
            prev_rl *= 0.85

        elev_comp = float(np.clip(elev_raw * comp + rl_res, -1.0, 1.0))
        elev_plant = float(np.clip(elev_comp * physical_gamma, -1.0, 1.0))
        prev_elev_cmd = elev_comp

        set_elev(fdm, elev_plant)
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])

        if not fdm.run():
            crash = True
            break

        st = flight_state(fdm)
        max_abs_dh = max(max_abs_dh, abs(st["h"] - h0))
        min_theta = min(min_theta, st["theta"])
        max_theta = max(max_theta, st["theta"])

        if abs(st["theta"]) > BAILOUT_THETA or st["h"] < BAILOUT_ALT_FRAC * alt:
            crash = True
            break

    return {
        "ok": True,
        "seed": seed,
        "alt": alt,
        "vc": vc,
        "gamma_remaining": gamma_remaining,
        "loss_percent": 100.0 * (1.0 - gamma_remaining),
        "mode": mode,
        "max_abs_dh": float(max_abs_dh),
        "min_theta": float(min_theta),
        "max_theta": float(max_theta),
        "crash": bool(crash),
        "residual_steps": int(residual_steps),
        "with_residual": residual_model is not None and mode == "full",
        "reason": "crash" if crash else "complete",
    }


def main():
    ap = argparse.ArgumentParser(description="PHI-CTRL multi-seed eval V2")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.0, 0.8, 0.5])
    ap.add_argument("--alts", type=float, nargs="+", default=[15000.0])
    ap.add_argument("--vc", type=float, default=400.0)
    ap.add_argument(
        "--mode",
        choices=["baseline", "hybrid", "full", "all"],
        default="all",
        help="baseline | hybrid | full | all (runs baseline+hybrid+full if residual available)",
    )
    ap.add_argument("--with-residual", action="store_true",
                    help="Load PPO residual (needed for mode=full)")
    ap.add_argument("--model", type=str,
                    default=str(HERE / "models" / "phi_ctrl_residual_f16.zip"))
    ap.add_argument("--out", type=str,
                    default=str(HERE / "results" / "eval_multiseed"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    residual_model = None
    if args.with_residual or args.mode in ("full", "all"):
        try:
            from stable_baselines3 import PPO
            if Path(args.model).exists():
                residual_model = PPO.load(args.model, device="cpu")
                print(f"[MC] Loaded residual {args.model}")
            else:
                print(f"[MC] No residual at {args.model}")
        except Exception as e:
            print(f"[MC] Residual load failed: {e}")

    if args.mode == "all":
        modes = ["baseline", "hybrid"]
        if residual_model is not None:
            modes.append("full")
    else:
        modes = [args.mode]
        if args.mode == "full" and residual_model is None:
            print("[MC] mode=full requires residual model; aborting")
            sys.exit(1)

    rows = []
    for alt in args.alts:
        for g in args.gamma:
            for mode in modes:
                print(f"[MC] alt={alt:.0f} γ={g:.2f} mode={mode} seeds={args.seeds}")
                for s in range(args.seeds):
                    seed = 2000 + s * 97 + int(alt) + int(g * 100) + hash(mode) % 1000
                    row = run_once(
                        alt, args.vc, g, seed=seed, mode=mode,
                        residual_model=residual_model if mode == "full" else None,
                    )
                    if row:
                        rows.append(row)
                        status = row.get("reason", "?")
                        dh = row.get("max_abs_dh", float("nan"))
                        print(f"  seed={s:02d}  max|dh|={dh:7.1f}  {status}")

    fields = [
        "ok", "seed", "alt", "vc", "gamma_remaining", "loss_percent", "mode",
        "max_abs_dh", "min_theta", "max_theta", "crash", "residual_steps",
        "with_residual", "reason",
    ]
    runs_path = out / "multiseed_runs.csv"
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    df = pd.DataFrame([r for r in rows if r.get("ok")])
    if df.empty:
        print("[MC] No successful runs — check JSBSim / settle")
        sys.exit(2)

    grp = df.groupby(["alt", "gamma_remaining", "mode"], as_index=False).agg(
        n=("max_abs_dh", "count"),
        mean_max_dh=("max_abs_dh", "mean"),
        std_max_dh=("max_abs_dh", "std"),
        mean_min_theta=("min_theta", "mean"),
        mean_max_theta=("max_theta", "mean"),
        crash_rate=("crash", "mean"),
    )
    # NaN std for n=1
    grp["std_max_dh"] = grp["std_max_dh"].fillna(0.0)

    sum_path = out / "multiseed_summary.csv"
    grp.to_csv(sum_path, index=False)

    print("\n" + "=" * 72)
    print("MULTI-SEED SUMMARY (comparable to unified path)")
    print("=" * 72)
    print(grp.to_string(index=False))
    print("=" * 72)
    print(f"Wrote {runs_path}")
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
