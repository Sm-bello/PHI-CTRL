#!/usr/bin/env python3
"""
PHI-CTRL evaluation harness: envelope × fault severity.

Runs short closed-loop scenarios using the F-16 plant + EnergyHold baseline
(and optionally a loaded residual policy) across a small trim grid and
γ ∈ {1.0, 0.8, 0.6}.

Usage:
  python eval/eval_residual_grid.py
  python eval/eval_residual_grid.py --with-residual --model models/phi_ctrl_residual_f16.zip
  python eval/eval_residual_grid.py --out results/eval_grid
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jsbsim

from plant.jsbsim_plant_f16 import (
    DT, native_trim, ownership, force_ic, set_throttle, set_elev, set_pitch_trim,
    flight_state, PROP_AIL, PROP_RUD,
)
from controller.energy_hold_f16 import EnergyHold

GRID = [
    {"alt_ft": 10000.0, "vc_kts": 400.0, "theta_seed": 2.5},
    {"alt_ft": 15000.0, "vc_kts": 400.0, "theta_seed": 2.5},
    {"alt_ft": 20000.0, "vc_kts": 400.0, "theta_seed": 2.5},
]
GAMMAS = [1.0, 0.8, 0.6]
DURATION_S = 30.0
FAULT_S = 8.0
SETTLE_S = 5.0


def run_case(alt, vc, gamma_fault, residual_model=None):
    env = {"alt_ft": alt, "vc_kts": vc, "theta_seed": 2.5, "desc": "eval"}
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise RuntimeError("load f16 failed")

    ok, thr, elev, ptrim, theta = native_trim(fdm, env)
    if not ok:
        return {"ok": False, "alt": alt, "vc": vc, "gamma": gamma_fault}

    force_ic(fdm, env)
    set_throttle(fdm, thr)
    set_elev(fdm, elev)
    set_pitch_trim(fdm, ptrim)
    ownership(fdm, ptrim, 0.0)
    fdm.run_ic()

    ctrl = EnergyHold(thr, elev, ptrim, theta, DT)
    for _ in range(int(SETTLE_S / DT)):
        cmds = ctrl.update(fdm, alt, vc)
        set_elev(fdm, cmds["elev"])
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

    st0 = flight_state(fdm)
    h0 = st0["h"]
    n = int(DURATION_S / DT)
    fault_step = int(FAULT_S / DT)
    max_abs_dh = 0.0
    min_theta = 99.0
    max_theta = -99.0
    crash = False

    for i in range(n):
        st = flight_state(fdm)
        g = gamma_fault if i >= fault_step else 1.0
        cmds = ctrl.update(fdm, alt, vc)
        elev_cmd = cmds["elev"]
        residual = 0.0
        if residual_model is not None and i >= fault_step:
            # Observation vector matches gym env order (best-effort)
            try:
                u = fdm.get_property_value("velocities/u-fps")
                w = fdm.get_property_value("velocities/w-fps")
                q = math.radians(st["q"])
                th = math.radians(st["theta"])
                obs = np.array(
                    [u, w, q, th, st["h"], alt, elev_cmd, g], dtype=np.float32
                )
                action, _ = residual_model.predict(obs, deterministic=True)
                residual = float(np.clip(action[0], -0.5, 0.5))
            except Exception:
                residual = 0.0
        elev_out = float(np.clip((elev_cmd + residual) * g, -1.0, 1.0))
        set_elev(fdm, elev_out)
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        if not fdm.run():
            crash = True
            break
        st = flight_state(fdm)
        dh = abs(st["h"] - h0)
        max_abs_dh = max(max_abs_dh, dh)
        min_theta = min(min_theta, st["theta"])
        max_theta = max(max_theta, st["theta"])
        if abs(st["theta"]) > 50 or st["h"] < 0.4 * alt:
            crash = True
            break

    return {
        "ok": True,
        "alt": alt,
        "vc": vc,
        "gamma": gamma_fault,
        "max_abs_dh": max_abs_dh,
        "min_theta": min_theta,
        "max_theta": max_theta,
        "crash": crash,
        "residual": residual_model is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(HERE / "results" / "eval_grid"))
    ap.add_argument("--with-residual", action="store_true")
    ap.add_argument("--model", type=str, default=str(HERE / "models" / "phi_ctrl_residual_f16.zip"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    residual_model = None
    if args.with_residual:
        try:
            from stable_baselines3 import PPO
            residual_model = PPO.load(args.model)
            print(f"[EVAL] Loaded residual: {args.model}")
        except Exception as e:
            print(f"[EVAL] Could not load residual ({e}); running baseline only")

    rows = []
    for pt in GRID:
        for g in GAMMAS:
            print(f"[EVAL] h={pt['alt_ft']:.0f} Vc={pt['vc_kts']:.0f} γ={g} ...")
            row = run_case(pt["alt_ft"], pt["vc_kts"], g, residual_model)
            rows.append(row)
            if row.get("ok"):
                print(
                    f"       max|Δh|={row['max_abs_dh']:.1f}  "
                    f"θ=[{row['min_theta']:.1f},{row['max_theta']:.1f}]  crash={row['crash']}"
                )
            else:
                print("       TRIM FAIL")

    csv_path = out / "eval_grid_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ok", "alt", "vc", "gamma", "max_abs_dh", "min_theta", "max_theta",
                "crash", "residual",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[EVAL] Wrote {csv_path}")


if __name__ == "__main__":
    main()
