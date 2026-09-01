#!/usr/bin/env python3
"""
PHI-CTRL F-16 fault-telemetry dataset generator (V2 — stable episodes)
======================================================================
Each episode creates a **fresh** JSBSim executive (required — reusing a
crashed FDM leaves the aircraft in a tumble and yields empty fault labels).

γ = remaining elevator effectiveness (1.0 healthy … 0.5 = 50% loss).

Usage (Windows one-liner):
  python scripts\\generate_fault_dataset_f16.py --smoke
  python scripts\\generate_fault_dataset_f16.py --episodes-per-gamma 50 --gammas 1.0 0.8 0.6 0.5 --out data\\phi_ctrl_f16_fault
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import jsbsim

from plant.jsbsim_plant_f16 import (
    DT, native_trim, ownership, force_ic, set_throttle, set_elev, set_pitch_trim,
    flight_state, wing_level, PROP_AIL, PROP_RUD, sget,
)
from controller.energy_hold_f16 import EnergyHold

TELEMETRY_FIELDS = [
    "episode_id", "time_s", "alt_ft", "vc_kts", "alpha_deg", "beta_deg",
    "theta_deg", "phi_deg", "psi_deg", "p_dps", "q_dps", "r_dps", "hdot_fps",
    "elevator_cmd", "elevator_pos", "aileron_cmd", "rudder_cmd",
    "throttle", "speedbrake", "thrust_lbs",
    "gamma_remaining", "loss_percent", "fault_active", "fault_type",
    "fault_onset_s", "seed", "target_alt_ft", "target_vc_kts",
]


def _deg(fdm, prop, default=0.0):
    try:
        return math.degrees(float(fdm.get_property_value(prop)))
    except Exception:
        return default


def make_fdm():
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise RuntimeError("Failed to load f16")
    return fdm


def snapshot_row(fdm, episode_id, t, cmds, elev_cmd, gamma, fault_active,
                 fault_onset, seed, target_alt, target_vc):
    st = flight_state(fdm)
    return {
        "episode_id": episode_id,
        "time_s": round(t, 6),
        "alt_ft": st["h"],
        "vc_kts": st["vc"],
        "alpha_deg": st["alpha"],
        "beta_deg": _deg(fdm, "aero/beta-rad", 0.0),
        "theta_deg": st["theta"],
        "phi_deg": _deg(fdm, "attitude/phi-rad", 0.0),
        "psi_deg": _deg(fdm, "attitude/psi-rad", 0.0),
        "p_dps": _deg(fdm, "velocities/p-rad_sec", 0.0),
        "q_dps": st["q"],
        "r_dps": _deg(fdm, "velocities/r-rad_sec", 0.0),
        "hdot_fps": st["hdot"],
        "elevator_cmd": elev_cmd,
        "elevator_pos": sget(fdm, "fcs/elevator-pos-norm", elev_cmd) or elev_cmd,
        "aileron_cmd": cmds.get("ail", 0.0),
        "rudder_cmd": cmds.get("rud", 0.0),
        "throttle": cmds.get("throttle", 0.0),
        "speedbrake": cmds.get("speedbrake", 0.0),
        "thrust_lbs": sget(fdm, "propulsion/engine[0]/thrust-lbs", 0.0) or 0.0,
        "gamma_remaining": float(gamma) if fault_active else 1.0,
        "loss_percent": (100.0 * (1.0 - float(gamma))) if fault_active else 0.0,
        "fault_active": int(bool(fault_active)),
        "fault_type": "elev_effectiveness" if fault_active else "none",
        "fault_onset_s": fault_onset,
        "seed": seed,
        "target_alt_ft": target_alt,
        "target_vc_kts": target_vc,
    }


def run_episode(
    episode_id, gamma_remaining, seed, duration_s, settle_s, fault_onset_s,
    log_every_n, alt_ft, vc_kts, writer,
):
    rng = np.random.default_rng(seed)
    alt = alt_ft + float(rng.uniform(-40.0, 40.0))
    vc = vc_kts + float(rng.uniform(-4.0, 4.0))
    onset = float(np.clip(fault_onset_s + rng.uniform(-0.5, 0.5), 3.0, max(4.0, duration_s - 3.0)))

    env = {"alt_ft": alt, "vc_kts": vc, "theta_seed": 2.5, "desc": f"ds{episode_id}"}
    fdm = make_fdm()

    ok, thr, elev, ptrim, theta = native_trim(fdm, env)
    if not ok:
        return {"episode_id": episode_id, "ok": False, "reason": "trim_fail",
                "gamma_remaining": gamma_remaining, "seed": seed, "n_rows": 0, "crash": False}

    force_ic(fdm, env)
    set_throttle(fdm, thr)
    set_elev(fdm, elev)
    set_pitch_trim(fdm, ptrim)
    wing_level(fdm)
    ownership(fdm, ptrim, 0.0)
    fdm.run_ic()

    # Extra IC steps at trim before closed-loop settle
    for _ in range(int(1.0 / DT)):
        set_elev(fdm, elev)
        set_pitch_trim(fdm, ptrim)
        set_throttle(fdm, thr)
        wing_level(fdm)
        ownership(fdm, ptrim, 0.0)
        fdm.run()

    ctrl = EnergyHold(thr, elev, ptrim, theta, DT)
    for _ in range(int(settle_s / DT)):
        cmds = ctrl.update(fdm, alt, vc)
        set_elev(fdm, cmds["elev"])
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

    st0 = flight_state(fdm)
    # Reject bad settle — do not log tumbling data
    if abs(st0["theta"]) > 15.0 or abs(st0["hdot"]) > 80.0 or abs(st0["q"]) > 20.0:
        return {
            "episode_id": episode_id, "ok": False, "reason": "settle_fail",
            "gamma_remaining": gamma_remaining, "seed": seed, "n_rows": 0, "crash": False,
            "fault_onset_s": onset, "target_alt_ft": alt, "target_vc_kts": vc,
        }

    ctrl.elev0 = st0["elev_cmd"] if abs(st0["elev_cmd"]) > 1e-6 else elev
    ctrl.thr0 = max(st0["thr"] if st0["thr"] else thr, 0.0)
    ctrl.theta0 = st0["theta"]
    ctrl.prev_elev = ctrl.elev0
    ctrl.prev_thr = ctrl.thr0

    n = int(duration_s / DT)
    fault_step = int(onset / DT)
    n_rows = 0
    crash = False
    t0 = fdm.get_sim_time()

    for i in range(n):
        t = fdm.get_sim_time() - t0
        fault_active = (i >= fault_step) and (gamma_remaining < 0.999)
        g = float(gamma_remaining) if fault_active else 1.0

        cmds = ctrl.update(fdm, alt, vc)
        elev_cmd = float(cmds["elev"])
        elev_plant = float(np.clip(elev_cmd * g, -1.0, 1.0))

        set_elev(fdm, elev_plant)
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])

        if not fdm.run():
            crash = True
            break

        if i % log_every_n == 0:
            writer.writerow(snapshot_row(
                fdm, episode_id, t, cmds, elev_cmd, g, fault_active,
                onset, seed, alt, vc,
            ))
            n_rows += 1

        st = flight_state(fdm)
        if abs(st["theta"]) > 50.0 or st["h"] < 0.4 * alt:
            crash = True
            break

    return {
        "episode_id": episode_id,
        "ok": n_rows > 10,
        "reason": "crash" if crash else ("complete" if n_rows > 10 else "too_short"),
        "gamma_remaining": gamma_remaining,
        "loss_percent": 100.0 * (1.0 - gamma_remaining),
        "seed": seed,
        "fault_onset_s": onset,
        "target_alt_ft": alt,
        "target_vc_kts": vc,
        "n_rows": n_rows,
        "crash": crash,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(HERE / "data" / "phi_ctrl_f16_fault"))
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--episodes-per-gamma", type=int, default=40)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--fault-onset", type=float, default=5.0)
    ap.add_argument("--alt", type=float, default=15000.0)
    ap.add_argument("--vc", type=float, default=400.0)
    ap.add_argument("--log-hz", type=float, default=20.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed0", type=int, default=42)
    args = ap.parse_args()

    if args.smoke:
        args.episodes_per_gamma = 2
        args.duration = 12.0
        args.settle = 6.0
        args.gammas = [1.0, 0.8, 0.5]

    log_hz = float(np.clip(args.log_hz, 1.0, 1.0 / DT))
    log_every_n = max(1, int(round((1.0 / DT) / log_hz)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHI-CTRL F-16 dataset generator V2 (fresh FDM per episode)")
    print(f"  gammas={args.gammas}  ep/g={args.episodes_per_gamma}  log={log_hz}Hz")
    print(f"  out={out}")
    print("=" * 70)

    index_rows = []
    episode_id = 0
    t0 = time.time()

    with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TELEMETRY_FIELDS)
        writer.writeheader()
        for g in args.gammas:
            for _k in range(args.episodes_per_gamma):
                seed = args.seed0 + episode_id * 997 + int(g * 1000)
                summary = run_episode(
                    episode_id, float(g), seed,
                    args.duration, args.settle, args.fault_onset,
                    log_every_n, args.alt, args.vc, writer,
                )
                index_rows.append(summary)
                print(
                    f"  ep={episode_id:04d} γ={g:.2f} rows={summary.get('n_rows', 0):4d} "
                    f"{summary.get('reason', '?')}"
                )
                episode_id += 1

    fields = [
        "episode_id", "ok", "reason", "gamma_remaining", "loss_percent",
        "seed", "fault_onset_s", "target_alt_ft", "target_vc_kts", "n_rows", "crash",
    ]
    with open(out / "episode_index.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    n_ok = sum(1 for r in index_rows if r.get("ok"))
    n_fault_rows = 0
    # quick scan
    try:
        import pandas as pd
        df = pd.read_csv(out / "episodes.csv")
        n_fault_rows = int((df["fault_active"] == 1).sum())
    except Exception:
        pass

    manifest = {
        "description": "PHI-CTRL F-16 elev-effectiveness fault telemetry (V2)",
        "gamma_meaning": "remaining_effectiveness",
        "n_episodes": len(index_rows),
        "n_episodes_ok": n_ok,
        "n_fault_active_rows": n_fault_rows,
        "gammas": list(args.gammas),
        "episodes_per_gamma": args.episodes_per_gamma,
        "duration_s": args.duration,
        "settle_s": args.settle,
        "log_hz": log_hz,
        "telemetry_fields": TELEMETRY_FIELDS,
        "wall_time_s": round(time.time() - t0, 2),
        "note": "Fresh FDM per episode. Reject settle failures. Train PHI-Twin on this.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("=" * 70)
    print(f"OK episodes: {n_ok}/{len(index_rows)}  fault_active rows≈{n_fault_rows}")
    print(f"Wrote {out / 'episodes.csv'}")
    print("=" * 70)
    if n_ok < max(3, 0.3 * len(index_rows)):
        print("[WARN] Few OK episodes — check JSBSim f16 model / baseline settle.")
        sys.exit(2)


if __name__ == "__main__":
    main()
