#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL — F-16 Thrust + Pitch Differential Diagnostic
======================================================
Answers two questions only:
  1) Does throttle actually produce thrust / raise airspeed?
  2) Does elevator-cmd-norm move pitch *relative to a zero-command baseline*?

Natural out-of-trim motion is cancelled by differential measurement.

Usage (same folder / JSBSim root as V12):
  python diagnose_f16_thrust_pitch.py
  python diagnose_f16_thrust_pitch.py --jsbsim-root /path/to/JSBSim
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_diag2"
OUT.mkdir(parents=True, exist_ok=True)

try:
    import jsbsim
except ImportError as e:
    raise SystemExit("jsbsim required") from e

DT = 1.0 / 120.0
ALT_FT = 15000.0
VC_KTS = 400.0


class Tee:
    def __init__(self, path: Path):
        self.t = sys.stdout
        self.f = open(path, "w", encoding="utf-8")
        sys.stdout = self

    def write(self, m):
        self.t.write(m)
        self.f.write(m)
        self.f.flush()

    def flush(self):
        self.t.flush()
        self.f.flush()

    def close(self):
        sys.stdout = self.t
        self.f.close()


def sset(fdm, prop, val):
    try:
        fdm.set_property_value(prop, float(val))
        return True
    except Exception:
        return False


def sget(fdm, prop, default=None):
    try:
        return fdm.get_property_value(prop)
    except Exception:
        return default


def engine_on(fdm):
    """Aggressive engine-alive sequence used by many JSBSim fighters."""
    sset(fdm, "propulsion/magnetos_all", 3)
    sset(fdm, "propulsion/set-running", -1)
    sset(fdm, "propulsion/engine[0]/set-running", 1)
    sset(fdm, "propulsion/engine[0]/starter-cmd", 1)
    sset(fdm, "fcs/mixture-cmd-norm", 1.0)
    sset(fdm, "fcs/throttle-cmd-norm", 0.5)
    # indexed variants some builds use
    sset(fdm, "fcs/throttle-cmd-norm[0]", 0.5)
    sset(fdm, "propulsion/engine[0]/throttle-cmd-norm", 0.5)


def clean_airframe(fdm, fbw=1.0):
    sset(fdm, "fcs/fbw-override", fbw)
    sset(fdm, "gear/gear-cmd-norm", 0.0)
    sset(fdm, "gear/gear-pos-norm", 0.0)
    sset(fdm, "fcs/flap-cmd-norm", 0.0)
    sset(fdm, "fcs/speedbrake-cmd-norm", 0.0)
    sset(fdm, "fcs/aileron-cmd-norm", 0.0)
    sset(fdm, "fcs/rudder-cmd-norm", 0.0)
    sset(fdm, "fcs/pitch-trim-cmd-norm", 0.0)
    sset(fdm, "fcs/roll-trim-cmd-norm", 0.0)
    sset(fdm, "fcs/yaw-trim-cmd-norm", 0.0)
    engine_on(fdm)


def force_ic(fdm):
    fdm.reset_to_initial_conditions(True)
    fdm.set_property_value("ic/h-sl-ft", ALT_FT)
    fdm.set_property_value("ic/vc-kts", VC_KTS)
    fdm.set_property_value("ic/gamma-deg", 0.0)
    fdm.set_property_value("ic/theta-deg", 2.5)
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    fdm.set_property_value("ic/p-rad_sec", 0.0)
    fdm.set_property_value("ic/q-rad_sec", 0.0)
    fdm.set_property_value("ic/r-rad_sec", 0.0)
    clean_airframe(fdm)
    fdm.run_ic()
    clean_airframe(fdm)


def read_propulsion(fdm):
    return {
        "thrust_lbs": sget(fdm, "propulsion/engine[0]/thrust-lbs"),
        "n1": sget(fdm, "propulsion/engine[0]/n1"),
        "n2": sget(fdm, "propulsion/engine[0]/n2"),
        "thr_cmd": sget(fdm, "fcs/throttle-cmd-norm"),
        "thr_cmd0": sget(fdm, "fcs/throttle-cmd-norm[0]"),
        "thr_eng": sget(fdm, "propulsion/engine[0]/throttle-cmd-norm"),
        "thr_pos": sget(fdm, "fcs/throttle-pos-norm"),
        "running": sget(fdm, "propulsion/engine[0]/set-running"),
        "fuel_flow": sget(fdm, "propulsion/engine[0]/fuel-flow-rate-pps"),
    }


def read_flight(fdm):
    return {
        "h": fdm.get_property_value("position/h-sl-ft"),
        "vc": fdm.get_property_value("velocities/vc-kts"),
        "theta": math.degrees(fdm.get_property_value("attitude/theta-rad")),
        "hdot": fdm.get_property_value("velocities/h-dot-fps"),
        "q": math.degrees(fdm.get_property_value("velocities/q-rad_sec")),
        "alpha": math.degrees(fdm.get_property_value("aero/alpha-rad")),
        "elev_cmd": sget(fdm, "fcs/elevator-cmd-norm"),
        "elev_pos": sget(fdm, "fcs/elevator-pos-norm"),
    }


def run_seconds(fdm, seconds, step_fn):
    """Integrate `seconds`, call step_fn(fdm) every frame. Return mean/final metrics."""
    n = int(seconds / DT)
    qs, hds, vcs, ths, thrusts = [], [], [], [], []
    for i in range(n):
        step_fn(fdm)
        clean_airframe(fdm)
        fdm.run()
        if i >= int(0.5 / DT):
            fl = read_flight(fdm)
            pr = read_propulsion(fdm)
            qs.append(fl["q"])
            hds.append(fl["hdot"])
            vcs.append(fl["vc"])
            ths.append(fl["theta"])
            if pr["thrust_lbs"] is not None:
                thrusts.append(pr["thrust_lbs"])
    return {
        "mean_q": float(np.mean(qs)) if qs else 0.0,
        "mean_hdot": float(np.mean(hds)) if hds else 0.0,
        "mean_vc": float(np.mean(vcs)) if vcs else 0.0,
        "final_vc": float(vcs[-1]) if vcs else 0.0,
        "final_theta": float(ths[-1]) if ths else 0.0,
        "mean_thrust": float(np.mean(thrusts)) if thrusts else None,
        "final": read_flight(fdm),
        "prop": read_propulsion(fdm),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_engine_alive(fdm):
    print("\n" + "=" * 70)
    print("A) ENGINE ALIVE CHECK (IC + 2 s at thr=0.6)")
    print("=" * 70)
    force_ic(fdm)

    def step(f):
        sset(f, "fcs/throttle-cmd-norm", 0.60)
        sset(f, "fcs/throttle-cmd-norm[0]", 0.60)
        sset(f, "propulsion/engine[0]/throttle-cmd-norm", 0.60)
        sset(f, "fcs/elevator-cmd-norm", 0.0)

    r = run_seconds(fdm, 2.0, step)
    p = r["prop"]
    print(f"  thrust-lbs     = {p['thrust_lbs']}")
    print(f"  n1 / n2        = {p['n1']} / {p['n2']}")
    print(f"  thr_cmd / [0]  = {p['thr_cmd']} / {p['thr_cmd0']}")
    print(f"  thr_eng        = {p['thr_eng']}")
    print(f"  thr_pos        = {p['thr_pos']}")
    print(f"  fuel-flow-pps  = {p['fuel_flow']}")
    print(f"  Vc             = {r['final_vc']:.1f} kts")
    alive = (p["thrust_lbs"] is not None and p["thrust_lbs"] > 100.0) or (
        p["n1"] is not None and p["n1"] > 20.0
    )
    print(f"  VERDICT: {'ENGINE PRODUCING THRUST' if alive else 'ENGINE APPEARS DEAD / WRONG PROPERTY'}")
    return alive


def test_throttle_ladder(fdm):
    print("\n" + "=" * 70)
    print("B) THROTTLE LADDER (elev frozen 0, thr = 0.2 / 0.5 / 0.8, 5 s each)")
    print("=" * 70)
    print(f"  {'thr':>6s}  {'mean_thrust':>12s}  {'final_Vc':>10s}  {'mean_Vc':>10s}")
    rows = []
    for thr in (0.20, 0.50, 0.80):
        force_ic(fdm)

        def step(f, t=thr):
            sset(f, "fcs/throttle-cmd-norm", t)
            sset(f, "fcs/throttle-cmd-norm[0]", t)
            sset(f, "propulsion/engine[0]/throttle-cmd-norm", t)
            sset(f, "fcs/elevator-cmd-norm", 0.0)

        r = run_seconds(fdm, 5.0, step)
        mt = r["mean_thrust"]
        print(f"  {thr:6.2f}  {str(mt):>12s}  {r['final_vc']:10.1f}  {r['mean_vc']:10.1f}")
        rows.append((thr, mt, r["final_vc"], r["mean_vc"]))

    # Positive thrust slope and non-decreasing speed trend?
    thrusts = [r[1] for r in rows if r[1] is not None]
    vcs = [r[2] for r in rows]
    thrust_ok = len(thrusts) >= 2 and thrusts[-1] > thrusts[0] + 200.0
    # speed may lag; require at least not collapsing with more throttle
    speed_ok = vcs[-1] >= vcs[0] - 15.0
    print(f"  VERDICT thrust slope: {'PASS' if thrust_ok else 'FAIL'}")
    print(f"  VERDICT speed trend:  {'PASS (or neutral)' if speed_ok else 'FAIL'}")
    return thrust_ok


def test_pitch_differential(fdm):
    """
    Same IC, three parallel 4 s holds:
      elev = -0.12 | 0.0 | +0.12
    Report mean_q and final θ relative to elev=0 baseline.
    """
    print("\n" + "=" * 70)
    print("C) PITCH DIFFERENTIAL (elevator-cmd-norm = -0.12 / 0 / +0.12, thr=0.55, 4 s)")
    print("=" * 70)

    results = {}
    for elev in (-0.12, 0.0, 0.12):
        force_ic(fdm)

        def step(f, e=elev):
            sset(f, "fcs/throttle-cmd-norm", 0.55)
            sset(f, "fcs/throttle-cmd-norm[0]", 0.55)
            sset(f, "propulsion/engine[0]/throttle-cmd-norm", 0.55)
            sset(f, "fcs/elevator-cmd-norm", e)
            sset(f, "fcs/pitch-trim-cmd-norm", 0.0)

        r = run_seconds(fdm, 4.0, step)
        results[elev] = r
        print(f"  elev={elev:+.2f}  mean_q={r['mean_q']:+7.2f} deg/s  "
              f"final_θ={r['final_theta']:+7.2f}°  elev_pos={r['final'].get('elev_pos')}")

    base = results[0.0]
    dq_pos = results[0.12]["mean_q"] - base["mean_q"]
    dq_neg = results[-0.12]["mean_q"] - base["mean_q"]
    dth_pos = results[0.12]["final_theta"] - base["final_theta"]
    dth_neg = results[-0.12]["final_theta"] - base["final_theta"]

    print("\n  Differential vs elev=0:")
    print(f"    +0.12:  Δmean_q={dq_pos:+.2f}  Δθ={dth_pos:+.2f}")
    print(f"    -0.12:  Δmean_q={dq_neg:+.2f}  Δθ={dth_neg:+.2f}")

    # Expected aero sign for conventional tail: +elev-cmd often nose-up or nose-down
    # depending on sign convention. We only require *asymmetric* response of useful size.
    authority = (abs(dq_pos) + abs(dq_neg)) > 1.0 or (abs(dth_pos) + abs(dth_neg)) > 2.0
    sign_consistent = (dq_pos * dq_neg) < 0 or (dth_pos * dth_neg) < 0
    print(f"  VERDICT authority:  {'PASS' if authority else 'FAIL'}")
    print(f"  VERDICT opposite sides differ: {'PASS' if sign_consistent else 'WEAK/NONE'}")
    return authority


def test_pitch_trim_assist(fdm):
    """Does pitch-trim-cmd-norm add authority on top of elevator-cmd-norm=0?"""
    print("\n" + "=" * 70)
    print("D) PITCH-TRIM ASSIST (elevator-cmd=0, pitch-trim = -0.1 / 0 / +0.1)")
    print("=" * 70)
    results = {}
    for trim in (-0.10, 0.0, 0.10):
        force_ic(fdm)

        def step(f, t=trim):
            sset(f, "fcs/throttle-cmd-norm", 0.55)
            sset(f, "fcs/elevator-cmd-norm", 0.0)
            sset(f, "fcs/pitch-trim-cmd-norm", t)

        r = run_seconds(fdm, 4.0, step)
        results[trim] = r
        print(f"  trim={trim:+.2f}  mean_q={r['mean_q']:+7.2f}  final_θ={r['final_theta']:+7.2f}")

    base = results[0.0]
    dq = abs(results[0.10]["mean_q"] - base["mean_q"]) + abs(results[-0.10]["mean_q"] - base["mean_q"])
    print(f"  VERDICT pitch-trim authority: {'PASS' if dq > 0.8 else 'WEAK/NONE'}  (|Δq|sum={dq:.2f})")
    return dq > 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsbsim-root", default=os.environ.get("JSBSIM_ROOT") or None)
    ap.add_argument("--aircraft", default="f16")
    args = ap.parse_args()

    tee = Tee(OUT / "console_log.txt")
    try:
        print("=" * 70)
        print("  PHI-CTRL F-16 THRUST + PITCH DIFFERENTIAL DIAGNOSTIC")
        print(f"  aircraft={args.aircraft}  IC {ALT_FT:.0f} ft / {VC_KTS:.0f} kts")
        print("=" * 70)

        fdm = jsbsim.FGFDMExec(args.jsbsim_root)
        fdm.set_dt(DT)
        if not fdm.load_model(args.aircraft):
            raise SystemExit(f"Failed to load {args.aircraft}")

        alive = test_engine_alive(fdm)
        thrust_ok = test_throttle_ladder(fdm)
        pitch_ok = test_pitch_differential(fdm)
        trim_ok = test_pitch_trim_assist(fdm)

        print("\n" + "=" * 70)
        print("SUMMARY — GO / NO-GO FOR V13")
        print("=" * 70)
        print(f"  Engine producing thrust:     {'GO' if alive else 'NO-GO'}")
        print(f"  Throttle → thrust slope:     {'GO' if thrust_ok else 'NO-GO'}")
        print(f"  elevator-cmd-norm authority: {'GO' if pitch_ok else 'NO-GO'}")
        print(f"  pitch-trim-cmd-norm assist:  {'GO' if trim_ok else 'NO-GO / optional'}")
        if alive and thrust_ok and pitch_ok:
            print("\n  → V13 can trim with elevator-cmd-norm + throttle-cmd-norm.")
        elif pitch_ok and not thrust_ok:
            print("\n  → Pitch path OK; fix throttle/engine property map before trim.")
        elif thrust_ok and not pitch_ok:
            print("\n  → Thrust OK; pitch may need surface-pos or different cmd.")
        else:
            print("\n  → Both paths weak. Inspect aircraft XML propulsion + pitch FCS.")
        print(f"\n  Log: {OUT / 'console_log.txt'}")
        print("=" * 70)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
