#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL — F-16 FCS Control-Path Diagnostic
===========================================
Maps which properties actually move the aircraft in pitch / throttle.
Run once, send console_log + any printed tables. No closed-loop, no fault.

Usage:
  export JSBSIM_ROOT=/path/to/JSBSim   # if needed
  python diagnose_f16_fcs.py
  python diagnose_f16_fcs.py --jsbsim-root /path/to/JSBSim
  python diagnose_f16_fcs.py --aircraft f16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_diag"
OUT.mkdir(parents=True, exist_ok=True)

try:
    import jsbsim
except ImportError as e:
    raise SystemExit("jsbsim required: conda install -c conda-forge jsbsim") from e

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


def safe_set(fdm, prop, val):
    try:
        fdm.set_property_value(prop, float(val))
        return True
    except Exception:
        return False


def safe_get(fdm, prop, default=None):
    try:
        return fdm.get_property_value(prop)
    except Exception:
        return default


def apply_clean(fdm, fbw=1.0):
    safe_set(fdm, "fcs/fbw-override", fbw)
    safe_set(fdm, "gear/gear-cmd-norm", 0.0)
    safe_set(fdm, "gear/gear-pos-norm", 0.0)
    safe_set(fdm, "fcs/flap-cmd-norm", 0.0)
    safe_set(fdm, "fcs/speedbrake-cmd-norm", 0.0)
    safe_set(fdm, "fcs/mixture-cmd-norm", 1.0)
    safe_set(fdm, "propulsion/magnetos_all", 3)
    safe_set(fdm, "propulsion/set-running", -1)
    safe_set(fdm, "propulsion/engine[0]/set-running", 1)
    safe_set(fdm, "fcs/aileron-cmd-norm", 0.0)
    safe_set(fdm, "fcs/rudder-cmd-norm", 0.0)
    safe_set(fdm, "fcs/roll-trim-cmd-norm", 0.0)
    safe_set(fdm, "fcs/yaw-trim-cmd-norm", 0.0)


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
    apply_clean(fdm, fbw=1.0)
    fdm.run_ic()
    apply_clean(fdm, fbw=1.0)


def state(fdm):
    return {
        "h": fdm.get_property_value("position/h-sl-ft"),
        "vc": fdm.get_property_value("velocities/vc-kts"),
        "theta": math.degrees(fdm.get_property_value("attitude/theta-rad")),
        "hdot": fdm.get_property_value("velocities/h-dot-fps"),
        "q": math.degrees(fdm.get_property_value("velocities/q-rad_sec")),
        "alpha": math.degrees(fdm.get_property_value("aero/alpha-rad")),
    }


def run_hold(fdm, seconds, thr, write_fn):
    """Run for `seconds` applying write_fn every step; return final state + mean q/hdot."""
    n = int(seconds / DT)
    qs, hds = [], []
    for i in range(n):
        safe_set(fdm, "fcs/throttle-cmd-norm", thr)
        write_fn(fdm)
        apply_clean(fdm, fbw=1.0)
        fdm.run()
        if i > int(1.0 / DT):
            qs.append(math.degrees(fdm.get_property_value("velocities/q-rad_sec")))
            hds.append(fdm.get_property_value("velocities/h-dot-fps"))
    st = state(fdm)
    st["mean_q"] = float(np.mean(qs)) if qs else 0.0
    st["mean_hdot"] = float(np.mean(hds)) if hds else 0.0
    return st


# ---------------------------------------------------------------------------
# Candidate pitch properties to test
# ---------------------------------------------------------------------------
PITCH_CANDIDATES = [
    "fcs/elevator-cmd-norm",
    "fcs/pitch-trim-cmd-norm",
    "fcs/elevator-pos-norm",
    "fcs/elevator-pos-rad",
    "fcs/pitch-cmd-norm",
    "fcs/elevator-cmd-sum",
    "fcs/elevator-command",
    "fcs/de-cmd",
    "fcs/pitch-stick",
    "fcs/stick-pitch",
    "fcs/Nz-cmd",
    "fcs/nz-cmd",
    "fcs/pitch-rate-cmd",
    "fcs/q-cmd",
    "ap/elevator_cmd",
    "ap/pitch_hold",
]

THR_CANDIDATES = [
    "fcs/throttle-cmd-norm",
    "fcs/throttle-cmd-norm[0]",
    "fcs/throttle-pos-norm",
    "fcs/throttle-pos-norm[0]",
    "propulsion/engine[0]/throttle-cmd-norm",
    "propulsion/engine[0]/set-running",
]

READBACK_PROPS = [
    "fcs/elevator-cmd-norm",
    "fcs/elevator-pos-norm",
    "fcs/elevator-pos-rad",
    "fcs/pitch-trim-cmd-norm",
    "fcs/elevator-scheduler",
    "fcs/elevator-cmd-limiter",
    "fcs/fbw-override",
    "fcs/throttle-cmd-norm",
    "fcs/throttle-pos-norm",
    "propulsion/engine[0]/thrust-lbs",
    "propulsion/engine[0]/n1",
    "velocities/vc-kts",
    "attitude/theta-rad",
    "velocities/q-rad_sec",
    "aero/alpha-rad",
    "position/h-sl-ft",
]


def probe_exists(fdm, props):
    print("\n" + "=" * 70)
    print("1) PROPERTY EXISTENCE PROBE")
    print("=" * 70)
    found, missing = [], []
    for p in props:
        v = safe_get(fdm, p)
        if v is not None:
            found.append((p, v))
            print(f"  FOUND   {p:45s}  = {v}")
        else:
            missing.append(p)
    print(f"\n  Found {len(found)} / {len(props)}")
    if missing:
        print("  Missing (not exposed or wrong name):")
        for p in missing:
            print(f"    - {p}")
    return found


def test_pitch_authority(fdm):
    """
    For each candidate: hold baseline, then step the property +0.15 and -0.15
    for 3 s each, measure Δθ and mean q. Strong response = real control path.
    """
    print("\n" + "=" * 70)
    print("2) PITCH AUTHORITY SWEEP (step ±0.15 for 3 s, thr fixed 0.55)")
    print("=" * 70)
    print(f"{'property':40s}  {'Δθ+':>8s}  {'mean_q+':>8s}  {'Δθ-':>8s}  {'mean_q-':>8s}  verdict")
    print("-" * 90)

    results = []
    thr = 0.55

    for prop in PITCH_CANDIDATES:
        # Does it exist?
        if safe_get(fdm, prop) is None:
            print(f"{prop:40s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  {'—':>8s}  NOT FOUND")
            continue

        # Baseline 2 s at 0
        force_ic(fdm)
        safe_set(fdm, prop, 0.0)
        st0 = run_hold(fdm, 2.0, thr, lambda f, p=prop: safe_set(f, p, 0.0))
        th0 = st0["theta"]

        # + step
        force_ic(fdm)
        st_p = run_hold(fdm, 3.0, thr, lambda f, p=prop: safe_set(f, p, 0.15))
        dth_p = st_p["theta"] - th0
        mq_p = st_p["mean_q"]

        # - step
        force_ic(fdm)
        st_m = run_hold(fdm, 3.0, thr, lambda f, p=prop: safe_set(f, p, -0.15))
        dth_m = st_m["theta"] - th0
        mq_m = st_m["mean_q"]

        # Verdict: significant pitch rate or angle change
        score = abs(mq_p) + abs(mq_m) + 0.3 * (abs(dth_p) + abs(dth_m))
        if score > 3.0:
            verd = "STRONG"
        elif score > 0.8:
            verd = "weak"
        else:
            verd = "none"
        print(f"{prop:40s}  {dth_p:+8.2f}  {mq_p:+8.2f}  {dth_m:+8.2f}  {mq_m:+8.2f}  {verd}")
        results.append((prop, dth_p, mq_p, dth_m, mq_m, verd, score))

    results.sort(key=lambda x: -x[6])
    print("\n  Ranked by authority score:")
    for r in results[:8]:
        print(f"    {r[0]:40s}  score={r[6]:.2f}  ({r[5]})")
    return results


def test_fbw_effect(fdm):
    """Compare elevator-cmd-norm response with fbw-override 0 vs 1."""
    print("\n" + "=" * 70)
    print("3) FBW-OVERRIDE EFFECT ON elevator-cmd-norm")
    print("=" * 70)
    thr = 0.55
    prop = "fcs/elevator-cmd-norm"

    for fbw in (0.0, 1.0):
        force_ic(fdm)
        safe_set(fdm, "fcs/fbw-override", fbw)

        def writer(f, p=prop, v=0.15):
            safe_set(f, "fcs/fbw-override", fbw)
            safe_set(f, p, v)

        st = run_hold(fdm, 3.0, thr, writer)
        print(f"  fbw-override={fbw:.0f}  →  θ={st['theta']:+.2f}°  mean_q={st['mean_q']:+.2f}  "
              f"hdot={st['mean_hdot']:+.1f}  Vc={st['vc']:.1f}")


def test_throttle_authority(fdm):
    print("\n" + "=" * 70)
    print("4) THROTTLE AUTHORITY (hold elev=0, step thr 0.3 → 0.8 for 4 s)")
    print("=" * 70)
    for prop in THR_CANDIDATES:
        if safe_get(fdm, prop) is None and "[" not in prop:
            # try anyway for indexed
            pass
        force_ic(fdm)
        # freeze elev
        def writer_lo(f, p=prop):
            safe_set(f, "fcs/elevator-cmd-norm", 0.0)
            safe_set(f, p, 0.30)

        def writer_hi(f, p=prop):
            safe_set(f, "fcs/elevator-cmd-norm", 0.0)
            safe_set(f, p, 0.80)

        force_ic(fdm)
        st_lo = run_hold(fdm, 4.0, 0.30, writer_lo)
        force_ic(fdm)
        st_hi = run_hold(fdm, 4.0, 0.80, writer_hi)
        dvc = st_hi["vc"] - st_lo["vc"]
        print(f"  {prop:45s}  Vc@0.3={st_lo['vc']:6.1f}  Vc@0.8={st_hi['vc']:6.1f}  ΔVc={dvc:+.1f}")


def try_native_trim_snapshot(fdm):
    print("\n" + "=" * 70)
    print("5) NATIVE do_trim SNAPSHOT (modes 1 then 0)")
    print("=" * 70)
    for mode in (1, 0):
        force_ic(fdm)
        safe_set(fdm, "fcs/throttle-cmd-norm", 0.55)
        safe_set(fdm, "fcs/elevator-cmd-norm", -0.05)
        apply_clean(fdm, fbw=1.0)
        try:
            fdm.do_trim(int(mode))
            st = state(fdm)
            elev_cmd = safe_get(fdm, "fcs/elevator-cmd-norm")
            elev_pos = safe_get(fdm, "fcs/elevator-pos-norm")
            thr_cmd = safe_get(fdm, "fcs/throttle-cmd-norm")
            thr_pos = safe_get(fdm, "fcs/throttle-pos-norm")
            print(f"  do_trim({mode}): θ={st['theta']:+.2f}° hdot={st['hdot']:+.2f} Vc={st['vc']:.1f} α={st['alpha']:+.2f}")
            print(f"           elev_cmd={elev_cmd}  elev_pos={elev_pos}")
            print(f"           thr_cmd={thr_cmd}  thr_pos={thr_pos}")
        except Exception as e:
            print(f"  do_trim({mode}) raised: {e}")


def dump_readbacks(fdm):
    print("\n" + "=" * 70)
    print("6) READBACK DUMP (after IC + 1 s hold elev=0 thr=0.55)")
    print("=" * 70)
    force_ic(fdm)
    for _ in range(int(1.0 / DT)):
        safe_set(fdm, "fcs/elevator-cmd-norm", 0.0)
        safe_set(fdm, "fcs/throttle-cmd-norm", 0.55)
        apply_clean(fdm, fbw=1.0)
        fdm.run()
    for p in READBACK_PROPS:
        v = safe_get(fdm, p)
        if v is not None:
            print(f"  {p:45s}  = {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsbsim-root", default=os.environ.get("JSBSIM_ROOT") or None)
    ap.add_argument("--aircraft", default="f16")
    args = ap.parse_args()

    tee = Tee(OUT / "console_log.txt")
    try:
        print("=" * 70)
        print("  PHI-CTRL F-16 FCS DIAGNOSTIC")
        print(f"  aircraft={args.aircraft}  target IC {ALT_FT:.0f} ft / {VC_KTS:.0f} kts")
        print("=" * 70)

        fdm = jsbsim.FGFDMExec(args.jsbsim_root)
        fdm.set_dt(DT)
        if not fdm.load_model(args.aircraft):
            raise SystemExit(f"Failed to load model '{args.aircraft}'")

        force_ic(fdm)
        dump_readbacks(fdm)
        probe_exists(fdm, PITCH_CANDIDATES + THR_CANDIDATES + READBACK_PROPS)
        try_native_trim_snapshot(fdm)
        pitch_results = test_pitch_authority(fdm)
        test_fbw_effect(fdm)
        test_throttle_authority(fdm)

        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)
        strong = [r for r in pitch_results if r[5] == "STRONG"]
        weak = [r for r in pitch_results if r[5] == "weak"]
        if strong:
            print("  Strong pitch authority found on:")
            for r in strong:
                print(f"    → {r[0]}")
            print("  V13 should command the strongest of these.")
        elif weak:
            print("  Only weak pitch response on:")
            for r in weak:
                print(f"    → {r[0]}")
            print("  Model may need surface-pos override or different interface.")
        else:
            print("  NO pitch authority detected on candidate list.")
            print("  Next: inspect aircraft XML FCS pitch channel for true input name.")
        print(f"\n  Full log: {OUT / 'console_log.txt'}")
        print("=" * 70)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
