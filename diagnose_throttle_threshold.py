#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL -- Throttle detent-threshold bisection.

Prior test showed 0.075/0.15/0.25 are bit-for-bit IDENTICAL across 177s
(no sensitivity at all), while 0.50 differs (n2 pinned at 100.0 vs 99.34).
This brackets the transition: 0.30, 0.35, 0.40, 0.45, plus prints BOTH
cmd-norm and pos-norm every step to check whether they diverge from each
other over time (would indicate the native FCS is recomputing pos-norm
from cmd-norm on its own schedule, overriding our direct write).

Auto-logs full console output to a timestamped .txt file in the same
folder, in addition to printing -- no need to redirect/copy manually.

Usage:
  cd PHI_CTRL_RELEASE
  python diagnose_throttle_threshold.py
"""
from __future__ import annotations

import sys
import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import jsbsim
except ImportError as e:
    raise SystemExit("jsbsim required: conda install -c conda-forge jsbsim") from e

from plant.jsbsim_plant_f16 import force_ic, set_throttle, set_elev, ownership, DT


class Tee:
    """Mirrors everything printed to both the console and a log file."""
    def __init__(self, path: Path):
        self.console = sys.stdout
        self.file = open(path, "w", encoding="utf-8")
        sys.stdout = self

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.console
        self.file.close()


DURATION_S = 120.0
THR_LEVELS = [0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
PRINT_EVERY_N_STEPS = 300  # every ~2.5s


def run_one(label: str, thr: float):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise SystemExit("Failed to load model 'f16'")

    force_ic(fdm)

    n = int(DURATION_S / DT)
    print(f"\n--- {label} (constant throttle={thr:.3f}) ---")
    for i in range(n):
        set_elev(fdm, -0.02)
        set_throttle(fdm, thr)
        ownership(fdm, 0.0, 0.0)
        fdm.run()

        if i % PRINT_EVERY_N_STEPS == 0:
            t = fdm.get_sim_time()
            cmd = fdm.get_property_value("fcs/throttle-cmd-norm")
            pos = fdm.get_property_value("fcs/throttle-pos-norm")
            n2 = fdm.get_property_value("propulsion/engine/n2")
            thrust = fdm.get_property_value("propulsion/engine/thrust-lbs")
            vc = fdm.get_property_value("velocities/vc-kts")
            print(f"  t={t:6.1f}  cmd={cmd:.4f}  pos={pos:.4f}  "
                  f"n2={n2:7.2f}  thrust_lbs={thrust:9.1f}  vc_kts={vc:7.2f}")

    return {"thr": thr, "final_thrust": thrust, "final_n2": n2, "final_vc": vc}


def main():
    log_path = HERE / f"throttle_threshold_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    tee = Tee(log_path)
    try:
        print("=" * 70)
        print("  THROTTLE DETENT-THRESHOLD BISECTION")
        print(f"  Bracketing 0.25 (identical to 0.15) vs 0.50 (differs)")
        print("=" * 70)

        results = []
        for thr in THR_LEVELS:
            results.append(run_one(f"throttle={thr}", thr))

        print("\n" + "=" * 70)
        print("SUMMARY -- final values at t=120s, one row per throttle level")
        print("=" * 70)
        print(f"{'thr':>6}  {'final_n2':>10}  {'final_thrust':>14}  {'final_vc':>10}")
        for r in results:
            print(f"{r['thr']:>6.2f}  {r['final_n2']:>10.2f}  {r['final_thrust']:>14.1f}  {r['final_vc']:>10.2f}")
        print("\nLook for the first thr value where final_n2/thrust/vc break away")
        print("from the 0.15/0.25 baseline -- that's the detent threshold.")
        print(f"\nFull log written to: {log_path}")
        print("=" * 70)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
