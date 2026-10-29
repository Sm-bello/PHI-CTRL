#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL -- Find the LOW-end edge of the flat idle/dry-power zone.

cmd=0.15..0.45 (pos-norm 0.3..0.9) all converge to the same n2=99.34,
thrust=16160 -- flat. This sweeps well below 0.15 to find where thrust
actually starts dropping, so a throttle fault can be placed where it has
a real physical effect instead of inside the flat zone.

Auto-logs to a timestamped .txt file next to the script.

Usage:
  cd PHI_CTRL_RELEASE
  python diagnose_throttle_lowend.py
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


DURATION_S = 90.0
THR_LEVELS = [0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.00]
PRINT_EVERY_N_STEPS = 300


def run_one(thr: float):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise SystemExit("Failed to load model 'f16'")
    force_ic(fdm)

    n = int(DURATION_S / DT)
    print(f"\n--- throttle={thr:.3f} (pos-norm target={2*thr:.3f}) ---")
    for i in range(n):
        set_elev(fdm, -0.02)
        set_throttle(fdm, thr)
        ownership(fdm, 0.0, 0.0)
        fdm.run()
        if i % PRINT_EVERY_N_STEPS == 0:
            t = fdm.get_sim_time()
            pos = fdm.get_property_value("fcs/throttle-pos-norm")
            n2 = fdm.get_property_value("propulsion/engine/n2")
            thrust = fdm.get_property_value("propulsion/engine/thrust-lbs")
            vc = fdm.get_property_value("velocities/vc-kts")
            print(f"  t={t:6.1f}  pos={pos:.4f}  n2={n2:7.2f}  thrust_lbs={thrust:9.1f}  vc_kts={vc:7.2f}")
    return {"thr": thr, "final_n2": n2, "final_thrust": thrust, "final_vc": vc}


def main():
    log_path = HERE / f"throttle_lowend_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    tee = Tee(log_path)
    try:
        print("=" * 70)
        print("  THROTTLE LOW-END EDGE SEARCH (below the flat 0.15-0.45 zone)")
        print("=" * 70)
        results = [run_one(thr) for thr in THR_LEVELS]

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'thr':>6}  {'final_n2':>10}  {'final_thrust':>14}  {'final_vc':>10}")
        for r in results:
            print(f"{r['thr']:>6.2f}  {r['final_n2']:>10.2f}  {r['final_thrust']:>14.1f}  {r['final_vc']:>10.2f}")
        print("\nFirst thr (going down from 0.15) where these break from the")
        print("99.34 / 16160 baseline marks the bottom of the flat zone --")
        print("a real fault needs to land below that point.")
        print(f"\nFull log: {log_path}")
        print("=" * 70)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
