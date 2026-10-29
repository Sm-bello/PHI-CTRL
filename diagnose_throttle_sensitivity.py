#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL -- Throttle sensitivity test (larger contrast, longer window).

Does NOT modify engine_on()/ownership() -- uses the original, validated
per-step call exactly as in plant/jsbsim_plant_f16.py (the engine_on()
change tested previously made the aircraft unstable and is reverted here).

Tests whether thrust_lbs/final airspeed ever diverges between two throttle
levels given (a) a much larger contrast than 0.15 vs 0.075, and (b) a much
longer window than 60s, to separate "genuinely insensitive in this regime"
from "engine spool lag longer than the episode."

Usage:
  cd PHI_CTRL_RELEASE
  python diagnose_throttle_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import jsbsim
except ImportError as e:
    raise SystemExit("jsbsim required: conda install -c conda-forge jsbsim") from e

from plant.jsbsim_plant_f16 import force_ic, set_throttle, set_elev, ownership, DT

DURATION_S = 180.0          # 3x longer than before
THR_LEVELS = [0.15, 0.075, 0.50, 0.25]  # original pair + a much bigger, higher-band contrast
PRINT_EVERY_N_STEPS = 300   # every ~2.5s


def run_one(label: str, thr: float):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise SystemExit("Failed to load model 'f16'")

    force_ic(fdm)  # calls ownership()/engine_on() internally, once, as designed

    n = int(DURATION_S / DT)
    print(f"\n--- {label} (constant throttle={thr:.3f}) ---")
    for i in range(n):
        set_elev(fdm, -0.02)
        set_throttle(fdm, thr)
        ownership(fdm, 0.0, 0.0)  # original per-step call, unmodified
        fdm.run()

        if i % PRINT_EVERY_N_STEPS == 0:
            t = fdm.get_sim_time()
            n2 = fdm.get_property_value("propulsion/engine/n2")
            thrust = fdm.get_property_value("propulsion/engine/thrust-lbs")
            vc = fdm.get_property_value("velocities/vc-kts")
            print(f"  t={t:6.1f}  thr={thr:.4f}  n2={n2:7.2f}  thrust_lbs={thrust:9.1f}  vc_kts={vc:7.2f}")


def main():
    print("=" * 70)
    print("  THROTTLE SENSITIVITY TEST (constant throttle, no fault step)")
    print("  Compares 4 fixed throttle levels over 180s each: the original")
    print("  0.15/0.075 pair (which showed no divergence at 60s) plus a")
    print("  much larger, higher-band 0.50/0.25 pair.")
    print("=" * 70)

    for thr in THR_LEVELS:
        run_one(f"throttle={thr}", thr)

    print("\n" + "=" * 70)
    print("Compare final thrust_lbs/vc_kts across the 4 blocks above.")
    print("If 0.15 vs 0.075 still don't diverge but 0.50 vs 0.25 clearly")
    print("do, the campaign's trim throttle sits in a low-sensitivity band")
    print("-- the throttle-boundary experiment needs a bigger physical_gamma")
    print("cut (e.g. 0.3 instead of 0.5) or a higher-throttle envelope to")
    print("show a real effect. If NONE of the 4 diverge even at 180s, the")
    print("wiring issue is still real and needs a different diagnostic.")
    print("=" * 70)


if __name__ == "__main__":
    main()
