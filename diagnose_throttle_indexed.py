#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL -- Re-run the low-end sweep reading INDEXED engine properties.

Every prior diagnostic read 'propulsion/engine/n2' and
'propulsion/engine/thrust-lbs' (unindexed) and found them frozen across
the entire 0.00-0.45 throttle range. diagnose_f16_fcs.py's own readback
list uses 'propulsion/engine[0]/n1' and 'propulsion/engine[0]/thrust-lbs'
-- indexed. This checks whether the indexed properties actually respond
to throttle where the unindexed ones didn't.

Auto-logs to a timestamped .txt file next to the script.

Usage:
  cd PHI_CTRL_RELEASE
  python diagnose_throttle_indexed.py
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
THR_LEVELS = [0.00, 0.15, 0.30, 0.45, 0.50]
PRINT_EVERY_N_STEPS = 300

PROPS_TO_TRY = [
    "propulsion/engine/n2",
    "propulsion/engine[0]/n2",
    "propulsion/engine/n1",
    "propulsion/engine[0]/n1",
    "propulsion/engine/thrust-lbs",
    "propulsion/engine[0]/thrust-lbs",
]


def safe_get(fdm, prop):
    try:
        return fdm.get_property_value(prop)
    except Exception:
        return None


def run_one(thr: float):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise SystemExit("Failed to load model 'f16'")
    force_ic(fdm)

    n = int(DURATION_S / DT)
    print(f"\n--- throttle={thr:.3f} ---")
    last = {}
    for i in range(n):
        set_elev(fdm, -0.02)
        set_throttle(fdm, thr)
        ownership(fdm, 0.0, 0.0)
        fdm.run()
        if i % PRINT_EVERY_N_STEPS == 0:
            t = fdm.get_sim_time()
            vals = {p: safe_get(fdm, p) for p in PROPS_TO_TRY}
            last = vals
            row = "  ".join(
                f"{p.split('/')[-1]}[{'idx' if '[' in p else 'noidx'}]={v:.2f}"
                if v is not None else f"{p}=MISSING"
                for p, v in vals.items()
            )
            print(f"  t={t:6.1f}  {row}")
    return {"thr": thr, **last}


def main():
    log_path = HERE / f"throttle_indexed_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
    tee = Tee(log_path)
    try:
        print("=" * 70)
        print("  INDEXED vs UNINDEXED ENGINE PROPERTY CHECK")
        print("  First run at each throttle level also prints which of the")
        print("  6 candidate properties EXIST at all (missing = None).")
        print("=" * 70)

        # existence probe once, up front
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_dt(DT)
        fdm.load_model("f16")
        force_ic(fdm)
        print("\nPROPERTY EXISTENCE:")
        for p in PROPS_TO_TRY:
            v = safe_get(fdm, p)
            print(f"  {p:40s}  {'FOUND  value=' + str(v) if v is not None else 'MISSING'}")

        results = [run_one(thr) for thr in THR_LEVELS]

        print("\n" + "=" * 70)
        print("SUMMARY -- final values, one row per throttle level")
        print("=" * 70)
        header = "  thr  " + "  ".join(p.split("/")[-1] + ("[idx]" if "[" in p else "") for p in PROPS_TO_TRY)
        print(header)
        for r in results:
            row = f"{r['thr']:>5.2f}  " + "  ".join(
                f"{r.get(p):>10.2f}" if r.get(p) is not None else f"{'--':>10}"
                for p in PROPS_TO_TRY
            )
            print(row)
        print(f"\nFull log: {log_path}")
        print("=" * 70)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
