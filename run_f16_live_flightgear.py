#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL F-16 -- live FlightGear visualization (UDP port 5606).

Physics = JSBSim. FlightGear only displays (external FDM).

---------------------------------------------------------------------------
LAUNCH (two terminals)
---------------------------------------------------------------------------
1) FlightGear FIRST — use the aircraft id from your hangar install.
   Your download path showed: .../Aircraft/f16/f16-block-52-set.xml
   so try:  f16-block-52
   Confirm with:  fgfs --show-aircraft

   fgfs --aircraft=f16-block-52 --fdm=external --native-fdm=socket,in,60,,5606,udp --lat=0 --lon=0 --altitude=15000 --vc=400 --disable-clouds3d --disable-random-objects

2) Then Python:

   python run_f16_live_flightgear.py
   python run_f16_live_flightgear.py --fault
   python run_f16_live_flightgear.py --case TECS_MRAC --fault
   python run_f16_live_flightgear.py --port 5606

---------------------------------------------------------------------------
PROTOCOL / XML
---------------------------------------------------------------------------
Native FDM does NOT use FlightGear's Protocol/ folder.
JSBSim writes flightgear_output.xml next to this script and attaches it
after trim. That file is the only XML you need (auto-generated each run).

Do not put a generic protocol in Protocol/ for this path — it will not
decode native FDM packets.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jsbsim
from plant.jsbsim_plant_f16 import (
    DT, ENV, PROP_AIL, PROP_RUD, native_trim, ownership, set_throttle,
    set_elev, set_pitch_trim, flight_state,
)
from controller.energy_hold_f16 import EnergyHold
from controller.mrac.adaptive_controller import MRACAdaptiveController

# Default port changed from 5550 (often already bound) to 5606
FG_PORT = 5606
FAULT_START_TIME = 15.0
EFF_GAMMA_FAULT = 0.50
SETTLE_S = 18.0


def write_fg_output_config(path: Path, port: int = FG_PORT, rate: int = 60) -> None:
    """JSBSim → FlightGear native FDM output directive (not a FG Protocol file)."""
    path.write_text(
        '<?xml version="1.0"?>\n'
        f'<output name="localhost" type="FLIGHTGEAR" port="{port}" '
        f'protocol="UDP" rate="{rate}"/>\n',
        encoding="utf-8",
    )
    print(f"[FG] Wrote JSBSim output directive: {path}  (UDP port {port})")


def main():
    parser = argparse.ArgumentParser(description="PHI-CTRL F-16 → FlightGear (native FDM)")
    parser.add_argument("--case", default="BASELINE", choices=["BASELINE", "TECS_MRAC"])
    parser.add_argument("--fault", action="store_true", help="50%% elev loss at t=15s")
    parser.add_argument("--port", type=int, default=FG_PORT, help=f"UDP port (default {FG_PORT})")
    parser.add_argument("--duration", type=float, default=120.0)
    args = parser.parse_args()

    fg_config_path = HERE / "flightgear_output.xml"
    write_fg_output_config(fg_config_path, port=args.port)

    print(f"[FG] Start FlightGear BEFORE this script if you have not:")
    print(
        f"  fgfs --aircraft=f16-block-52 --fdm=external "
        f"--native-fdm=socket,in,60,,{args.port},udp "
        f"--lat=0 --lon=0 --altitude=15000 --vc=400"
    )
    print(f"[FG] (If aircraft id differs, run: fgfs --show-aircraft)")

    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        raise RuntimeError("Failed to load JSBSim model 'f16'")

    ok, trim_thr, trim_elev, trim_ptrim, trim_theta = native_trim(fdm)
    if not ok:
        print("[TRIM] FAILED -- aborting.")
        return

    # Attach socket AFTER trim so FG is not flooded with trim sub-steps
    fdm.set_output_directive(str(fg_config_path))

    print(f"[SETTLE] {SETTLE_S:.0f}s closed-loop settle (wall-clock)...")
    baseline = EnergyHold(trim_thr, trim_elev, trim_ptrim, trim_theta, DT)
    t_wall_start = time.time()
    for i in range(int(SETTLE_S / DT)):
        cmds = baseline.update(fdm, ENV["alt_ft"], ENV["vc_kts"])
        set_elev(fdm, cmds["elev"])
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()
        sleep_s = (t_wall_start + (i + 1) * DT) - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

    st0 = flight_state(fdm)
    baseline.elev0 = st0["elev_cmd"] if abs(st0["elev_cmd"]) > 1e-6 else trim_elev
    baseline.thr0 = max(st0["thr"] if st0["thr"] else trim_thr, 0.0)
    baseline.theta0 = st0["theta"]
    baseline.prev_elev = baseline.elev0
    baseline.prev_thr = baseline.thr0
    print(
        f"[SETTLE] h={st0['h']:.0f}ft Vc={st0['vc']:.1f}kts "
        f"theta={st0['theta']:+.2f}deg -- streaming UDP :{args.port}"
    )

    mrac = MRACAdaptiveController(dt=DT) if args.case == "TECS_MRAC" else None
    mrac_prev_u = 0.0

    t_offset = fdm.get_sim_time()
    t_wall_start = time.time()
    n = int(args.duration / DT)
    for i in range(n):
        t = fdm.get_sim_time() - t_offset
        fault_active = args.fault and t >= FAULT_START_TIME
        physical_gamma = EFF_GAMMA_FAULT if fault_active else 1.0

        cmds = baseline.update(fdm, ENV["alt_ft"], ENV["vc_kts"])
        elev_raw = cmds["elev"]

        mrac_elev = 0.0
        if mrac is not None:
            st = flight_state(fdm)
            target_theta_rad = math.radians(cmds["pitch_cmd_deg"])
            mrac_state = [
                st["vc"] * 1.68781, 0.0,
                math.radians(st["q"]), math.radians(st["theta"]), st["h"],
            ]
            try:
                u_out = mrac.compute_action(mrac_state, target_theta=target_theta_rad)
                u_out = float(u_out[0]) if hasattr(u_out, "__len__") else float(u_out)
            except Exception:
                u_out = 0.0
            u_out = max(-0.25, min(0.25, u_out))
            du = max(-0.08, min(0.08, u_out - mrac_prev_u))
            mrac_elev = -(mrac_prev_u + du)
            mrac_prev_u = mrac_prev_u + du

        elev_comp = max(-1.0, min(1.0, elev_raw + mrac_elev))
        elev_plant = elev_comp * physical_gamma

        set_elev(fdm, elev_plant)
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

        if i % int(5.0 / DT) == 0:
            st = flight_state(fdm)
            tag = " [FAULT ACTIVE]" if fault_active else ""
            print(
                f"  t={t:6.1f}s  h={st['h']:7.0f}ft  Vc={st['vc']:5.1f}kts  "
                f"theta={st['theta']:+.2f}deg{tag}"
            )

        sleep_s = (t_wall_start + (i + 1) * DT) - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

    print("[DONE]")


if __name__ == "__main__":
    main()
