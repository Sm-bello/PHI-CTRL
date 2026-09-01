#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-trim probe — do NOT overwrite pitch after do_trim.
Only hold throttle + wing-level. Log elev/ptrim/alpha/θ/hdot every 0.5 s.
Shows whether the trimmed state holds when we leave the pitch FCS alone.
"""

from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path
import numpy as np

try:
    import jsbsim
except ImportError as e:
    raise SystemExit("jsbsim required") from e

DT = 1.0 / 120.0
ALT, VC = 15000.0, 400.0
OUT = Path(__file__).resolve().parent / "results_probe"
OUT.mkdir(parents=True, exist_ok=True)


def sset(fdm, p, v):
    try:
        fdm.set_property_value(p, float(v))
        return True
    except Exception:
        return False


def sget(fdm, p, d=None):
    try:
        return fdm.get_property_value(p)
    except Exception:
        return d


def engine_on(fdm):
    sset(fdm, "propulsion/magnetos_all", 3)
    sset(fdm, "propulsion/set-running", -1)
    sset(fdm, "propulsion/engine[0]/set-running", 1)
    sset(fdm, "propulsion/engine[0]/starter-cmd", 1)
    sset(fdm, "fcs/mixture-cmd-norm", 1.0)


def set_thr(fdm, t):
    t = float(np.clip(t, 0, 1))
    sset(fdm, "fcs/throttle-cmd-norm", t)
    sset(fdm, "fcs/throttle-cmd-norm[0]", t)
    sset(fdm, "propulsion/engine[0]/throttle-cmd-norm", t)


def force_ic(fdm):
    fdm.reset_to_initial_conditions(True)
    fdm.set_property_value("ic/h-sl-ft", ALT)
    fdm.set_property_value("ic/vc-kts", VC)
    fdm.set_property_value("ic/gamma-deg", 0.0)
    fdm.set_property_value("ic/theta-deg", 2.5)
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    fdm.set_property_value("ic/p-rad_sec", 0.0)
    fdm.set_property_value("ic/q-rad_sec", 0.0)
    fdm.set_property_value("ic/r-rad_sec", 0.0)
    sset(fdm, "fcs/fbw-override", 1.0)
    sset(fdm, "gear/gear-cmd-norm", 0.0)
    sset(fdm, "gear/gear-pos-norm", 0.0)
    sset(fdm, "fcs/flap-cmd-norm", 0.0)
    sset(fdm, "fcs/speedbrake-cmd-norm", 0.0)
    sset(fdm, "fcs/aileron-cmd-norm", 0.0)
    sset(fdm, "fcs/rudder-cmd-norm", 0.0)
    sset(fdm, "fcs/elevator-cmd-norm", -0.02)
    sset(fdm, "fcs/pitch-trim-cmd-norm", 0.0)
    engine_on(fdm)
    set_thr(fdm, 0.55)
    fdm.run_ic()
    engine_on(fdm)


def snap(fdm):
    return {
        "t": fdm.get_sim_time(),
        "h": fdm.get_property_value("position/h-sl-ft"),
        "vc": fdm.get_property_value("velocities/vc-kts"),
        "theta": math.degrees(fdm.get_property_value("attitude/theta-rad")),
        "hdot": fdm.get_property_value("velocities/h-dot-fps"),
        "q": math.degrees(fdm.get_property_value("velocities/q-rad_sec")),
        "alpha": math.degrees(fdm.get_property_value("aero/alpha-rad")),
        "elev_cmd": sget(fdm, "fcs/elevator-cmd-norm"),
        "elev_pos": sget(fdm, "fcs/elevator-pos-norm"),
        "ptrim": sget(fdm, "fcs/pitch-trim-cmd-norm"),
        "thr": sget(fdm, "fcs/throttle-cmd-norm"),
        "thrust": sget(fdm, "propulsion/engine[0]/thrust-lbs"),
        "elev_sch": sget(fdm, "fcs/elevator-scheduler"),
        "elev_lim": sget(fdm, "fcs/elevator-cmd-limiter"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsbsim-root", default=os.environ.get("JSBSIM_ROOT") or None)
    ap.add_argument("--aircraft", default="f16")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    log_path = OUT / "console_log.txt"
    sys_stdout = sys.stdout
    f_log = open(log_path, "w", encoding="utf-8")

    class Tee:
        def write(self, m):
            sys_stdout.write(m)
            f_log.write(m)
            f_log.flush()
        def flush(self):
            sys_stdout.flush()
            f_log.flush()

    sys.stdout = Tee()
    try:
        print("=" * 70)
        print("  POST-TRIM PROBE — pitch FCS left untouched after do_trim")
        print("=" * 70)

        fdm = jsbsim.FGFDMExec(args.jsbsim_root)
        fdm.set_dt(DT)
        if not fdm.load_model(args.aircraft):
            raise SystemExit("load failed")

        force_ic(fdm)
        print("[TRIM] do_trim(1) ...")
        fdm.do_trim(1)
        s0 = snap(fdm)
        print(f"[TRIM] θ={s0['theta']:+.2f} hdot={s0['hdot']:+.2f} Vc={s0['vc']:.1f} "
              f"elev_cmd={s0['elev_cmd']} elev_pos={s0['elev_pos']} ptrim={s0['ptrim']} "
              f"thr={s0['thr']} thrust={s0['thrust']}")
        print(f"       elev_lim={s0['elev_lim']} elev_sch={s0['elev_sch']}")

        thr0 = s0["thr"] if s0["thr"] is not None else 0.35
        print(f"\n[PROBE] {args.seconds:.0f}s — only throttle + wing-level; NO elev/ptrim writes")
        print(f"{'t':>6} {'h':>8} {'Vc':>7} {'θ':>7} {'hdot':>8} {'q':>7} {'α':>6} "
              f"{'e_cmd':>7} {'e_pos':>7} {'ptrim':>7} {'lim':>7}")

        n = int(args.seconds / DT)
        for i in range(n):
            # throttle only + lateral; leave pitch properties alone
            set_thr(fdm, thr0)
            phi = math.degrees(fdm.get_property_value("attitude/phi-rad"))
            p = math.degrees(fdm.get_property_value("velocities/p-rad_sec"))
            r = math.degrees(fdm.get_property_value("velocities/r-rad_sec"))
            sset(fdm, "fcs/aileron-cmd-norm", float(np.clip(-0.08 * phi - 0.18 * p, -1, 1)))
            sset(fdm, "fcs/rudder-cmd-norm", float(np.clip(-0.12 * r, -1, 1)))
            sset(fdm, "fcs/fbw-override", 1.0)
            sset(fdm, "gear/gear-cmd-norm", 0.0)
            engine_on(fdm)
            fdm.run()

            if i % int(0.5 / DT) == 0:
                s = snap(fdm)
                print(f"{s['t']:6.1f} {s['h']:8.0f} {s['vc']:7.1f} {s['theta']:+7.2f} "
                      f"{s['hdot']:+8.1f} {s['q']:+7.2f} {s['alpha']:+6.2f} "
                      f"{str(s['elev_cmd']):>7} {str(s['elev_pos']):>7} "
                      f"{str(s['ptrim']):>7} {str(s['elev_lim']):>7}")

        s = snap(fdm)
        print(f"\n[END] θ={s['theta']:+.2f} hdot={s['hdot']:+.1f} Vc={s['vc']:.1f} h={s['h']:.0f}")
        held = abs(s["theta"]) < 15 and abs(s["hdot"]) < 40 and abs(s["vc"] - VC) < 80
        print(f"[VERDICT] {'HOLD — pitch FCS self-consistent' if held else 'DIVERGED — need different hold strategy'}")
        print(f"Log: {log_path}")
    finally:
        sys.stdout = sys_stdout
        f_log.close()


if __name__ == "__main__":
    main()
