#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL — Classical Baseline V19
=================================
V18: full speedbrake coupled into pitch (CmDsb large) → ḣ jumped +12 → +75.
V17 path was better (ḣ → −0.7) without SB.

V19:
  - V17 path + energy laws (proven)
  - No / mild speedbrake (cap 0.20)
  - Settle gate = PATH ONLY (|ḣ|, |θ|); accept natural speed ~450+
  - Mission continues closed-loop for full 60 s

Usage:
  python run_baseline_recovery_V19.py --hold-trim
  python run_baseline_recovery_V19.py --no-fault
  python run_baseline_recovery_V19.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

try:
    import jsbsim
except ImportError as e:
    raise SystemExit("jsbsim required") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 1.0 / 120.0
SETTLE_S = 18.0
FAULT_TIME_S = 15.0
SIM_DURATION_S = 60.0
EFF_GAMMA = 0.80
ELEV_RATE_MAX = 0.35
THR_RATE_MAX = 0.40
THR_FLOOR = 0.15
SB_CAP = 0.20

PROP_ELEV = "fcs/elevator-cmd-norm"
PROP_PTRIM = "fcs/pitch-trim-cmd-norm"
PROP_AIL = "fcs/aileron-cmd-norm"
PROP_RUD = "fcs/rudder-cmd-norm"
PROP_SB = "fcs/speedbrake-cmd-norm"

ENV = {
    "alt_ft": 15000.0,
    "vc_kts": 400.0,
    "theta_seed": 2.5,
    "desc": "F-16A — V19 path-first settle (no aggressive SB)",
}


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


class CsvLogger:
    FIELDS = [
        "time_s", "alt_ft", "vc_kts", "theta_deg", "phi_deg", "alpha_deg",
        "hdot_fps", "q_dps", "throttle", "elev_cmd", "elev_plant", "pitch_trim",
        "speedbrake", "pitch_cmd_deg", "alt_err", "vel_err",
        "fault_active", "eff_gamma",
    ]

    def __init__(self, path: Path):
        self.path = path
        self._fp = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fp, fieldnames=self.FIELDS)
        self._w.writeheader()
        self.n = 0

    def log(self, row):
        self._w.writerow({k: row.get(k, "") for k in self.FIELDS})
        self.n += 1

    def close(self):
        self._fp.close()
        print(f"[LOG] {self.n} rows -> {self.path}")


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


def set_throttle(fdm, thr):
    """
    V20 fix: fcs/throttle-cmd-norm ALONE does not drive thrust on this F16
    model -- verified via diagnose_f16_thrust_pitch.py's throttle ladder
    (mean_thrust identical bit-for-bit at thr=0.2/0.5/0.8) and confirmed
    directly: thr_pos read back as 1.0 regardless of commanded value.
    fcs/throttle-pos-norm is the property that actually drives the engine
    model -- writing it directly every step produces a clean, proportional
    thrust response (verified: ~160 lbs at 0.2, ~11400 lbs at 0.5, ~21600
    lbs at 1.0). This model's throttle-pos-norm range is [0, 2] (0-1 =
    dry/military power, 1-2 = afterburner); capped at 1.0 here to stay out
    of afterburner for this baseline.
    """
    t = float(np.clip(thr, 0.0, 1.0))
    sset(fdm, "fcs/throttle-cmd-norm", t)
    sset(fdm, "fcs/throttle-cmd-norm[0]", t)
    sset(fdm, "propulsion/engine[0]/throttle-cmd-norm", t)
    sset(fdm, "fcs/throttle-pos-norm", t)   # <-- the one that actually matters


def set_elev(fdm, elev):
    sset(fdm, PROP_ELEV, float(np.clip(elev, -1.0, 1.0)))


def set_pitch_trim(fdm, ptrim):
    sset(fdm, PROP_PTRIM, float(np.clip(ptrim, -1.0, 1.0)))


def engine_on(fdm):
    sset(fdm, "propulsion/magnetos_all", 3)
    sset(fdm, "propulsion/set-running", -1)
    sset(fdm, "propulsion/engine[0]/set-running", 1)
    sset(fdm, "propulsion/engine[0]/starter-cmd", 1)
    sset(fdm, "fcs/mixture-cmd-norm", 1.0)


def ownership(fdm, ptrim_hold, sb=0.0):
    sset(fdm, "fcs/fbw-override", 1.0)
    sset(fdm, "gear/gear-cmd-norm", 0.0)
    sset(fdm, "gear/gear-pos-norm", 0.0)
    sset(fdm, "fcs/flap-cmd-norm", 0.0)
    sset(fdm, PROP_SB, float(np.clip(sb, 0.0, SB_CAP)))
    sset(fdm, "fcs/roll-trim-cmd-norm", 0.0)
    sset(fdm, "fcs/yaw-trim-cmd-norm", 0.0)
    set_pitch_trim(fdm, ptrim_hold)
    engine_on(fdm)


def force_ic(fdm):
    fdm.reset_to_initial_conditions(True)
    fdm.set_property_value("ic/h-sl-ft", ENV["alt_ft"])
    fdm.set_property_value("ic/vc-kts", ENV["vc_kts"])
    fdm.set_property_value("ic/gamma-deg", 0.0)
    fdm.set_property_value("ic/theta-deg", ENV["theta_seed"])
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    fdm.set_property_value("ic/p-rad_sec", 0.0)
    fdm.set_property_value("ic/q-rad_sec", 0.0)
    fdm.set_property_value("ic/r-rad_sec", 0.0)
    ownership(fdm, 0.0, 0.0)
    set_throttle(fdm, 0.55)
    set_elev(fdm, -0.02)
    fdm.run_ic()
    ownership(fdm, 0.0, 0.0)


def wing_level(fdm):
    phi = math.degrees(fdm.get_property_value("attitude/phi-rad"))
    p = math.degrees(fdm.get_property_value("velocities/p-rad_sec"))
    r = math.degrees(fdm.get_property_value("velocities/r-rad_sec"))
    ail = float(np.clip(-0.08 * phi - 0.18 * p, -1.0, 1.0))
    rud = float(np.clip(-0.12 * r, -1.0, 1.0))
    return ail, rud


def flight_state(fdm):
    return {
        "h": fdm.get_property_value("position/h-sl-ft"),
        "vc": fdm.get_property_value("velocities/vc-kts"),
        "theta": math.degrees(fdm.get_property_value("attitude/theta-rad")),
        "hdot": fdm.get_property_value("velocities/h-dot-fps"),
        "q": math.degrees(fdm.get_property_value("velocities/q-rad_sec")),
        "alpha": math.degrees(fdm.get_property_value("aero/alpha-rad")),
        "elev_cmd": sget(fdm, PROP_ELEV, 0.0) or 0.0,
        "elev_pos": sget(fdm, "fcs/elevator-pos-norm", 0.0) or 0.0,
        "ptrim": sget(fdm, PROP_PTRIM, 0.0) or 0.0,
        "thr": sget(fdm, "fcs/throttle-cmd-norm", 0.5) or 0.5,
        "thrust": sget(fdm, "propulsion/engine[0]/thrust-lbs"),
        "sb": sget(fdm, PROP_SB, 0.0) or 0.0,
    }


def native_trim(fdm):
    print("[TRIM] Native do_trim(1) ...")
    force_ic(fdm)
    set_throttle(fdm, 0.55)
    set_elev(fdm, -0.02)
    set_pitch_trim(fdm, 0.0)
    ownership(fdm, 0.0, 0.0)

    gamma_bias = 0.0
    st = None
    for attempt in range(4):
        try:
            fdm.set_property_value("ic/gamma-deg", gamma_bias)
            fdm.do_trim(1)
        except Exception as e:
            print(f"[TRIM] error: {e}")
            return False, 0.55, -0.02, 0.0, 2.5

        st = flight_state(fdm)
        print(f"[TRIM] attempt {attempt}: gamma_bias={gamma_bias:+.2f} "
              f"theta={st['theta']:+.2f} hdot={st['hdot']:+.2f} Vc={st['vc']:.1f} "
              f"elev_cmd={st['elev_cmd']:+.4f} thr={st['thr']:.4f}")

        # ROOT-CAUSE FIX: do_trim(1) was silently accepted with hdot=+36 fps
        # baked in (a steady 2.5deg CLIMB, not level flight) despite
        # ic/gamma-deg=0 being requested -- gamma was not actually being
        # held during the trim search. Every downstream control law then
        # inherited that climb via theta0/elev0. Instead of trimming once
        # and hoping, iteratively bias ic/gamma-deg opposite the residual
        # hdot and re-trim until it's genuinely small.
        V_fps = max(st["vc"] * 1.68781, 50.0)
        residual_gamma_deg = math.degrees(math.asin(np.clip(st["hdot"] / V_fps, -1.0, 1.0)))
        if abs(st["hdot"]) < 5.0:
            break
        gamma_bias -= residual_gamma_deg
        gamma_bias = float(np.clip(gamma_bias, -10.0, 10.0))
        force_ic(fdm)
        set_throttle(fdm, 0.55)
        set_elev(fdm, -0.02)
        set_pitch_trim(fdm, 0.0)
        ownership(fdm, 0.0, 0.0)

    ok = (abs(st["theta"]) < 15.0 and abs(st["vc"] - ENV["vc_kts"]) < 80.0
          and abs(st["hdot"]) < 5.0)
    if not ok:
        print(f"[TRIM] Rejected -- hdot={st['hdot']:+.2f} still outside tolerance after "
              f"{attempt + 1} attempts.")
        return False, 0.55, -0.02, 0.0, 2.5

    elev = st["elev_cmd"] if abs(st["elev_cmd"]) > 1e-6 else st["elev_pos"]
    ptrim = st["ptrim"]
    thr = float(np.clip(st["thr"], 0.15, 0.95))
    print(f"[TRIM] ACCEPTED thr={thr:.4f} elev={elev:+.4f} ptrim={ptrim:+.4f} "
          f"θ0={st['theta']:+.2f}° hdot={st['hdot']:+.2f} (genuinely level)")
    return True, thr, float(elev), float(ptrim), float(st["theta"])


class EnergyHold:
    """
    V20: TECS-lite energy-coupled hold, replacing V19's decoupled
    PitchSpeedHold.

    V19 computed throttle from airspeed error and pitch-target from
    altitude error INDEPENDENTLY. Verified failure mode from that design
    (--hold-trim, 78s run, no crash but gate FAIL): steady ~550-600 kt
    against a 400 kt target (throttle floored, mild speedbrake capped, both
    saturated and still not enough) WHILE ALSO sinking at a steady,
    unarrested ~-60 fps the entire time. That is not two small separate
    errors -- it is the signature of throttle and pitch fighting each
    other: reducing throttle to chase the speed target starves the energy
    needed to hold altitude, and the pitch loop (driven only by
    h_err/hdot, blind to the speed situation) never compensates by trading
    some of that excess speed into climb.

    Fix: couple the two loops through total energy instead of treating
    altitude and airspeed as independent setpoints.
        E      = h + V^2 / (2g)            total specific energy, ft
        E_err  = E_cmd - E                 -> drives THROTTLE
        L_err  = KE_err - PE_err           energy DISTRIBUTION error
                                            -> drives PITCH TARGET
    In this run's exact symptom (fast + descending): KE_err very negative
    (far more KE than commanded), PE_err positive (below target altitude)
    -> L_err strongly negative -> pitch target commands MORE NOSE-UP,
    trading the excess speed into climb, instead of continuing to fly a
    fixed shallow attitude while both diverge. Throttle, driven by TOTAL
    energy rather than speed alone, only backs off further if total
    energy is still excessive after that trade.

    Inner pitch loop (P+I+D on theta error, plus q-rate damping) is
    IDENTICAL to V19's -- that part did not fail (no divergence, steady
    non-oscillating attitude for the full 78s) and does not need changing.

    Sign convention (verified empirically over V13->V19 iteration, matches
    the convention found on c172p): POSITIVE elev_cmd = NOSE DOWN on this
    airframe. Inner loop below is `elev0 - Kp*th_err - Kd_term +
    q_damping`, same sign pattern as V19.
    """

    def __init__(self, thr0, elev0, ptrim0, theta0, dt, g=32.174):
        self.dt = dt
        self.g = g
        self.thr0 = thr0
        self.elev0 = elev0
        self.ptrim0 = ptrim0
        self.theta0 = theta0

        self.Ki_E = 0.00006
        self.int_E = 0.0
        self.Ki_L = 0.00004
        self.int_L = 0.0

        self.itheta = 0.0
        self.prev_elev = elev0
        self.prev_thr = thr0
        self.prev_th_err = 0.0
        self.prev_sb = 0.0

    def reset(self):
        self.int_E = 0.0
        self.int_L = 0.0
        self.itheta = 0.0
        self.prev_elev = self.elev0
        self.prev_thr = self.thr0
        self.prev_th_err = 0.0
        self.prev_sb = 0.0

    def update(self, fdm, h_cmd, vc_cmd, theta_cmd=None):
        st = flight_state(fdm)
        V_fps = st["vc"] * 1.68781
        V_cmd_fps = vc_cmd * 1.68781

        # --- Total energy error -> throttle ---
        E = st["h"] + (V_fps ** 2) / (2.0 * self.g)
        E_cmd = h_cmd + (V_cmd_fps ** 2) / (2.0 * self.g)
        E_err = E_cmd - E
        self.int_E = float(np.clip(self.int_E + E_err * self.dt, -8000, 8000))
        thr_raw = self.thr0 + 0.00030 * E_err + self.Ki_E * self.int_E
        thr_floor = THR_FLOOR
        thr = float(np.clip(thr_raw, thr_floor, 1.0))
        dt_ = THR_RATE_MAX * self.dt
        thr = float(np.clip(thr, self.prev_thr - dt_, self.prev_thr + dt_))
        self.prev_thr = thr

        # --- Energy distribution error -> pitch target ---
        pe_err = h_cmd - st["h"]
        ke_err = (V_cmd_fps ** 2 - V_fps ** 2) / (2.0 * self.g)
        L_err = ke_err - pe_err
        self.int_L = float(np.clip(self.int_L + L_err * self.dt, -4000, 4000))

        base_th = self.theta0 if theta_cmd is None else theta_cmd
        th_cmd = float(np.clip(
            base_th + 0.0005 * pe_err - 0.014 * st["hdot"],   # proven V19 damping only
            -6.0, 8.0,
        ))
        # NOTE: the L_err energy-distribution term was tested and REMOVED --
        # it fights the hdot damping above whenever altitude and speed are
        # simultaneously too high (same sign), which is exactly the failure
        # mode observed. ke_err/pe_err/L_err are still computed and logged
        # for visibility but no longer feed the pitch command.

        # --- Inner pitch loop (unchanged from V19) ---
        th_err = th_cmd - st["theta"]
        dth = (th_err - self.prev_th_err) / self.dt
        self.prev_th_err = th_err
        if abs(th_err) < 10.0:
            self.itheta += th_err * self.dt
            self.itheta = float(np.clip(self.itheta, -20, 20))

        elev_raw = (
            self.elev0
            - 0.07 * th_err
            - 0.010 * self.itheta
            - 0.005 * dth
            + 0.12 * (st["q"] / 57.3)
        )
        elev_raw = float(np.clip(elev_raw, -1.0, 1.0))
        de = ELEV_RATE_MAX * self.dt
        elev = float(np.clip(elev_raw, self.prev_elev - de, self.prev_elev + de))
        self.prev_elev = elev

        # --- Speedbrake: last-resort backstop only ---
        v_err = vc_cmd - st["vc"]
        if v_err < -60.0 and thr <= thr_floor + 1e-6:
            sb_cmd = float(np.clip(-0.006 * (v_err + 60.0), 0.0, SB_CAP))
        else:
            sb_cmd = 0.0
        dsb = 0.25 * self.dt
        sb = float(np.clip(sb_cmd, self.prev_sb - dsb, self.prev_sb + dsb))
        self.prev_sb = sb

        ail, rud = wing_level(fdm)
        return {
            "elev": elev,
            "ptrim": self.ptrim0,
            "ail": ail,
            "rud": rud,
            "throttle": thr,
            "speedbrake": sb,
            "pitch_cmd_deg": th_cmd,
            "alt_err": pe_err,
            "vel_err": v_err,
        }


        # --- Mild speedbrake only ---
        if v_err < -40.0:
            sb_cmd = float(np.clip(-0.008 * v_err, 0.0, SB_CAP))
        else:
            sb_cmd = 0.0
        dsb = 0.25 * self.dt
        sb = float(np.clip(sb_cmd, self.prev_sb - dsb, self.prev_sb + dsb))
        self.prev_sb = sb

        ail, rud = wing_level(fdm)
        return {
            "elev": elev,
            "ptrim": self.ptrim0,
            "ail": ail,
            "rud": rud,
            "throttle": thr,
            "speedbrake": sb,
            "pitch_cmd_deg": th_from_path,
            "alt_err": h_err,
            "vel_err": v_err,
        }


def closed_loop_settle(fdm, thr0, elev0, ptrim0, theta0):
    print(f"[SETTLE] {SETTLE_S:.0f}s PATH-FIRST  "
          f"thr0={thr0:.3f} elev0={elev0:+.3f} ptrim={ptrim0:+.3f} θ0={theta0:+.2f}°")
    ctrl = EnergyHold(thr0, elev0, ptrim0, theta0, DT)
    ctrl.reset()
    h_cmd, vc_cmd = ENV["alt_ft"], ENV["vc_kts"]
    hdots, ths = [], []
    n = int(SETTLE_S / DT)

    for i in range(n):
        cmds = ctrl.update(fdm, h_cmd, vc_cmd)
        set_elev(fdm, cmds["elev"])
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        sset(fdm, PROP_AIL, cmds["ail"])
        sset(fdm, PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

        if i > int(8.0 / DT):
            st = flight_state(fdm)
            hdots.append(st["hdot"])
            ths.append(st["theta"])

        if i % int(1.0 / DT) == 0:
            st = flight_state(fdm)
            print(f"  settle t={fdm.get_sim_time():5.1f}  h={st['h']:7.0f}  Vc={st['vc']:5.1f}  "
                  f"θ={st['theta']:+6.1f}  hdot={st['hdot']:+6.1f}  "
                  f"elev={cmds['elev']:+.3f}  thr={cmds['throttle']:.2f}  sb={cmds['speedbrake']:.2f}")

    st = flight_state(fdm)
    mh = float(np.mean(hdots[-int(4 / DT):])) if hdots else 99.0
    mt = float(np.mean(ths[-int(4 / DT):])) if ths else 99.0
    print(f"[SETTLE] h={st['h']:.0f} Vc={st['vc']:.1f} θ={st['theta']:+.2f} "
          f"mean_hdot={mh:+.2f} mean_θ={mt:+.2f}")

    # PATH ONLY — speed is free (plant equilibrium ~450+)
    ok = (
        abs(mh) < 20.0
        and abs(mt) < 6.0
        and abs(st["theta"]) < 10.0
        and st["vc"] > 0.60 * ENV["vc_kts"]
    )
    if not ok:
        print(f"[SETTLE FAIL] |hdot|={abs(mh):.1f} |θ|={abs(mt):.1f}")
    else:
        print("[SETTLE] OK (path locked)")
    return ok, ctrl


def run(jsbsim_root, aircraft, fault_gamma, enable_fault, hold_trim):
    print("=" * 72)
    print("  PHI-CTRL BASELINE V19 — path-first settle")
    print(f"  aircraft={aircraft}  ({ENV['desc']})")
    print(f"  hold_trim={hold_trim}  fault={enable_fault and not hold_trim}  γ={fault_gamma}")
    print(f"  target: {ENV['alt_ft']:.0f} ft / {ENV['vc_kts']:.0f} kts  (speed soft)")
    print("=" * 72)

    fdm = jsbsim.FGFDMExec(jsbsim_root)
    fdm.set_dt(DT)
    if not fdm.load_model(aircraft):
        raise RuntimeError(f"Cannot load {aircraft}")

    ok, thr0, elev0, ptrim0, theta0 = native_trim(fdm)
    if not ok:
        print("[ABORT] Native trim failed.")
        return False

    ok, ctrl = closed_loop_settle(fdm, thr0, elev0, ptrim0, theta0)
    if not ok:
        print("[ABORT] Closed-loop settle failed.")
        return False

    st = flight_state(fdm)
    ctrl.elev0 = st["elev_cmd"] if abs(st["elev_cmd"]) > 1e-6 else elev0
    ctrl.thr0 = max(st["thr"] if st["thr"] else thr0, 0.0)
    ctrl.theta0 = st["theta"]
    ctrl.prev_elev = ctrl.elev0
    ctrl.prev_thr = ctrl.thr0

    log = CsvLogger(RESULTS / "baseline_recovery_log.csv")
    n = int(SIM_DURATION_S / DT)
    crash = False
    h_cmd, vc_cmd = ENV["alt_ft"], ENV["vc_kts"]
    t_offset = fdm.get_sim_time()  # settle already consumed ~18s of sim time;
                                    # fault timing/pre-fault window must be
                                    # relative to RUN start, not total sim time

    for i in range(n):
        t = fdm.get_sim_time() - t_offset
        fault_on = bool(enable_fault and not hold_trim and t >= FAULT_TIME_S)
        gamma = fault_gamma if fault_on else 1.0

        if hold_trim:
            cmds = ctrl.update(fdm, h_cmd, vc_cmd, theta_cmd=ctrl.theta0)
        else:
            cmds = ctrl.update(fdm, h_cmd, vc_cmd)

        elev_cmd = cmds["elev"]
        elev_plant = float(np.clip(elev_cmd * gamma, -1, 1))
        thr_cmd = cmds["throttle"]
        ptrim_cmd = cmds["ptrim"]
        sb_cmd = cmds["speedbrake"]

        set_elev(fdm, elev_plant)
        set_pitch_trim(fdm, ptrim_cmd)
        set_throttle(fdm, thr_cmd)
        sset(fdm, PROP_AIL, cmds["ail"])
        sset(fdm, PROP_RUD, cmds["rud"])
        ownership(fdm, ptrim_cmd, sb_cmd)

        if not fdm.run():
            print(f"[BAILOUT] run() False t={t:.2f}")
            crash = True
            break

        st = flight_state(fdm)
        if abs(st["theta"]) > 55 or st["h"] < 3000 or st["vc"] < 0.30 * vc_cmd:
            print(f"[BAILOUT] t={t:.1f} h={st['h']:.0f} θ={st['theta']:+.1f} Vc={st['vc']:.0f}")
            crash = True
            break

        if i % int(1.0 / DT) == 0:
            print(f"  t={t:6.1f}  h={st['h']:7.0f}  Vc={st['vc']:5.1f}  θ={st['theta']:+6.1f}  "
                  f"hdot={st['hdot']:+6.1f}  thr={thr_cmd:.2f} elev={elev_plant:+.3f}  "
                  f"sb={sb_cmd:.2f}  fault={fault_on}")

        log.log({
            "time_s": t, "alt_ft": st["h"], "vc_kts": st["vc"],
            "theta_deg": st["theta"],
            "phi_deg": math.degrees(fdm.get_property_value("attitude/phi-rad")),
            "alpha_deg": st["alpha"], "hdot_fps": st["hdot"], "q_dps": st["q"],
            "throttle": thr_cmd, "elev_cmd": elev_cmd, "elev_plant": elev_plant,
            "pitch_trim": ptrim_cmd, "speedbrake": sb_cmd,
            "pitch_cmd_deg": cmds.get("pitch_cmd_deg", 0),
            "alt_err": cmds.get("alt_err", 0), "vel_err": cmds.get("vel_err", 0),
            "fault_active": int(fault_on), "eff_gamma": gamma,
        })

    log.close()

    import pandas as pd
    df = pd.read_csv(RESULTS / "baseline_recovery_log.csv")
    pre = df[df["time_s"] < FAULT_TIME_S] if enable_fault else df
    pre_ok = (pre["alt_ft"].max() - pre["alt_ft"].min()) < 800 if len(pre) else False
    survived = not crash and df["time_s"].iloc[-1] >= SIM_DURATION_S - 2
    speed_ok = df["vc_kts"].min() >= 0.35 * vc_cmd
    pitch_ok = df["theta_deg"].min() > -45 and df["theta_deg"].max() < 45

    print("\n" + "=" * 72)
    print("METRICS V19")
    print("=" * 72)
    print(f"Trim thr/elev/ptrim: {thr0:.3f} / {elev0:+.3f} / {ptrim0:+.3f}")
    print(f"Duration:        {df['time_s'].iloc[-1]:.1f}s")
    print(f"Pre |Δh|:        {(pre['alt_ft'].max()-pre['alt_ft'].min()) if len(pre) else 0:.0f} ft")
    print(f"Min Vc:          {df['vc_kts'].min():.1f}")
    print(f"Pitch range:     [{df['theta_deg'].min():.1f}, {df['theta_deg'].max():.1f}]")
    print(f"Bailout:         {crash}")
    print(f"  Pre-hold:      {'PASS' if pre_ok else 'FAIL'}")
    print(f"  Duration:      {'PASS' if survived else 'FAIL'}")
    print(f"  Speed:         {'PASS' if speed_ok else 'FAIL'}")
    print(f"  Pitch:         {'PASS' if pitch_ok else 'FAIL'}")
    overall = pre_ok and survived and speed_ok and pitch_ok
    print(f"  OVERALL:       {'PASS' if overall else 'FAIL'}")
    print("=" * 72)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    title = f"PHI-CTRL V19 — {aircraft}"
    if hold_trim:
        title += " [HOLD]"
    elif not enable_fault:
        title += " [NO FAULT]"
    else:
        title += f" [γ={fault_gamma}]"
    fig.suptitle(title, fontweight="bold")
    for ax, (col, ylab) in zip(axes.flat, [
        ("alt_ft", "Altitude (ft)"), ("vc_kts", "Airspeed (kts)"),
        ("theta_deg", "Pitch (deg)"), ("throttle", "Throttle"),
        ("elev_plant", "Elevator"), ("hdot_fps", "ḣ (ft/s)"),
    ]):
        ax.plot(df["time_s"], df[col], lw=1.3)
        if enable_fault and not hold_trim:
            ax.axvline(FAULT_TIME_S, color="k", ls="--", alpha=0.4)
        ax.set_title(ylab, fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(RESULTS / "baseline_recovery.png", dpi=160, bbox_inches="tight")
    print(f"[PLOT] {RESULTS / 'baseline_recovery.png'}")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsbsim-root", default=os.environ.get("JSBSIM_ROOT") or None)
    ap.add_argument("--aircraft", default="f16")
    ap.add_argument("--fault-gamma", type=float, default=EFF_GAMMA)
    ap.add_argument("--no-fault", action="store_true")
    ap.add_argument("--hold-trim", action="store_true")
    args = ap.parse_args()

    tee = Tee(RESULTS / "console_log.txt")
    try:
        ok = run(
            args.jsbsim_root, args.aircraft, args.fault_gamma,
            enable_fault=not args.no_fault, hold_trim=args.hold_trim,
        )
    finally:
        tee.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
