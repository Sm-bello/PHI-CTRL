# -*- coding: utf-8 -*-
"""
PHI-CTRL Layer 0 -- F-16 JSBSim plant interface.

Extracted verbatim (not rewritten) from the validated, gate-PASSING V20
run_baseline_recovery.py so the proven fixes travel with it exactly:
  - set_throttle() writes fcs/throttle-pos-norm directly -- throttle-cmd-norm
    ALONE was verified to leave thrust pinned regardless of commanded value
    on this F16 model (diagnose_f16_thrust_pitch.py's throttle ladder showed
    identical thrust at 0.2/0.5/0.8 throttle before this fix).
  - native_trim() runs an iterative gamma-bias correction loop -- do_trim(1)
    was verified to silently accept a trim with hdot=+36 fps baked in (a
    real 2.5deg steady CLIMB, not level flight) despite ic/gamma-deg=0 being
    requested. This loop re-trims with a corrected gamma bias until hdot is
    genuinely small, instead of trusting do_trim's first answer.
  - ownership() sets fcs/fbw-override=1.0, which is what gives our own
    control laws direct authority instead of fighting the F16's native FCS.

Sign convention (load-bearing, do not change without re-verifying against
the aircraft's own Cmde table): POSITIVE elev_cmd = NOSE DOWN on this
airframe -- same convention independently found on c172p.
"""
import math
import numpy as np

DT = 1.0 / 120.0
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
    "desc": "F-16A -- V20 path-first settle, genuinely-level trim",
}


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
    fcs/throttle-cmd-norm ALONE does not drive thrust on this F16 model --
    verified via the throttle ladder diagnostic (thrust identical bit-for-bit
    at thr=0.2/0.5/0.8). fcs/throttle-pos-norm is what actually drives the
    engine model; writing it directly every step produces a clean,
    proportional thrust response. This model's throttle-pos-norm range is
    [0, 2] (0-1 = dry/military power, 1-2 = afterburner); capped at 1.0 here
    to stay out of afterburner for the baseline.
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
    """fbw-override=1.0 gives OUR control laws direct authority instead of
    fighting the F16's native fly-by-wire FCS."""
    sset(fdm, "fcs/fbw-override", 1.0)
    sset(fdm, "gear/gear-cmd-norm", 0.0)
    sset(fdm, "gear/gear-pos-norm", 0.0)
    sset(fdm, "fcs/flap-cmd-norm", 0.0)
    sset(fdm, PROP_SB, float(np.clip(sb, 0.0, SB_CAP)))
    sset(fdm, "fcs/roll-trim-cmd-norm", 0.0)
    sset(fdm, "fcs/yaw-trim-cmd-norm", 0.0)
    set_pitch_trim(fdm, ptrim_hold)
    engine_on(fdm)


def force_ic(fdm, env=None):
    env = env or ENV
    fdm.reset_to_initial_conditions(1)
    fdm.set_property_value("ic/h-sl-ft", env["alt_ft"])
    fdm.set_property_value("ic/vc-kts", env["vc_kts"])
    fdm.set_property_value("ic/gamma-deg", 0.0)
    fdm.set_property_value("ic/theta-deg", env["theta_seed"])
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


def native_trim(fdm, env=None):
    """
    Iterative gamma-bias-corrected native trim. do_trim(1) alone was
    verified to accept a trim with hdot=+36 fps baked in (steady climb, not
    level flight) despite ic/gamma-deg=0 being requested -- gamma was not
    actually being held during the trim search. This re-trims with a
    corrected gamma bias, opposite the residual hdot, until hdot is
    genuinely small (<5 fps), instead of trusting the first answer.
    """
    env = env or ENV
    print("[TRIM] Native do_trim(1) ...")
    force_ic(fdm, env)
    set_throttle(fdm, 0.55)
    set_elev(fdm, -0.02)
    set_pitch_trim(fdm, 0.0)
    ownership(fdm, 0.0, 0.0)

    gamma_bias = 0.0
    st = None
    attempt = 0
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

        V_fps = max(st["vc"] * 1.68781, 50.0)
        residual_gamma_deg = math.degrees(math.asin(np.clip(st["hdot"] / V_fps, -1.0, 1.0)))
        if abs(st["hdot"]) < 5.0:
            break
        gamma_bias -= residual_gamma_deg
        gamma_bias = float(np.clip(gamma_bias, -10.0, 10.0))
        force_ic(fdm, env)
        set_throttle(fdm, 0.55)
        set_elev(fdm, -0.02)
        set_pitch_trim(fdm, 0.0)
        ownership(fdm, 0.0, 0.0)

    ok = (abs(st["theta"]) < 15.0 and abs(st["vc"] - env["vc_kts"]) < 80.0
          and abs(st["hdot"]) < 5.0)
    if not ok:
        print(f"[TRIM] Rejected -- hdot={st['hdot']:+.2f} still outside tolerance after "
              f"{attempt + 1} attempts.")
        return False, 0.55, -0.02, 0.0, 2.5

    elev = st["elev_cmd"] if abs(st["elev_cmd"]) > 1e-6 else st["elev_pos"]
    ptrim = st["ptrim"]
    thr = float(np.clip(st["thr"], 0.15, 0.95))
    print(f"[TRIM] ACCEPTED thr={thr:.4f} elev={elev:+.4f} ptrim={ptrim:+.4f} "
          f"theta0={st['theta']:+.2f} hdot={st['hdot']:+.2f} (genuinely level)")
    return True, thr, float(elev), float(ptrim), float(st["theta"])
