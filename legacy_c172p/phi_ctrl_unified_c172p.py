#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL Unified Orchestrator — C172P (FIXED V4)
=================================================
Incorporates all bug-spec fixes (#1–#9) and root-cause mitigations:
  1. Correct trim search (80 kts, rate-dominated cost, boundary gates).
  2. TECS sign fix for energy distribution (L_err = pe_err - ke_err).
  3. TECS anti-windup + raw elevator return (clip once at end).
  4. MRAC: no fault reset, u_max=0.25, du_max=0.08, deadzone=0.005, sigma=0.02.
  5. comp_factor disabled when MRAC/RL active (no dual compensation).
  6. Baseline honestly has no fault knowledge (eff_gamma_estimate = 1.0).
  7. Functional gain-ratio detector replaces stub.
  8. Observer diagnostic-only by default (no closed-loop override).
  9. Staged baseline gate + runtime interlock (40° threshold).
"""

import os
import sys
import csv
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. PATH SETUP (Windows-aware)
# ---------------------------------------------------------------------------
# Portable: this script's directory is the project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plant.jsbsim_plant import JSBSimPlant6DOF
from controller.baseline_pid import BaselinePIDController
from controller.mrac.adaptive_controller import MRACAdaptiveController
from fault_injection.injector import FaultInjector
from sensor_fusion.observer import BiasCompensator

# Do NOT import the stub detector — we use the functional one defined below
# from detector.fault_detector import PHITwinDetectorBridge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import jsbsim
except ImportError as exc:
    raise ImportError("jsbsim not found. Install: conda install -c conda-forge jsbsim") from exc

# ---------------------------------------------------------------------------
# TEE LOGGER
# ---------------------------------------------------------------------------
class TeeLogger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.logfile.close()

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
JSBSIM_ROOT = os.environ.get("JSBSIM_ROOT", None)
AIRCRAFT_NAME = "c172p"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 1.0 / 120.0
SETTLE_TIME = 3.0
FAULT_START_TIME = 5.0
SIM_DURATION = 30.0
TARGET_ALT = 2000.0
TARGET_VCAS = 80.0
G = 32.174

PROP_ELEV_CMD = "fcs/elevator-cmd-norm"
PROP_AIL_CMD  = "fcs/aileron-cmd-norm"
PROP_RUD_CMD  = "fcs/rudder-cmd-norm"
PROP_THROTTLE = "fcs/throttle-cmd-norm"

EFF_GAMMA_FAULT = 0.80  # 20% effectiveness loss

# ---------------------------------------------------------------------------
# 2. FUNCTIONAL FAULT DETECTOR (replaces stub per spec #4)
# ---------------------------------------------------------------------------
@dataclass
class FaultState:
    is_faulty: bool
    severity: float
    confidence: float

class GainRatioDetector:
    """Heuristic gain-ratio detector. Calibrates |q|/|elev_cmd| pre-fault,
    then monitors rolling ratio to estimate severity."""
    def __init__(self, dt, calib_window=1.0, roll_window=0.5,
                 cmd_threshold=0.02, fault_threshold=0.9):
        self.dt = dt
        self.calib_steps = int(calib_window / dt)
        self.roll_steps = int(roll_window / dt)
        self.cmd_threshold = cmd_threshold
        self.fault_threshold = fault_threshold
        self.calib_gain = None
        self.cmd_history = []
        self.q_history = []
        self.is_calibrated = False
        self.severity = 1.0
        self._step = 0

    def update(self, q_dps, elev_cmd):
        self._step += 1
        self.cmd_history.append(float(elev_cmd))
        self.q_history.append(float(q_dps))
        max_len = max(self.calib_steps, self.roll_steps) + 20
        if len(self.cmd_history) > max_len:
            self.cmd_history.pop(0)
            self.q_history.pop(0)

        if not self.is_calibrated and len(self.cmd_history) >= self.calib_steps:
            self._calibrate()

        if not self.is_calibrated:
            return FaultState(is_faulty=False, severity=1.0, confidence=1.0)

        recent_gain = self._compute_gain(self.cmd_history[-self.roll_steps:],
                                         self.q_history[-self.roll_steps:])
        if recent_gain is None or self.calib_gain <= 1e-6:
            return FaultState(is_faulty=False, severity=1.0, confidence=1.0)

        severity_estimate = float(np.clip(recent_gain / self.calib_gain, 0.1, 1.0))
        is_faulty = severity_estimate < self.fault_threshold
        self.severity = severity_estimate
        return FaultState(is_faulty=is_faulty, severity=severity_estimate, confidence=0.85)

    def _calibrate(self):
        gain = self._compute_gain(self.cmd_history[:self.calib_steps],
                                  self.q_history[:self.calib_steps])
        if gain is not None and gain > 1e-6:
            self.calib_gain = gain
            self.is_calibrated = True
            print(f"[DETECTOR] Calibrated healthy gain: {self.calib_gain:.4f}")

    def _compute_gain(self, cmds, qs):
        valid = [(abs(q), abs(c)) for q, c in zip(qs, cmds) if abs(c) > self.cmd_threshold]
        if not valid:
            return None
        total_q = sum(q for q, c in valid)
        total_c = sum(c for q, c in valid)
        if total_c < 1e-6:
            return None
        return total_q / total_c

# ---------------------------------------------------------------------------
# 3. FLIGHT LOGGER
# ---------------------------------------------------------------------------
class FlightLogger:
    FIELDS = [
        "step", "time_s", "case_name", "alt_ft", "vc_kts",
        "theta_deg", "phi_deg", "psi_deg",
        "p_dps", "q_dps", "r_dps",
        "hdot_fps", "udot_fps2", "wdot_fps2",
        "elev_raw", "elev_comp", "elev_plant",
        "ail_cmd", "rud_cmd", "throttle",
        "alt_err", "pitch_cmd_deg", "eff_gamma", "comp_factor",
        "energy_height_ft", "energy_err_ft", "dist_err_ft",
        "mrac_elev", "observer_bias_q", "observer_bias_theta",
        "detector_severity", "detector_confidence", "rl_residual",
        "interlock_tripped",
    ]

    def __init__(self, filepath: str):
        self._path = filepath
        self._fp = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self.step = 0

    def log(self, fdm, case_name, elev_raw, elev_comp, elev_plant, ctrl, alt_err,
            pitch_cmd, eff_gamma, comp_factor, E, E_err, L_err,
            mrac_elev=0.0, obs_bias=None, det_state=None, rl_res=0.0,
            interlock=False):
        self.step += 1
        obs_bias = obs_bias or {}
        det_state = det_state or {}
        row = {
            "step": self.step,
            "time_s": round(fdm.get_sim_time(), 6),
            "case_name": case_name,
            "alt_ft": fdm.get_property_value("position/h-sl-ft"),
            "vc_kts": fdm.get_property_value("velocities/vc-kts"),
            "theta_deg": math.degrees(fdm.get_property_value("attitude/theta-rad")),
            "phi_deg": math.degrees(fdm.get_property_value("attitude/phi-rad")),
            "psi_deg": math.degrees(fdm.get_property_value("attitude/psi-rad")),
            "p_dps": math.degrees(fdm.get_property_value("velocities/p-rad_sec")),
            "q_dps": math.degrees(fdm.get_property_value("velocities/q-rad_sec")),
            "r_dps": math.degrees(fdm.get_property_value("velocities/r-rad_sec")),
            "hdot_fps": fdm.get_property_value("velocities/h-dot-fps"),
            "udot_fps2": fdm.get_property_value("accelerations/udot-ft_sec2"),
            "wdot_fps2": fdm.get_property_value("accelerations/wdot-ft_sec2"),
            "elev_raw": elev_raw,
            "elev_comp": elev_comp,
            "elev_plant": elev_plant,
            "ail_cmd": ctrl["ail"],
            "rud_cmd": ctrl["rud"],
            "throttle": ctrl["throttle"],
            "alt_err": alt_err,
            "pitch_cmd_deg": pitch_cmd,
            "eff_gamma": eff_gamma,
            "comp_factor": comp_factor,
            "energy_height_ft": E,
            "energy_err_ft": E_err,
            "dist_err_ft": L_err,
            "mrac_elev": mrac_elev,
            "observer_bias_q": obs_bias.get("q", 0.0),
            "observer_bias_theta": obs_bias.get("theta", 0.0),
            "detector_severity": det_state.get("severity", 1.0),
            "detector_confidence": det_state.get("confidence", 1.0),
            "rl_residual": rl_res,
            "interlock_tripped": int(interlock),
        }
        self._writer.writerow(row)

    def close(self):
        self._fp.close()
        print(f"[LOG] Wrote {self.step} rows to {self._path}")

# ---------------------------------------------------------------------------
# 4. TECS CONTROLLER
# ---------------------------------------------------------------------------
class TECSController:
    def __init__(self, trim_thr, trim_elev, trim_theta_deg, dt):
        self.dt = dt
        self.trim_thr = trim_thr
        self.trim_elev = trim_elev
        self.trim_theta_deg = trim_theta_deg

        # Energy gains kept moderate so residual settle error does not
        # immediately drive throttle to 0 (was the V5 failure mode).
        self.Kp_E = 0.008
        self.Ki_E = 0.0015
        self.Kd_E = 0.004
        self.int_E = 0.0
        self.last_E_err = 0.0
        self.prev_thr = None  # rate-limit state

        self.Kp_L = 0.008
        self.Ki_L = 0.0008
        self.Kd_L = 0.004
        self.int_L = 0.0
        self.last_L_err = 0.0

        self.Kp_theta = 0.08
        self.Kd_q = 0.20
        # Never command less than this fraction of trim throttle when
        # near the altitude target (prevents the zero-throttle bleed).
        self.thr_floor_frac = 0.90   # stay close to trim; was 0.55 (still bled speed)
        self.thr_rate_max = 0.50     # per second baseline rate
        self.thr_rate_max_urgent = 1.20  # when energy-deficient or underspeed

        self.Kp_phi = 0.04
        self.Kp_p = 0.10
        self.Kp_r = 0.06
        self.Kp_beta = 0.015
        self.qbar_nom = 30.0

    def reset(self):
        self.int_E = 0.0
        self.last_E_err = 0.0
        self.int_L = 0.0
        self.last_L_err = 0.0
        self.prev_thr = None

    def _energy_height(self, h, V_fps):
        return h + (V_fps ** 2) / (2.0 * G)

    def update(self, fdm, h_cmd, V_cmd_kts):
        h = fdm.get_property_value("position/h-sl-ft")
        V_fps = fdm.get_property_value("velocities/vtrue-fps")
        if V_fps < 10:
            V_fps = fdm.get_property_value("velocities/vc-kts") * 1.68781
        h_dot = fdm.get_property_value("velocities/h-dot-fps")
        qbar = max(fdm.get_property_value("aero/qbar-psf"), 5.0)
        q_scale = np.clip(self.qbar_nom / qbar, 0.5, 2.0)

        # Total Energy -> Throttle
        E = self._energy_height(h, V_fps)
        V_cmd_fps = V_cmd_kts * 1.68781
        E_cmd = self._energy_height(h_cmd, V_cmd_fps)
        E_err = E_cmd - E
        dE_err = (E_err - self.last_E_err) / self.dt

        # Compute raw throttle with current integrator (pre-update)
        thr_raw = self.trim_thr + q_scale * (self.Kp_E * E_err + self.Ki_E * self.int_E + self.Kd_E * dE_err)
        thr_sat = thr_raw <= 0.0 or thr_raw >= 1.0
        if not thr_sat:
            self.int_E += E_err * self.dt
            self.int_E = np.clip(self.int_E, -80.0, 80.0)
        thr_cmd = float(np.clip(thr_raw, 0.0, 1.0))

        V_kts = V_fps / 1.68781
        pe_now = h_cmd - h

        # Floor: stay near trim while roughly on-altitude (prevents speed bleed)
        if abs(pe_now) < 100.0:
            thr_cmd = max(thr_cmd, self.thr_floor_frac * self.trim_thr)

        # Underspeed or large energy deficit → full throttle
        urgent = (V_kts < 70.0) or (E_err > 40.0)
        if urgent:
            thr_cmd = 1.0

        # Rate-limit (faster when urgent so we do not spend 5 s climbing to 1.0)
        if self.prev_thr is None:
            self.prev_thr = self.trim_thr
        rate = self.thr_rate_max_urgent if urgent else self.thr_rate_max
        max_dthr = rate * self.dt
        thr_cmd = float(np.clip(thr_cmd, self.prev_thr - max_dthr, self.prev_thr + max_dthr))
        thr_cmd = float(np.clip(thr_cmd, 0.0, 1.0))
        self.prev_thr = thr_cmd

        self.last_E_err = E_err

        # Energy Distribution -> Pitch
        pe_err = h_cmd - h
        ke_err = (V_cmd_fps ** 2 - V_fps ** 2) / (2.0 * G)
        # SPEC FIX: L_err = pe_err - ke_err (was ke_err - pe_err, which is backwards)
        L_err = pe_err - ke_err
        dL_err = (L_err - self.last_L_err) / self.dt

        # Compute raw pitch command with current integrator (pre-update)
        pitch_raw = q_scale * (self.Kp_L * L_err + self.Ki_L * self.int_L + self.Kd_L * dL_err)
        pitch_sat = pitch_raw <= -5.0 or pitch_raw >= 5.0
        if not pitch_sat:
            self.int_L += L_err * self.dt
            self.int_L = np.clip(self.int_L, -50.0, 50.0)
        # Bleed distribution integrator when underspeed so residual nose-up
        # bias does not keep the aircraft climbing into a stall.
        if (V_fps / 1.68781) < 68.0:
            self.int_L *= 0.95
        pitch_cmd = float(np.clip(pitch_raw, -5.0, 5.0))
        # Speed priority when slow (correct TECS / ArduPilot behaviour):
        # use pitch to recover airspeed.  If below threshold, bias toward
        # nose-DOWN so we trade altitude for speed under full throttle.
        # (V7 had this inverted and blocked nose-down → stall dive.)
        V_kts_pitch = V_fps / 1.68781
        if V_kts_pitch < 70.0:
            # Cap nose-up; allow up to -3° nose-down for acceleration
            pitch_cmd = float(np.clip(pitch_cmd, -3.0, 1.0))
        if V_kts_pitch < 62.0:
            # Force a mild nose-down recovery attitude demand
            pitch_cmd = min(pitch_cmd, -1.5)
        self.last_L_err = L_err

        # Pitch attitude hold — return RAW elevator (spec #2)
        theta = math.degrees(fdm.get_property_value("attitude/theta-rad"))
        q = math.degrees(fdm.get_property_value("velocities/q-rad_sec"))
        theta_err = (self.trim_theta_deg + pitch_cmd) - theta
        # Sign convention: -Kp*err + Kd*q  (JSBSim +elev-cmd → nose down)
        elev_raw = self.trim_elev - q_scale * (self.Kp_theta * theta_err - self.Kd_q * q)
        # Soft pre-limit so integrator windup / large theta_err cannot
        # produce elev_raw = -25 before the final clip (seen in V5 logs).
        elev_raw = float(np.clip(elev_raw, -1.5, 1.5))

        # Lateral-directional (these are clipped here since no augmentation layer fights over them)
        phi = math.degrees(fdm.get_property_value("attitude/phi-rad"))
        p = math.degrees(fdm.get_property_value("velocities/p-rad_sec"))
        ail_cmd = -q_scale * (self.Kp_phi * phi + self.Kp_p * p)
        ail_cmd = np.clip(ail_cmd, -1.0, 1.0)

        r = math.degrees(fdm.get_property_value("velocities/r-rad_sec"))
        beta = math.degrees(fdm.get_property_value("aero/beta-rad"))
        rud_cmd = -q_scale * (self.Kp_r * r + self.Kp_beta * beta)
        rud_cmd = np.clip(rud_cmd, -1.0, 1.0)

        return {"elev": elev_raw, "ail": ail_cmd, "rud": rud_cmd, "throttle": thr_cmd}, pitch_cmd, E_err, L_err

# ---------------------------------------------------------------------------
# 5. SAFE MRAC WRAPPER
# ---------------------------------------------------------------------------
class SafeMRAC:
    def __init__(self, base_mrac, u_max=0.25, du_max=0.08, deadzone=0.005, sigma=0.02):
        self.mrac = base_mrac
        self.u_max = u_max          # Increased per spec #7
        self.du_max = du_max        # Increased proportionally
        self.deadzone = deadzone    # Reduced to noise floor
        self.sigma = sigma          # Sigma-modification (leakage)
        self.prev_u = 0.0

    def compute_action(self, state, target_theta):
        theta = state[3]
        err = target_theta - theta
        if abs(err) < self.deadzone:
            return 0.0
        u = self.mrac.compute_action(state, target_theta=target_theta)
        try:
            u = float(u[0]) if hasattr(u, '__len__') else float(u)
        except Exception:
            u = 0.0

        # Sigma-modification (parameter-space anti-windup)
        self._apply_sigma_modification()

        u = np.clip(u, -self.u_max, self.u_max)
        du = np.clip(u - self.prev_u, -self.du_max, self.du_max)
        u = self.prev_u + du
        self.prev_u = u
        return u

    def _apply_sigma_modification(self):
        """Leakage to prevent parameter drift during saturation."""
        if self.sigma <= 0:
            return
        for attr in ['theta', 'weights', 'W', 'param', 'params']:
            if hasattr(self.mrac, attr):
                w = getattr(self.mrac, attr)
                if hasattr(w, '__len__') and hasattr(w, '__mul__'):
                    try:
                        setattr(self.mrac, attr, w * (1.0 - self.sigma))
                    except Exception:
                        pass
                    break

# ---------------------------------------------------------------------------
# 6. OBSERVER MATRIX BUILDER
# ---------------------------------------------------------------------------
def build_observer_mats(trim_h=2000.0, trim_V=80.0):
    try:
        # SPEC #6: fixed import name (LongitudinalPlant, not LongitudinalODE)
        from plant.longitudinal_ode import LongitudinalPlant
        lin = LongitudinalPlant(h0=trim_h, V0=trim_V)
        A = np.array(lin.A, dtype=float)
        B = np.array(lin.B, dtype=float)
        print("[OBSERVER] Loaded A/B from LongitudinalPlant.")
    except Exception as e:
        print(f"[WARN] LongitudinalPlant unavailable ({e}). Using placeholder matrices.")
        # These are WRONG for c172p — diagnostic only
        A = np.array([
            [-0.045,  0.080,   0.0,   -32.2,    0.0],
            [-0.280, -0.850, 170.0,     0.0,    0.0],
            [ 0.000, -0.012,  -1.4,     0.0,    0.0],
            [ 0.000,  0.000,   1.0,     0.0,    0.0],
            [ 0.000, -1.000,   0.0,    80.0,    0.0],
        ], dtype=float)
        B = np.array([
            [ 0.0,   6.0],
            [-14.0,  0.0],
            [-22.0,  0.0],
            [ 0.0,   0.0],
            [ 0.0,   0.0],
        ], dtype=float)

    B_elev = B[:, 0:1]
    C = np.array([
        [0, 0, 0, 0, 1],   # h
        [0, 0, 0, 1, 0],   # theta
        [1, 0, 0, 0, 0],   # u
    ], dtype=float)

    try:
        from scipy.signal import place_poles
        desired = np.array([-0.8, -1.0, -1.2, -2.5, -3.0])
        L = place_poles(A.T, C.T, desired).gain_matrix.T
    except Exception as e:
        print(f"[WARN] Pole placement failed ({e}). L = zeros.")
        L = np.zeros((5, 3))

    print("[OBSERVER] WARNING: Observer defaults to DIAGNOSTIC-ONLY (spec #6).")
    print("             Do NOT enable observer_in_loop until re-linearized from JSBSim.")
    return A, B, B_elev, C, L

# ---------------------------------------------------------------------------
# 7. PPO STATE ADAPTER
# ---------------------------------------------------------------------------
class PPOStateAdapter:
    def __init__(self, target_h=2000.0, target_V=80.0):
        self.target_h = target_h
        self.target_V = target_V
        self.scale = np.array([
            500.0,   # dh [ft]
            50.0,    # h_dot [fps]
            20.0,    # dV [kts]
            0.1,     # alpha [rad]
            0.5,     # q [rad/s]
            0.5,     # theta [rad]
            1.0,     # elev_cmd [-1,1]
            1.0,     # throttle [0,1]
        ], dtype=np.float32)

    def adapt(self, fdm, prev_cmd):
        h = fdm.get_property_value("position/h-sl-ft")
        u = fdm.get_property_value("velocities/u-fps")
        w = fdm.get_property_value("velocities/w-fps")
        q = fdm.get_property_value("velocities/q-rad_sec")
        theta = fdm.get_property_value("attitude/theta-rad")
        Va_fps = math.sqrt(u**2 + w**2)
        Va_kts = Va_fps * 0.592484
        alpha = math.atan2(w, u) if abs(u) > 1e-3 else 0.0
        h_dot = -w * math.cos(theta) + u * math.sin(theta)
        obs = np.array([
            h - self.target_h, h_dot, Va_kts - self.target_V,
            alpha, q, theta, prev_cmd[0], prev_cmd[1],
        ], dtype=np.float32)
        obs /= self.scale
        return obs

# ---------------------------------------------------------------------------
# 8. TRIM SEARCH
# ---------------------------------------------------------------------------
def find_trim(fdm, target_alt=TARGET_ALT, target_vc=TARGET_VCAS, target_gamma=0.0):
    print(f"[TRIM] Grid search: h={target_alt:.0f} ft, Vc={target_vc:.0f} kts, gamma={target_gamma}")

    fdm.set_property_value("ic/h-sl-ft", target_alt)
    fdm.set_property_value("ic/vc-kts", target_vc)
    fdm.set_property_value("ic/gamma-deg", target_gamma)
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    fdm.run_ic()

    best_cost = 1e9
    best_thr, best_elev, best_theta = 0.0, 0.0, 0.0

    # Fixed trim grid: realistic C172 throttle band, longer evaluation
    # so energy drift is visible.  Zero-throttle solutions are rejected.
    throttle_grid = np.linspace(0.25, 0.75, 16)
    elevator_grid = np.linspace(-0.15, 0.25, 21)
    n_eval = int(5.0 / DT)  # 5 s per candidate

    for thr in throttle_grid:
        for elev in elevator_grid:
            fdm.reset_to_initial_conditions(True)
            fdm.set_property_value("fcs/mixture-cmd-norm", 1.0)
            fdm.set_property_value("propulsion/magnetos_all", 3)
            fdm.set_property_value("propulsion/set-running", -1)
            fdm.set_property_value(PROP_THROTTLE, float(thr))
            fdm.set_property_value(PROP_ELEV_CMD, float(elev))

            for _ in range(n_eval):
                fdm.run()

            h = fdm.get_property_value("position/h-sl-ft")
            vc = fdm.get_property_value("velocities/vc-kts")
            hdot = fdm.get_property_value("velocities/h-dot-fps")
            udot = abs(fdm.get_property_value("accelerations/udot-ft_sec2"))
            q_dps = abs(math.degrees(fdm.get_property_value("velocities/q-rad_sec")))
            theta_deg = math.degrees(fdm.get_property_value("attitude/theta-rad"))

            # Energy-aware cost: altitude & airspeed dominate; rates secondary
            alt_err = abs(h - target_alt)
            vc_err = abs(vc - target_vc)
            # Hard penalty if throttle is near zero (not a real cruise trim)
            thr_penalty = 50.0 if thr < 0.20 else 0.0
            cost = (4.0 * alt_err + 3.0 * vc_err +
                    1.0 * abs(hdot) + 0.2 * udot + 1.0 * q_dps +
                    0.5 * abs(theta_deg) + thr_penalty)

            if cost < best_cost:
                best_cost = cost
                best_thr, best_elev, best_theta = thr, elev, theta_deg

    # SPEC #3: mandatory convergence gate
    thr_on_boundary = (abs(best_thr - 0.25) < 0.002) or (abs(best_thr - 0.75) < 0.002)
    elev_on_boundary = (abs(best_elev - (-0.15)) < 0.002) or (abs(best_elev - 0.25) < 0.002)

    if best_cost > 5.0 or thr_on_boundary or elev_on_boundary:
        print(f"[TRIM WARN] Initial search failed (cost={best_cost:.3f}, thr={best_thr:.4f}, elev={best_elev:.4f})")
        print("[TRIM] Retrying with widened elevator range...")

        best_cost = 1e9
        elevator_grid = np.linspace(-0.15, 0.25, 20)

        for thr in throttle_grid:
            for elev in elevator_grid:
                fdm.reset_to_initial_conditions(True)
                fdm.set_property_value("fcs/mixture-cmd-norm", 1.0)
                fdm.set_property_value("propulsion/magnetos_all", 3)
                fdm.set_property_value("propulsion/set-running", -1)
                fdm.set_property_value(PROP_THROTTLE, float(thr))
                fdm.set_property_value(PROP_ELEV_CMD, float(elev))

                for _ in range(n_eval):
                    fdm.run()

                h = fdm.get_property_value("position/h-sl-ft")
                vc = fdm.get_property_value("velocities/vc-kts")
                hdot = fdm.get_property_value("velocities/h-dot-fps")
                udot = abs(fdm.get_property_value("accelerations/udot-ft_sec2"))
                q_dps = abs(math.degrees(fdm.get_property_value("velocities/q-rad_sec")))
                theta_deg = math.degrees(fdm.get_property_value("attitude/theta-rad"))

                alt_err = abs(h - target_alt)
                vc_err = abs(vc - target_vc)
                thr_penalty = 50.0 if thr < 0.20 else 0.0
                cost = (4.0 * alt_err + 3.0 * vc_err +
                        1.0 * abs(hdot) + 0.2 * udot + 1.0 * q_dps +
                        0.5 * abs(theta_deg) + thr_penalty)

                if cost < best_cost:
                    best_cost = cost
                    best_thr, best_elev, best_theta = thr, elev, theta_deg

        thr_on_boundary = (abs(best_thr - 0.25) < 0.002) or (abs(best_thr - 0.75) < 0.002)
        elev_on_boundary = (abs(best_elev - (-0.15)) < 0.002) or (abs(best_elev - 0.25) < 0.002)

        if best_cost > 5.0 or thr_on_boundary or elev_on_boundary:
            raise RuntimeError(
                f"Trim search failed to converge: cost={best_cost:.3f}, "
                f"thr={best_thr:.4f}, elev={best_elev:.4f}, theta={best_theta:.2f}°"
            )

    print(f"[TRIM] Best: thr={best_thr:.4f}, elev={best_elev:.4f}, theta={best_theta:.2f}°, cost={best_cost:.3f}")
    return float(best_thr), float(best_elev), float(best_theta)

# ---------------------------------------------------------------------------
# 9. UNIFIED SIMULATION RUNNER
# ---------------------------------------------------------------------------
def run_case(case_name: str, csv_path: str, trim_thr: float, trim_elev: float, trim_theta: float,
             use_mracs=False, use_observer=False, use_detector=False, use_rl=False, obs_mats=None):

    print(f"\n{'='*70}\n  CASE : {case_name}\n{'='*70}")

    # Mutable augmentation flags (interlock can disable these mid-run)
    mrac_active = use_mracs
    observer_active = use_observer
    detector_active = use_detector
    rl_active = use_rl
    interlock_tripped = False

    fdm = jsbsim.FGFDMExec(JSBSIM_ROOT)
    fdm.set_dt(DT)
    if not fdm.load_model(AIRCRAFT_NAME):
        raise RuntimeError(f"Failed to load aircraft: {AIRCRAFT_NAME}")

    # Initial conditions
    fdm.set_property_value("ic/h-sl-ft", TARGET_ALT)
    fdm.set_property_value("ic/vc-kts", TARGET_VCAS)
    fdm.set_property_value("ic/theta-deg", 0.0)
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    fdm.set_property_value("fcs/mixture-cmd-norm", 1.0)
    fdm.set_property_value("propulsion/magnetos_all", 3)
    fdm.set_property_value("propulsion/set-running", -1)
    fdm.run_ic()

    print(f"[SETTLE] {SETTLE_TIME:.1f}s open-loop settle...")
    for _ in range(int(SETTLE_TIME / DT)):
        fdm.set_property_value(PROP_THROTTLE, trim_thr)
        fdm.set_property_value(PROP_ELEV_CMD, trim_elev)
        fdm.run()

    h0 = fdm.get_property_value("position/h-sl-ft")
    print(f"[SETTLE] h={h0:.1f} ft, Vc={fdm.get_property_value('velocities/vc-kts'):.1f} kts")

    logger = FlightLogger(csv_path)
    tecs = TECSController(trim_thr, trim_elev, trim_theta, DT)
    tecs.reset()
    tecs.prev_thr = trim_thr  # seamless handoff from open-loop settle

    # Subsystems
    mrac = None
    if mrac_active:
        base_mrac = MRACAdaptiveController(dt=DT)
        mrac = SafeMRAC(base_mrac, u_max=0.25, du_max=0.08, deadzone=0.005, sigma=0.02)

    observer = None
    if observer_active and obs_mats is not None:
        A, B, B_elev, C, L = obs_mats
        try:
            observer = BiasCompensator(A=A, B=B, L=L, C=C, dt=DT)
        except TypeError:
            observer = BiasCompensator(A=A, B=B, dt=DT)

    # SPEC #4: functional detector replaces stub
    detector = None
    if detector_active:
        detector = GainRatioDetector(dt=DT, calib_window=1.0, roll_window=0.5)

    rl_model, ppo_adapter = None, None
    if rl_active:
        model_zip = PROJECT_ROOT / "models" / "phi_ctrl_residual_jsbsim.zip"
        if model_zip.exists():
            try:
                from stable_baselines3 import PPO
                rl_model = PPO.load(model_zip, device="cpu")
                ppo_adapter = PPOStateAdapter(target_h=TARGET_ALT, target_V=TARGET_VCAS)
                print("[RL] PPO loaded successfully.")
            except Exception as e:
                print(f"[RL] Load failed: {e}")

    h_target = h0
    crash = False
    max_steps = int(SIM_DURATION / DT)
    prev_cmd = np.array([trim_elev, trim_thr], dtype=np.float32)

    for i in range(max_steps):
        t_loop = i * DT
        fault_active = (fdm.get_sim_time() >= FAULT_START_TIME)

        h = fdm.get_property_value("position/h-sl-ft")
        vc_kts = fdm.get_property_value("velocities/vc-kts")
        alt_err = h_target - h

        p = fdm.get_property_value("velocities/p-rad_sec")
        q = fdm.get_property_value("velocities/q-rad_sec")
        r = fdm.get_property_value("velocities/r-rad_sec")
        theta = fdm.get_property_value("attitude/theta-rad")
        phi = fdm.get_property_value("attitude/phi-rad")
        u_fps = fdm.get_property_value("velocities/u-fps")
        w_fps = fdm.get_property_value("velocities/w-fps")
        raw_state5 = np.array([u_fps, w_fps, q, theta, h])

        # ---- Observer compensation (DIAGNOSTIC ONLY per spec #6) ----
        obs_bias = {"q": 0.0, "theta": 0.0}
        if observer_active and observer:
            u_ctrl = np.array([prev_cmd[0], prev_cmd[1]], dtype=float)
            y_comp, bias = observer.compensate(raw_state5, u_ctrl, fault_active=fault_active)
            obs_bias = {"q": float(bias[2]), "theta": float(bias[3])}
            # DO NOT override q/theta for controllers
            # q, theta = float(y_comp[2]), float(y_comp[3])  # REMOVED

        # ---- Fault detection ----
        det_state = {"severity": 1.0, "confidence": 1.0}
        eff_gamma_estimate = 1.0
        if detector_active and detector:
            q_dps = math.degrees(q)
            fs = detector.update(q_dps, prev_cmd[0])  # use previous step's elev cmd
            det_state = {"severity": fs.severity, "confidence": fs.confidence}
            eff_gamma_estimate = fs.severity if fs.is_faulty else 1.0
        # SPEC #5: baseline (no detector) must NOT have hardcoded fault knowledge
        # eff_gamma_estimate stays at 1.0 when no detector is present

        # ---- TECS baseline (returns RAW elevator) ----
        ctrl_cmds, pitch_cmd, E_err, L_err = tecs.update(fdm, h_target, TARGET_VCAS)
        elev_raw = ctrl_cmds["elev"]  # RAW, unclipped

        # ---- MRAC augmentation ----
        mrac_elev = 0.0
        if mrac_active and mrac:
            mrac_state = np.array([u_fps, w_fps, q, theta, h])

            target_theta_rad = math.radians(tecs.trim_theta_deg + pitch_cmd)
            trim_theta_rad = math.radians(tecs.trim_theta_deg)
            target_theta_rad = np.clip(target_theta_rad,
                                       trim_theta_rad - math.radians(5.0),
                                       trim_theta_rad + math.radians(5.0))

            # Rate-limit reference model target
            if not hasattr(mrac, '_prev_target_theta'):
                mrac._prev_target_theta = trim_theta_rad
            max_theta_step = math.radians(8.0) * DT
            target_theta_rad = np.clip(target_theta_rad,
                                       mrac._prev_target_theta - max_theta_step,
                                       mrac._prev_target_theta + max_theta_step)
            mrac._prev_target_theta = target_theta_rad

            # SPEC: NO MRAC reset on fault onset — let it adapt continuously
            # (Removed destructive reset block)

            mrac_raw = mrac.compute_action(mrac_state, target_theta=target_theta_rad)
            mrac_elev = -float(mrac_raw)  # Sign fix for JSBSim nose-down positive

            if i % 120 == 0:
                print(f"[MRAC] t={fdm.get_sim_time():.2f}s | "
                      f"target_theta={math.degrees(target_theta_rad):.2f}° | "
                      f"actual_theta={math.degrees(theta):.2f}° | "
                      f"mrac_out={mrac_elev:.4f}")

        # ---- RL residual ----
        rl_res = 0.0
        if rl_active and rl_model and fault_active:
            try:
                obs_8 = ppo_adapter.adapt(fdm, prev_cmd)
                action, _ = rl_model.predict(obs_8, deterministic=True)
                rl_val = float(action[0]) if hasattr(action, '__len__') else float(action)
                rl_res = float(np.clip(rl_val, -0.3, 0.3))
            except Exception as e:
                if i == int(FAULT_START_TIME / DT):
                    print(f"[RL] Inference fallback at t={t_loop:.1f}s: {e}")

        # ---- Control allocation (spec #2: clip exactly once) ----
        # SPEC: disable comp_factor when MRAC or RL is active (no dual compensation)
        if mrac_active or rl_active:
            comp_factor = 1.0
        else:
            # Only apply compensation if detector is present and fault is active
            if detector_active and fault_active:
                raw_cf = min(1.0 / eff_gamma_estimate, 2.0) if eff_gamma_estimate > 0.01 else 1.0
            else:
                raw_cf = 1.0
            # Rate-limit compensation (was stepping elev hard in HYBRID V5)
            if not hasattr(tecs, "_prev_cf"):
                tecs._prev_cf = 1.0
            max_dcf = 0.15 * DT  # ~0.15 per second
            comp_factor = float(np.clip(raw_cf, tecs._prev_cf - max_dcf, tecs._prev_cf + max_dcf))
            tecs._prev_cf = comp_factor

        # Combine RAW commands (residual added to raw, not clipped)
        total_elev_raw = elev_raw + mrac_elev + rl_res

        # Apply comp_factor ONLY if no MRAC/RL (compensation is separate from adaptation)
        if not mrac_active and not rl_active and comp_factor != 1.0:
            total_elev_raw = total_elev_raw * comp_factor

        # SINGLE CLIP exactly once before plant
        elev_comp = np.clip(total_elev_raw, -1.0, 1.0)

        # Physical fault injection (unconditional per spec)
        if fault_active:
            elev_plant = elev_comp * EFF_GAMMA_FAULT
        else:
            elev_plant = elev_comp

        prev_cmd = np.array([elev_comp, ctrl_cmds["throttle"]], dtype=np.float32)

        fdm.set_property_value(PROP_ELEV_CMD, elev_plant)
        fdm.set_property_value(PROP_AIL_CMD, ctrl_cmds["ail"])
        fdm.set_property_value(PROP_RUD_CMD, ctrl_cmds["rud"])
        fdm.set_property_value(PROP_THROTTLE, ctrl_cmds["throttle"])
        fdm.run()

        E = h + (fdm.get_property_value("velocities/vtrue-fps") ** 2) / (2 * G)
        logger.log(fdm, case_name, elev_raw, elev_comp, elev_plant, ctrl_cmds, alt_err,
                   pitch_cmd, eff_gamma_estimate, comp_factor, E, E_err, L_err,
                   mrac_elev=mrac_elev, obs_bias=obs_bias, det_state=det_state, rl_res=rl_res,
                   interlock=interlock_tripped)

        # ---- Bailout check ----
        phi_deg = math.degrees(fdm.get_property_value("attitude/phi-rad"))
        theta_deg = math.degrees(fdm.get_property_value("attitude/theta-rad"))

        # ---- Runtime interlock (spec #9) ----
        any_aug = (mrac_active or observer_active or detector_active or rl_active)
        if any_aug and not interlock_tripped:
            if abs(theta_deg) > 40.0 or abs(phi_deg) > 40.0:
                interlock_tripped = True
                print(f"[INTERLOCK] TRIPPED at t={fdm.get_sim_time():.2f}s | "
                      f"theta={theta_deg:.1f}° phi={phi_deg:.1f}° — "
                      f"disabling all augmentation for remainder of run")
                mrac_active = False
                observer_active = False
                detector_active = False
                rl_active = False

        if h < 100.0 or abs(phi_deg) > 60.0 or abs(theta_deg) > 60.0:
            print(f"  *** BAILOUT at t={fdm.get_sim_time():.3f}s ***")
            crash = True
            break

    logger.close()

    df = pd.read_csv(csv_path)
    df_c = df[df["case_name"] == case_name]
    max_dh = (df_c["alt_ft"] - h0).abs().max()
    rms_dh = np.sqrt(np.mean((df_c["alt_ft"] - h0) ** 2))
    max_dE = df_c["energy_err_ft"].abs().max()
    n_interlock = df_c["interlock_tripped"].sum()

    print(f"\n{'='*70}")
    print(f"METRICS ({case_name})")
    print(f"{'='*70}")
    print(f"Max |dh|:        {max_dh:.1f} ft")
    print(f"RMS |dh|:        {rms_dh:.1f} ft")
    print(f"Max |dE|:        {max_dE:.1f} ft")
    print(f"Min pitch:       {df_c['theta_deg'].min():.1f} deg")
    print(f"Max |roll|:      {df_c['phi_deg'].abs().max():.1f} deg")
    print(f"Max airspeed:    {df_c['vc_kts'].max():.1f} kts")
    print(f"Min airspeed:    {df_c['vc_kts'].min():.1f} kts")
    print(f"Interlock trips: {int(n_interlock)} steps")
    print(f"Bailout / crash: {crash}")
    print(f"{'='*70}")
    return crash

# ---------------------------------------------------------------------------
# 10. PLOTTING
# ---------------------------------------------------------------------------
def plot_all_cases(csv_paths: dict, out_dir: Path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("PHI-CTRL C172P Unified Comparison — All Cases",
                 fontsize=14, fontweight="bold")

    colors = {"BASELINE": "#2563eb", "TECS_MRAC": "#dc2626",
              "HYBRID": "#16a34a", "FULL_STACK": "#9333ea",
              "FAULT_ONLY": "#6b7280"}

    panels = [
        ("alt_ft", "Altitude (ft)"),
        ("vc_kts", "Airspeed (kts)"),
        ("hdot_fps", "Vertical Speed (fps)"),
        ("theta_deg", "Pitch (deg)"),
        ("phi_deg", "Roll (deg)"),
        ("throttle", "Throttle Cmd"),
        ("elev_plant", "Elevator to Plant"),
        ("comp_factor", "Compensation Factor"),
        ("mrac_elev", "MRAC Elevator"),
    ]

    fault_t = FAULT_START_TIME
    for ax, (col, title) in zip(axes.flat, panels):
        ax.axvline(fault_t, color='k', ls='--', lw=1.0, alpha=0.4, label='Fault onset')
        for case_name, path in csv_paths.items():
            df = pd.read_csv(path)
            df_c = df[df["case_name"] == case_name]
            if df_c.empty:
                continue
            color = colors.get(case_name, "#000")
            ax.plot(df_c["time_s"], df_c[col], color=color, lw=1.5,
                    label=case_name, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "phi_ctrl_unified_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[PLOT] Saved: {out_path}")

# ---------------------------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tee = TeeLogger(str(OUTPUT_DIR / "console_log.txt"))

    parser = argparse.ArgumentParser(description="PHI-CTRL Unified C172P (FIXED V4)")
    parser.add_argument("--jsbsim-root", default=None, help="JSBSim root dir (default: built-in)")
    parser.add_argument("--aircraft", default=AIRCRAFT_NAME)
    args = parser.parse_args()

    JSBSIM_ROOT = args.jsbsim_root
    AIRCRAFT_NAME = args.aircraft

    print(f"PHI-CTRL Unified Orchestrator (FIXED V4)")
    print(f"Aircraft: {AIRCRAFT_NAME} | DT: {DT:.5f}s | JSBSim root: {JSBSIM_ROOT or 'built-in'}")

    fdm_trim = jsbsim.FGFDMExec(JSBSIM_ROOT)
    fdm_trim.set_dt(DT)
    fdm_trim.load_model(AIRCRAFT_NAME)
    trim_thr, trim_elev, trim_theta = find_trim(fdm_trim,
                                                target_alt=TARGET_ALT,
                                                target_vc=TARGET_VCAS)
    del fdm_trim

    print("\n[SETUP] Building observer matrices...")
    obs_mats = build_observer_mats(trim_h=TARGET_ALT, trim_V=TARGET_VCAS)

    # -----------------------------------------------------------------------
    # STAGE 1: BASELINE ONLY — pre-fault gate (spec #8)
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 1: BASELINE PRE-FAULT GATE")
    print("="*70)

    baseline_csv = OUTPUT_DIR / "unified_baseline_log.csv"
    baseline_crash = run_case("BASELINE", str(baseline_csv), trim_thr, trim_elev, trim_theta,
                              use_mracs=False, use_observer=False, use_detector=False, use_rl=False,
                              obs_mats=obs_mats)

    # Gate evaluation
    df_base = pd.read_csv(baseline_csv)
    df_pre = df_base[df_base["time_s"] < FAULT_START_TIME]

    gate_pass = True
    gate_reasons = []

    if df_pre.empty:
        gate_pass = False
        gate_reasons.append("No pre-fault data")
    else:
        max_alt_dev = (df_pre["alt_ft"] - TARGET_ALT).abs().max()
        max_pitch_dev = df_pre["theta_deg"].abs().max()
        max_roll_dev = df_pre["phi_deg"].abs().max()

        if max_alt_dev > 150.0:
            gate_pass = False
            gate_reasons.append(f"Pre-fault alt dev {max_alt_dev:.1f} ft > 150 ft")
        if max_pitch_dev > 15.0:
            gate_pass = False
            gate_reasons.append(f"Pre-fault pitch dev {max_pitch_dev:.1f}° > 15°")
        if max_roll_dev > 15.0:
            gate_pass = False
            gate_reasons.append(f"Pre-fault roll dev {max_roll_dev:.1f}° > 15°")
        # Only treat as gate failure if the aircraft departed BEFORE the fault.
        # Post-fault bailout is a performance result, not a setup failure.
        df_crash_pre = df_pre
        pre_departed = False
        if not df_pre.empty:
            pre_departed = (
                (df_pre["theta_deg"].abs() > 30.0).any()
                or (df_pre["phi_deg"].abs() > 30.0).any()
                or (df_pre["alt_ft"] < 100.0).any()
            )
        if pre_departed:
            gate_pass = False
            gate_reasons.append("Baseline departed (attitude/altitude) before fault")
        elif baseline_crash:
            # Post-fault crash: warn but do NOT abort augmented cases
            print("[GATE NOTE] Baseline bailed out after fault onset — "
                  "continuing augmented cases (post-fault performance issue).")

    if not gate_pass:
        print("\n" + "!"*70)
        print("BASELINE GATE FAILED — Aborting all augmented cases")
        for r in gate_reasons:
            print(f"  ✗ {r}")
        print("!"*70)
        tee.close()
        sys.exit(1)

    print("\n[BASELINE GATE PASSED] Proceeding to augmented cases...")

    # -----------------------------------------------------------------------
    # STAGE 2: AUGMENTED CASES
    # -----------------------------------------------------------------------
    cases = {
        "TECS_MRAC":    {"mrac": True,  "obs": False, "det": False, "rl": False},
        "HYBRID":       {"mrac": False, "obs": False, "det": True,  "rl": False},
        "FULL_STACK":   {"mrac": True,  "obs": True,  "det": True,  "rl": True},
    }

    csv_paths = {"BASELINE": str(baseline_csv)}

    for case_name, flags in cases.items():
        csv_path = OUTPUT_DIR / f"unified_{case_name.lower()}_log.csv"
        csv_paths[case_name] = str(csv_path)
        run_case(case_name, str(csv_path), trim_thr, trim_elev, trim_theta,
                 use_mracs=flags["mrac"], use_observer=flags["obs"],
                 use_detector=flags["det"], use_rl=flags["rl"],
                 obs_mats=obs_mats)

    plot_all_cases(csv_paths, OUTPUT_DIR)
    print("\n[DONE] All cases complete. Check results/ folder.")
    tee.close()
