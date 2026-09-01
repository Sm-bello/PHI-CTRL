# -*- coding: utf-8 -*-
"""
PHI-CTRL Layer 1 -- F-16 baseline controller (EnergyHold, V20).

Extracted verbatim from the validated, gate-PASSING run_baseline_recovery.py.
This is a TECS-lite energy-coupled hold: throttle driven by TOTAL energy
error, pitch by direct hdot+altitude-error damping (NOT by an energy-
distribution term -- that was tried and removed, see note below).

Do not reintroduce an energy-distribution (L_err) term into the pitch
command without re-verifying against a long (60s+) closed-loop run first:
it was tested and found to fight the hdot damping whenever altitude and
speed are simultaneously too high in the same direction (the exact
failure mode that blocked this baseline for a long time), producing an
un-arrested climb+accelerate even at throttle floor.
"""
import numpy as np

from plant.jsbsim_plant_f16 import (
    flight_state, wing_level, THR_FLOOR, THR_RATE_MAX, ELEV_RATE_MAX, SB_CAP,
)


class EnergyHold:
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

        # --- Energy distribution error -> logged only, NOT fed to pitch (see module docstring) ---
        pe_err = h_cmd - st["h"]
        ke_err = (V_cmd_fps ** 2 - V_fps ** 2) / (2.0 * self.g)
        L_err = ke_err - pe_err
        self.int_L = float(np.clip(self.int_L + L_err * self.dt, -4000, 4000))

        base_th = self.theta0 if theta_cmd is None else theta_cmd
        th_cmd = float(np.clip(
            base_th + 0.0005 * pe_err - 0.014 * st["hdot"],   # proven damping only
            -6.0, 8.0,
        ))

        # --- Inner pitch loop ---
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
