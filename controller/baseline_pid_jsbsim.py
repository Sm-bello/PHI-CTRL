import numpy as np


class BaselinePIDJSBSim:
    """
    PHI-CTRL: Cascaded PID baseline, retuned for JSBSim c172p units
    (altitude in ft, angles in degrees, cruise ~100-120 kts / ~170-200 ft/s)
    instead of the linear-ODE plant's Mach-0.8 transport regime (U0=800 ft/s).

    Sign convention (verified empirically against c172p's own FCS + Cmde
    table, see aero/coefficient/Cmde in the JSBSim console log):
        elev_cmd = +1  ->  elevator-pos-deg = +23  ->  Cmde more NEGATIVE
                        ->  NOSE DOWN
        elev_cmd = -1  ->  elevator-pos-deg = -28  ->  NOSE UP
    i.e. POSITIVE elev_cmd is nose-down on this airframe. Every term below
    is signed accordingly:  elev = -Kp*pitch_err + Kd*q_deg_s  (too-high
    theta => negative pitch_err => POSITIVE elev => nose down, correct).
    """

    def __init__(self, dt=1.0 / 120.0):
        self.dt = dt

        # Outer loop: altitude error (ft) -> desired pitch (deg)
        self.kp_alt = 0.035
        self.ki_alt = 0.004
        self.kd_alt = 0.002  # damps on h_dot (fps), not raw error derivative

        # Inner loop: pitch error (deg) + pitch rate (deg/s) -> elevator (norm)
        self.kp_pitch = 0.35
        self.kd_pitch = 0.12

        self.int_alt = 0.0
        self.last_alt_err = 0.0
        self.pitch_cmd_deg = 2.0  # small nose-up trim target, deg

    def reset(self):
        self.int_alt = 0.0
        self.last_alt_err = 0.0
        self.pitch_cmd_deg = 2.0

    def compute_action(self, state, target_h):
        """
        state: dict with keys 'h_ft', 'hdot_fps', 'theta_deg', 'q_dps'
        target_h: target altitude, ft
        Returns RAW (UNCLIPPED) normalized elevator command. Caller is
        responsible for applying fault compensation THEN clipping exactly
        once -- do not clip here (see run_jsbsim_layer3_fixed_v3.py for why).
        """
        h = state["h_ft"]
        hdot = state["hdot_fps"]
        theta_deg = state["theta_deg"]
        q_dps = state["q_dps"]

        alt_err = target_h - h
        self.int_alt += alt_err * self.dt
        self.int_alt = np.clip(self.int_alt, -80.0, 80.0)

        self.pitch_cmd_deg = (
            2.0
            + self.kp_alt * alt_err
            + self.ki_alt * self.int_alt
            - self.kd_alt * hdot
        )
        self.pitch_cmd_deg = np.clip(self.pitch_cmd_deg, -15.0, 15.0)

        pitch_err = self.pitch_cmd_deg - theta_deg
        elev_raw = -self.kp_pitch * pitch_err + self.kd_pitch * q_dps

        return elev_raw, self.pitch_cmd_deg
