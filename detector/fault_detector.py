import numpy as np
from dataclasses import dataclass


@dataclass
class FaultState:
    is_faulty: bool
    severity: float
    confidence: float


class GainRatioDetector:
    """
    PHI-CTRL functional fault detector (heuristic gain-ratio, NOT the
    trained CNN-BiLSTM the architecture diagram eventually calls for).

    This REPLACES the old PHITwinDetectorBridge stub, which was
    instantiated with no model_path anywhere in the project, so
    is_loaded was permanently False and update() always returned
    "healthy" -- silently disabling every case that depended on it.

    Calibrates the healthy |q|/|elev_cmd| gain over an initial window,
    then compares a rolling recent gain against that baseline to
    estimate actuator-effectiveness severity. Airframe-agnostic --
    works for c172p and F16 alike since it calibrates empirically
    against whatever plant it's attached to, rather than assuming a
    fixed gain.
    """

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

    def reset(self):
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


# Backward-compat alias -- anything still importing the old stub name
# gets the functional detector instead of a permanently-healthy stub.
PHITwinDetectorBridge = GainRatioDetector
