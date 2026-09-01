"""
Luenberger-style bias estimator / compensator for PHI-CTRL.

Kept diagnostic-only in the unified orchestrator until the linear model
is re-derived from a real JSBSim trim.
"""
import numpy as np


class BiasCompensator:
    """
    Parallel-model bias estimator.

    Accepts either the simple (A, B, dt) signature or the extended
    (A, B, L, C, dt) signature used by the unified orchestrator.
    Extra kwargs (L, C, ...) are accepted and stored but the basic
    innovation update does not require them.
    """

    def __init__(self, A, B, dt, alpha=0.02, L=None, C=None, **kwargs):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.L = L
        self.C = C
        n = self.A.shape[0]
        self.x_pred = np.zeros(n)
        self.bias_est = np.zeros(n)

    def reset(self):
        self.x_pred[:] = 0.0
        self.bias_est[:] = 0.0

    def compensate(self, y_meas, u, fault_active=False):
        y_meas = np.asarray(y_meas, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)

        # Predict
        dx = self.A @ self.x_pred + self.B @ u
        self.x_pred = self.x_pred + self.dt * dx

        # Innovation
        innov = y_meas - self.x_pred

        if fault_active:
            self.bias_est += self.alpha * (innov - self.bias_est)
            self.bias_est = np.clip(self.bias_est, -100.0, 100.0)

        y_comp = y_meas - self.bias_est
        # Nudge predictor toward compensated measurement
        self.x_pred += 0.05 * (y_comp - self.x_pred)

        return y_comp, self.bias_est
