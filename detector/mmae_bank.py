# -*- coding: utf-8 -*-
"""
PHI-CTRL — small Multiple-Model Adaptive Estimation (MMAE) style bank
for elevator-effectiveness hypotheses.

Hypotheses:
  H0: nominal        γ = 1.0
  H1: moderate loss  γ = 0.8
  H2: severe loss    γ = 0.6

Each hypothesis maintains a simple pitch-channel residual model:
  predicted q-dot proxy from elevator command * hypothesized γ.
Bank outputs posterior weights and a point estimate of γ for use by
the residual enable / gain schedule.

This is intentionally lightweight (no full EKF bank) so it runs at 120 Hz
inside the unified loop. Replace the prediction model with a higher-fidelity
twin later without changing the interface.
"""
from __future__ import annotations

import numpy as np


class ElevEffectivenessBank:
    """
    Parameters
    ----------
    gammas : sequence of hypothesized effectiveness values
    process_var : innovation weighting (larger → slower belief change)
    """

    def __init__(self, gammas=(1.0, 0.8, 0.6), process_var=0.05, dt=1.0 / 120.0):
        self.gammas = np.asarray(gammas, dtype=float)
        self.n = len(self.gammas)
        self.process_var = float(process_var)
        self.dt = float(dt)
        self.weights = np.ones(self.n) / self.n
        self._prev_q = 0.0
        self._prev_elev = 0.0

    def reset(self):
        self.weights = np.ones(self.n) / self.n
        self._prev_q = 0.0
        self._prev_elev = 0.0

    def update(self, elev_cmd: float, q_dps: float) -> dict:
        """
        elev_cmd : commanded elevator (pre-plant), normalized
        q_dps    : measured pitch rate, deg/s

        Returns dict with gamma_hat, weights, best_index.
        """
        q = float(q_dps)
        e = float(np.clip(elev_cmd, -1.0, 1.0))
        # Crude pitch acceleration proxy (deg/s^2): measured Δq/dt
        qdot_meas = (q - self._prev_q) / max(self.dt, 1e-6)

        # Expected: more positive elev (nose down on this airframe) → negative q tendency.
        # Scale is empirical; absolute scale cancels in relative likelihoods.
        k = 8.0  # deg/s^2 per unit elev at γ=1 (order-of-magnitude)
        likes = []
        for g in self.gammas:
            qdot_hat = -k * (e * g)
            innov = qdot_meas - qdot_hat
            likes.append(np.exp(-0.5 * (innov ** 2) / max(self.process_var, 1e-6)))

        likes = np.asarray(likes, dtype=float) + 1e-12
        post = self.weights * likes
        self.weights = post / post.sum()

        self._prev_q = q
        self._prev_elev = e

        idx = int(np.argmax(self.weights))
        gamma_hat = float(np.dot(self.weights, self.gammas))
        return {
            "gamma_hat": gamma_hat,
            "weights": self.weights.copy(),
            "best_index": idx,
            "best_gamma": float(self.gammas[idx]),
        }

    def residual_enable(self, gamma_hat: float, threshold: float = 0.92) -> bool:
        """Enable residual path when bank believes effectiveness dropped."""
        return bool(gamma_hat < threshold)
