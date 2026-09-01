"""
Direct Model Reference Adaptive Controller (MRAC) for pitch channel.

Tracks a 2nd-order ideal short-period reference model.
Returns a scalar elevator residual (or a 2-vector for older call sites).
"""
import numpy as np


class MRACAdaptiveController:
    """
    Lyapunov-based direct MRAC on [q, theta].

    State expected by compute_action: [u, w, q, theta, h]
    target_theta in radians.
    """

    def __init__(self, dt=0.02):
        self.dt = float(dt)

        # Reference model: wn ≈ 1.5 rad/s, zeta ≈ 0.8 (gentler than before)
        wn, zeta = 1.5, 0.80
        self.Am = np.array([
            [-2.0 * zeta * wn, -wn ** 2],
            [1.0,               0.0],
        ], dtype=float)
        self.Bm = np.array([[wn ** 2], [0.0]], dtype=float)

        self.xm = np.zeros((2, 1))

        # P solving Am'P + P Am = -Q (approximate positive-definite)
        self.P = np.array([
            [0.40, 0.08],
            [0.08, 0.55],
        ], dtype=float)

        # Adaptation rates (kept moderate to avoid high-frequency chatter)
        self.gamma_x = 4.0
        self.gamma_r = 3.0

        self.Kx_hat = np.zeros((1, 2))
        self.Kr_hat = np.ones((1, 1))

        # Soft parameter bounds (projection)
        self.Kx_max = 2.0
        self.Kr_max = 3.0
        self.max_elevator = 0.25  # matches SafeMRAC u_max default

        # Sigma-modification leakage (anti-drift under saturation)
        self.sigma = 0.02

    def reset(self):
        self.xm[:] = 0.0
        self.Kx_hat[:] = 0.0
        self.Kr_hat[:] = 1.0

    def compute_action(self, state, target_theta=0.0, delta_gain=0.0):
        """
        Returns either a scalar residual or a 2-vector [elev_norm, 0]
        depending on how the caller unpacks it.  SafeMRAC handles both.
        """
        q = float(state[2])
        theta = float(state[3])
        x_plant = np.array([[q], [theta]], dtype=float)
        r = np.array([[float(target_theta)]], dtype=float)

        # Reference model step
        dx_m = self.Am @ self.xm + self.Bm @ r
        self.xm = self.xm + dx_m * self.dt

        # Tracking error
        e = x_plant - self.xm

        # Lyapunov adaptation
        b_p_e = self.Bm.T @ self.P @ e  # (1,1)
        dKx = -self.gamma_x * (b_p_e @ x_plant.T)
        dKr = -self.gamma_r * (b_p_e @ r.T)

        # Sigma leakage
        dKx = dKx - self.sigma * self.Kx_hat
        dKr = dKr - self.sigma * (self.Kr_hat - 1.0)

        self.Kx_hat = self.Kx_hat + dKx * self.dt
        self.Kr_hat = self.Kr_hat + dKr * self.dt

        # Projection
        self.Kx_hat = np.clip(self.Kx_hat, -self.Kx_max, self.Kx_max)
        self.Kr_hat = np.clip(self.Kr_hat, 0.1, self.Kr_max)

        # Control law
        effective_Kx = self.Kx_hat + float(delta_gain)
        delta_e = float((effective_Kx @ x_plant + self.Kr_hat @ r).item())
        delta_e = float(np.clip(delta_e, -self.max_elevator, self.max_elevator))

        # Normalised residual in [-1, 1] (SafeMRAC will re-scale / rate-limit)
        norm = delta_e / max(self.max_elevator, 1e-6)
        return np.array([norm, 0.0], dtype=np.float32)
