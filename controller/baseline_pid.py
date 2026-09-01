import numpy as np

class BaselinePIDController:
    """
    PHI-CTRL: Cascaded PID with Phugoid Damping.
    Outer:  h -> theta_cmd (with h_dot damping)
    Middle: theta -> q_cmd
    Inner:  q -> delta_e
    """
    def __init__(self, dt=0.02, u0=800.0):
        self.dt = dt
        self.u0 = u0
        
        # Outer loop: Altitude -> Pitch Angle
        self.Kp_h = 0.0004      # Was 0.002 (WAY too aggressive)
        self.Ki_h = 0.00002     # Small integral for trim offset
        self.Kd_h = 0.003       # PHUGOID DAMPING: rate feedback on climb rate
        
        # Middle loop: Pitch Angle -> Pitch Rate
        self.Kp_theta = 1.2
        
        # Inner loop: Pitch Rate -> Elevator
        self.Kp_q = -1.5
        self.Ki_q = -0.15
        
        self.max_elevator = 0.35  # ~20 deg
        self.max_throttle = 1.0
        
        self.integral_h = 0.0
        self.integral_q = 0.0

    def reset(self):
        self.integral_h = 0.0
        self.integral_q = 0.0

    def compute_action(self, state, target_h=0.0):
        u, w, q, theta, h = state[:5]
        
        # === PHUGOID DAMPING ===
        # Exact kinematic climb rate: h_dot = -w + u0*theta
        h_dot = -w + self.u0 * theta
        
        # 1. Outer Loop: Altitude Error -> Desired Theta
        err_h = target_h - h
        self.integral_h += err_h * self.dt
        self.integral_h = np.clip(self.integral_h, -200.0, 200.0)
        
        # The damping term SUBTRACTS h_dot: if climbing fast, reduce pitch command
        theta_ref = (self.Kp_h * err_h + 
                     self.Ki_h * self.integral_h - 
                     self.Kd_h * h_dot)
        theta_ref = np.clip(theta_ref, -0.17, 0.17)  # Max ±10 deg
        
        # 2. Middle Loop: Theta Error -> Desired Pitch Rate
        err_theta = theta_ref - theta
        q_ref = self.Kp_theta * err_theta
        q_ref = np.clip(q_ref, -0.17, 0.17)  # Max ±10 deg/s
        
        # 3. Inner Loop: Pitch Rate Error -> Elevator
        err_q = q_ref - q
        self.integral_q += err_q * self.dt
        self.integral_q = np.clip(self.integral_q, -0.2, 0.2)
        
        delta_e = self.Kp_q * err_q + self.Ki_q * self.integral_q
        delta_e = np.clip(delta_e, -self.max_elevator, self.max_elevator)
        norm_elevator = delta_e / self.max_elevator
        
        return np.array([norm_elevator, 0.0], dtype=np.float32)