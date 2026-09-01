import numpy as np
from scipy.integrate import solve_ivp

class LongitudinalPlant:
    """
    Penelope PHI-CTRL: Continuous-time longitudinal aircraft dynamics.
    Default matrices (A, B) approximate a transport aircraft at Mach 0.8 cruise.
    
    States (x):  [u (ft/s), w (ft/s), q (rad/s), theta (rad), h (ft)]
    Controls (u): [delta_elevator (rad), delta_throttle (normalized)]
    """
    def __init__(self, dt=0.02):
        self.dt = dt
        self.U0 = 800.0  # Nominal cruise speed (ft/s)
        
        # System Matrix (A): [u, w, q, theta, h]
        # Includes linearized kinematic relation for altitude: h_dot = U0*theta - w
        self.A = np.array([
            [-0.005,  0.015, -15.0,  -32.2,   0.0],
            [-0.090, -0.600,  800.0,   0.0,   0.0],
            [ 0.001, -0.015, -0.500,   0.0,   0.0],
            [ 0.000,  0.000,  1.000,   0.0,   0.0],
            [ 0.000, -1.000,  0.000, self.U0, 0.0] 
        ])
        
        # Control Matrix (B): [elevator, throttle]
        self.B = np.array([
            [  0.0,  10.0],
            [-25.0,   0.0],
            [-2.50,   0.0],
            [  0.0,   0.0],
            [  0.0,   0.0]
        ])
        
        self.state = np.zeros(5)

    def reset(self, initial_state=None):
        self.state = initial_state if initial_state is not None else np.zeros(5)
        return self.state

    def step(self, action, B_effective=None):
        """
        Steps the ODE forward by dt using RK45.
        B_effective allows the fault injector to override control effectiveness mid-flight.
        """
        B_mat = B_effective if B_effective is not None else self.B
        
        # Define the linear ODE: dx/dt = A*x + B*u
        def dynamics(t, x):
            return self.A @ x + B_mat @ action
            
        # Integrate from t=0 to t=dt
        sol = solve_ivp(dynamics, [0, self.dt], self.state, method='RK45')
        self.state = sol.y[:, -1]
        
        return self.state
