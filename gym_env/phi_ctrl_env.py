import gymnasium as gym
from gymnasium import spaces
import numpy as np
from plant.longitudinal_ode import LongitudinalPlant
from fault_injection.injector import FaultInjector
from controller.baseline_pid import BaselinePIDController

class PhiCtrlEnv(gym.Env):
    """
    Penelope PHI-CTRL: Production-Grade Residual-Learning Gymnasium Environment.
    Wraps the plant with a Cascaded PID baseline and trains the PPO agent 
    to learn bounded additive residual corrections during detected faults.
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, max_steps=1000, fault_time_range=(3.0, 5.0)):
        super(PhiCtrlEnv, self).__init__()
        
        self.dt = 0.02
        self.max_steps = max_steps
        self.fault_time_range = fault_time_range
        
        # Core Subsystems
        self.plant = LongitudinalPlant(dt=self.dt)
        self.fault_injector = FaultInjector(nominal_B=self.plant.B)
        self.pid_baseline = BaselinePIDController(dt=self.dt)
        
        # Flight Envelope Constraints
        self.max_elevator = 0.35  # radians (~20 deg)
        self.max_throttle = 1.0
        
        # Action Space: Bounded residual correction multiplier [-0.3, 0.3]
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(1,), dtype=np.float32)
        
        # Observation Space: [u, w, q, theta, h, target_h, pid_action, fault_estimate] (Shape: 8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.last_pid_action = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset Subsystems
        self.state = self.plant.reset()
        self.pid_baseline.reset()
        self.last_pid_action = 0.0
        
        # Curriculum Learning: Randomize target profile (0ft level flight or 50ft/100ft climbs)
        self.target_altitude = np.random.choice([0.0, 50.0, 100.0])
        
        # Randomize fault timing and severe degradation parameters (30% to 70% loss)
        self.fault_trigger_time = np.random.uniform(*self.fault_time_range)
        self.fault_trigger_step = int(self.fault_trigger_time / self.dt)
        self.elevator_health = np.random.uniform(0.3, 0.7)
        
        self.fault_active = False
        self.detector_estimate = 1.0
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Update Fault Schedule & Detector Mock State
        if self.current_step >= self.fault_trigger_step:
            self.fault_active = True
            self.detector_estimate = self.elevator_health
            
        current_health = self.elevator_health if self.fault_active else 1.0
        B_eff = self.fault_injector.get_effective_B(elevator_health=current_health)
        
        # 2. Compute Baseline Cascaded PID Action
        pid_action_arr = self.pid_baseline.compute_action(self.state, target_h=self.target_altitude)
        pid_elevator = float(pid_action_arr[0])
        self.last_pid_action = pid_elevator
        
        # 3. Apply Residual Correction strictly under active fault (Safety Shield)
        residual_correction = float(action[0]) if self.fault_active else 0.0
        combined_elevator = np.clip(pid_elevator + residual_correction, -1.0, 1.0)
        
        physical_action = np.array([
            combined_elevator * self.max_elevator,
            0.0  # Cruise throttle trim
        ], dtype=np.float32)
        
        # 4. Step Plant ODE Integration
        self.state = self.plant.step(physical_action, B_effective=B_eff)
        u, w, q, theta, h = self.state[:5]
        
        # 5. Advanced Reward Shaping (Tracking Error + Rate Penalties + Safety Envelope)
        alt_error = self.target_altitude - h
        
        reward = -(
            2.5 * (alt_error**2) / 100.0 +
            2.0 * (theta**2) +
            1.0 * (q**2) +
            0.5 * (residual_correction**2)  # Penalize excessive AI override magnitude
        )
        reward += 3.0  # Continuous survival bonus
        
        # 6. Envelope Violation & Termination Checks
        terminated = False
        if abs(theta) > 1.0 or abs(h) > 400.0:
            reward -= 400.0
            terminated = True
            
        truncated = bool(self.current_step >= self.max_steps)
        
        info = {
            "fault_active": self.fault_active,
            "elevator_health": current_health,
            "altitude": h,
            "target_altitude": self.target_altitude,
            "residual_correction": residual_correction
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        """Strictly constructs the 8-dimensional observation vector."""
        u, w, q, theta, h = self.state[:5]
        return np.array([
            u, 
            w, 
            q, 
            theta, 
            h, 
            self.target_altitude, 
            self.last_pid_action, 
            self.detector_estimate
        ], dtype=np.float32)
