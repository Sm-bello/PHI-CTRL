"""
PHI-CTRL residual env — train/infer residual path MATCHED to phi_ctrl_unified_f16.py

Unified residual law (must stay identical):
  raw_a   = clip(action[0], -0.5, 0.5)
  deficit = max(1 - gamma_hat, 0.15)          # when fault active
  desired = clip(raw_a * 0.35 * deficit, -0.20, 0.20)
  rl_res  = prev + clip(desired - prev, -0.05, 0.05)   # rate limit
  elev_comp = clip(elev_raw * kappa + rl_res, -1, 1)
  elev_plant = elev_comp * physical_gamma
  kappa = min(1/gamma_hat, 4)

Observation (8,): same as unified FULL_STACK obs8
"""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim

from plant.jsbsim_plant_f16 import (
    DT, ENV, PROP_AIL, PROP_RUD, native_trim, ownership, force_ic,
    set_throttle, set_elev, set_pitch_trim, flight_state,
)
from controller.energy_hold_f16 import EnergyHold

# ---- MUST MATCH phi_ctrl_unified_f16.py constants ----
RESIDUAL_MAX = 0.20
RESIDUAL_RATE_MAX = 0.05
RESIDUAL_GAIN = 0.35
KAPPA_MAX = 4.0
RESIDUAL_ENABLE_GAMMA = 0.92

DEFAULT_TRIM_GRID = [
    {"alt_ft": 10000.0, "vc_kts": 350.0, "theta_seed": 2.5},
    {"alt_ft": 10000.0, "vc_kts": 400.0, "theta_seed": 2.5},
    {"alt_ft": 15000.0, "vc_kts": 400.0, "theta_seed": 2.5},
    {"alt_ft": 15000.0, "vc_kts": 450.0, "theta_seed": 2.0},
    {"alt_ft": 20000.0, "vc_kts": 400.0, "theta_seed": 2.5},
    {"alt_ft": 20000.0, "vc_kts": 450.0, "theta_seed": 2.0},
]


class JSBSimF16PhiCtrlEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        max_episode_time_s=20.0,
        fault_time_range=(3.0, 8.0),
        fault_severity_range=(0.3, 0.7),
        settle_time_s=6.0,
        trim_grid=None,
        curriculum_phase=0.0,
        randomize_trim=True,
        gamma_noise_std=0.02,
        gamma_lag_s=1.0,
    ):
        super().__init__()
        self.dt = DT
        self.max_steps = int(max_episode_time_s / self.dt)
        self.fault_time_range = fault_time_range
        self.fault_severity_range = fault_severity_range
        self.settle_time_s = settle_time_s
        self.randomize_trim = randomize_trim
        self.curriculum_phase = float(np.clip(curriculum_phase, 0.0, 1.0))
        self.gamma_noise_std = float(gamma_noise_std)
        self.gamma_lag_s = float(gamma_lag_s)

        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self.fdm = None
        self._trim_cache = []
        self._active_trim = None
        self._build_fdm_and_trim_grid(trim_grid or DEFAULT_TRIM_GRID)

        self.baseline = None
        self.last_pid_action = 0.0
        self.prev_rl_res = 0.0

    def _build_fdm_and_trim_grid(self, grid):
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_dt(self.dt)
        if not self.fdm.load_model("f16"):
            raise RuntimeError("Failed to load f16 model")

        print(f"[ENV] Pre-computing trim grid ({len(grid)} points)...")
        accepted = []
        for i, point in enumerate(grid):
            env = {
                "alt_ft": float(point["alt_ft"]),
                "vc_kts": float(point["vc_kts"]),
                "theta_seed": float(point.get("theta_seed", 2.5)),
                "desc": f"grid[{i}]",
            }
            ok, thr, elev, ptrim, theta = native_trim(self.fdm, env)
            if ok:
                accepted.append({
                    "env": env, "thr": thr, "elev": elev,
                    "ptrim": ptrim, "theta": theta, "ok": True,
                })
                print(
                    f"[ENV]   grid[{i}] OK  h={env['alt_ft']:.0f} "
                    f"Vc={env['vc_kts']:.0f} thr={thr:.3f} θ={theta:+.2f}"
                )
            else:
                print(f"[ENV]   grid[{i}] FAIL — skipped")

        if not accepted:
            ok, thr, elev, ptrim, theta = native_trim(self.fdm, ENV)
            if not ok:
                raise RuntimeError("F16 trim failed")
            accepted = [{
                "env": dict(ENV), "thr": thr, "elev": elev,
                "ptrim": ptrim, "theta": theta, "ok": True,
            }]

        self._trim_cache = accepted
        self._nominal_idx = 0
        for i, e in enumerate(self._trim_cache):
            if (
                abs(e["env"]["alt_ft"] - 15000.0) < 1.0
                and abs(e["env"]["vc_kts"] - 400.0) < 1.0
            ):
                self._nominal_idx = i
                break
        print(f"[ENV] Trim cache: {len(self._trim_cache)} pts (nominal={self._nominal_idx})")

    def set_curriculum_phase(self, phase: float):
        self.curriculum_phase = float(np.clip(phase, 0.0, 1.0))

    def _sample_trim_entry(self):
        if not self.randomize_trim or len(self._trim_cache) == 1:
            return self._trim_cache[self._nominal_idx]
        if self.np_random.random() > self.curriculum_phase:
            return self._trim_cache[self._nominal_idx]
        idx = int(self.np_random.integers(0, len(self._trim_cache)))
        return self._trim_cache[idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        entry = self._sample_trim_entry()
        self._active_trim = entry
        env = entry["env"]
        thr, elev, ptrim, theta = entry["thr"], entry["elev"], entry["ptrim"], entry["theta"]

        force_ic(self.fdm, env)
        set_throttle(self.fdm, thr)
        set_elev(self.fdm, elev)
        set_pitch_trim(self.fdm, ptrim)
        ownership(self.fdm, ptrim, 0.0)
        self.fdm.run_ic()

        self.baseline = EnergyHold(thr, elev, ptrim, theta, self.dt)
        for _ in range(int(self.settle_time_s / self.dt)):
            cmds = self.baseline.update(self.fdm, env["alt_ft"], env["vc_kts"])
            set_elev(self.fdm, cmds["elev"])
            set_pitch_trim(self.fdm, cmds["ptrim"])
            set_throttle(self.fdm, cmds["throttle"])
            self.fdm.set_property_value(PROP_AIL, cmds["ail"])
            self.fdm.set_property_value(PROP_RUD, cmds["rud"])
            ownership(self.fdm, cmds["ptrim"], cmds["speedbrake"])
            self.fdm.run()

        st = flight_state(self.fdm)
        self.baseline.elev0 = st["elev_cmd"] if abs(st["elev_cmd"]) > 1e-6 else elev
        self.baseline.thr0 = max(st["thr"] if st["thr"] else thr, 0.0)
        self.baseline.theta0 = st["theta"]
        self.baseline.prev_elev = self.baseline.elev0
        self.baseline.prev_thr = self.baseline.thr0

        self.current_step = 0
        self.last_pid_action = 0.0
        self.prev_rl_res = 0.0
        self.target_altitude = float(env["alt_ft"])
        self.target_vc = float(env["vc_kts"])
        self.fault_trigger_time = float(self.np_random.uniform(*self.fault_time_range))
        self.fault_trigger_step = int(self.fault_trigger_time / self.dt)
        self.elevator_health = float(self.np_random.uniform(*self.fault_severity_range))
        self.fault_active = False
        self.detector_estimate = 1.0
        self._fault_onset_step = None

        return self._get_obs(), {
            "trim_alt_ft": self.target_altitude,
            "trim_vc_kts": self.target_vc,
            "curriculum_phase": self.curriculum_phase,
        }

    def _update_gamma_hat(self):
        """Approximate twin: lag + noise (not pure oracle)."""
        if not self.fault_active:
            self.detector_estimate = 1.0
            return
        if self._fault_onset_step is None:
            self._fault_onset_step = self.current_step
        lag_steps = max(1, int(self.gamma_lag_s / self.dt))
        alpha = min(1.0, (self.current_step - self._fault_onset_step) / lag_steps)
        g = 1.0 + alpha * (self.elevator_health - 1.0)
        if self.gamma_noise_std > 0:
            g += float(self.np_random.normal(0.0, self.gamma_noise_std))
        self.detector_estimate = float(np.clip(g, 0.05, 1.0))

    def step(self, action):
        self.current_step += 1

        if self.current_step >= self.fault_trigger_step:
            self.fault_active = True
        physical_gamma = self.elevator_health if self.fault_active else 1.0
        self._update_gamma_hat()
        gamma_hat = float(np.clip(self.detector_estimate, 0.05, 1.0))

        cmds = self.baseline.update(self.fdm, self.target_altitude, self.target_vc)
        elev_raw = float(cmds["elev"])
        self.last_pid_action = float(np.clip(elev_raw, -1.0, 1.0))

        # ---- residual path IDENTICAL to unified ----
        residual_enabled = self.fault_active and (gamma_hat < RESIDUAL_ENABLE_GAMMA)
        if residual_enabled:
            raw_a = float(np.clip(float(action[0]), -0.5, 0.5))
            deficit = float(np.clip(1.0 - gamma_hat, 0.0, 1.0))
            desired = raw_a * RESIDUAL_GAIN * max(deficit, 0.15)
            desired = float(np.clip(desired, -RESIDUAL_MAX, RESIDUAL_MAX))
            du = float(np.clip(desired - self.prev_rl_res, -RESIDUAL_RATE_MAX, RESIDUAL_RATE_MAX))
            rl_res = self.prev_rl_res + du
        else:
            rl_res = self.prev_rl_res * 0.85
        self.prev_rl_res = rl_res

        kappa = min(1.0 / gamma_hat, KAPPA_MAX) if self.fault_active else 1.0
        elev_comp = float(np.clip(elev_raw * kappa + rl_res, -1.0, 1.0))
        elev_out = float(np.clip(elev_comp * physical_gamma, -1.0, 1.0))

        set_elev(self.fdm, elev_out)
        set_pitch_trim(self.fdm, cmds["ptrim"])
        set_throttle(self.fdm, cmds["throttle"])
        self.fdm.set_property_value(PROP_AIL, cmds["ail"])
        self.fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(self.fdm, cmds["ptrim"], cmds["speedbrake"])
        self.fdm.run()

        st = flight_state(self.fdm)
        alt_error = self.target_altitude - st["h"]
        theta_rad = math.radians(st["theta"])
        q_rad_s = math.radians(st["q"])

        # Reward: prioritize altitude hold; penalize residual effort lightly
        reward = -(
            2.5 * (alt_error ** 2) / 10000.0
            + 2.0 * (theta_rad ** 2)
            + 1.0 * (q_rad_s ** 2)
            + 0.3 * (rl_res ** 2)
        )
        reward += 3.0

        terminated = False
        hard_floor = max(3000.0, 0.4 * self.target_altitude)
        if abs(theta_rad) > math.radians(45.0) or abs(alt_error) > 3000.0 or st["h"] < hard_floor:
            reward -= 400.0
            terminated = True
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "fault_active": self.fault_active,
            "elevator_health": physical_gamma,
            "gamma_hat": gamma_hat,
            "altitude": st["h"],
            "target_altitude": self.target_altitude,
            "rl_residual": rl_res,
            "kappa": kappa,
            "residual_enabled": residual_enabled,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        st = flight_state(self.fdm)
        # Match unified obs8:
        # [vc*1.68781, 0.0, q_rad, theta_rad, h, target_h, prev_elev, gamma_hat]
        return np.array([
            st["vc"] * 1.68781,
            0.0,
            math.radians(st["q"]),
            math.radians(st["theta"]),
            st["h"],
            self.target_altitude,
            self.last_pid_action,
            float(self.detector_estimate),
        ], dtype=np.float32)
