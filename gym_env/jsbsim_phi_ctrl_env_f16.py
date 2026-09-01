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


# Default envelope grid for generalization training.
# Solved once at env construction; reset() only indexes into the cache.
DEFAULT_TRIM_GRID = [
    {"alt_ft": 10000.0, "vc_kts": 350.0, "theta_seed": 2.5},
    {"alt_ft": 10000.0, "vc_kts": 400.0, "theta_seed": 2.5},
    {"alt_ft": 10000.0, "vc_kts": 450.0, "theta_seed": 2.0},
    {"alt_ft": 15000.0, "vc_kts": 350.0, "theta_seed": 2.5},
    {"alt_ft": 15000.0, "vc_kts": 400.0, "theta_seed": 2.5},  # original point
    {"alt_ft": 15000.0, "vc_kts": 450.0, "theta_seed": 2.0},
    {"alt_ft": 20000.0, "vc_kts": 350.0, "theta_seed": 3.0},
    {"alt_ft": 20000.0, "vc_kts": 400.0, "theta_seed": 2.5},
    {"alt_ft": 20000.0, "vc_kts": 450.0, "theta_seed": 2.0},
]


class JSBSimF16PhiCtrlEnv(gym.Env):
    """
    PHI-CTRL residual-learning env on the REAL F-16 JSBSim plant.

    Observation (8,):
        [u (ft/s), w (ft/s), q (rad/s), theta (rad), h (ft),
         target_h (ft), last_pid_action (normalized elev), fault_estimate]

    Action (1,): additive residual on raw elevator, applied only during the
    active fault window, clipped with the baseline command once. Bound [-0.5, 0.5].

    Trim strategy (generalization-ready):
      - At construction, native_trim() is solved once for each point in a
        small altitude×airspeed grid and cached.
      - reset() randomly picks one cached trim (near-zero cost).
      - Curriculum: early training can restrict sampling to the original
        15k/400 point, then widen to the full grid (see set_curriculum_phase).

    This preserves the training-speed win (no per-episode trim solve) while
    enabling a policy that generalizes across flight conditions.
    """
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
    ):
        super().__init__()
        self.dt = DT
        self.max_steps = int(max_episode_time_s / self.dt)
        self.fault_time_range = fault_time_range
        self.fault_severity_range = fault_severity_range
        self.settle_time_s = settle_time_s
        self.randomize_trim = randomize_trim
        # 0.0 → only original point; 1.0 → full grid uniform
        self.curriculum_phase = float(np.clip(curriculum_phase, 0.0, 1.0))

        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self.fdm = None
        # list of dicts: {env, thr, elev, ptrim, theta, ok}
        self._trim_cache = []
        self._active_trim = None
        self._build_fdm_and_trim_grid(trim_grid or DEFAULT_TRIM_GRID)

        self.baseline = None
        self.last_pid_action = 0.0

    # ------------------------------------------------------------------
    # Construction: pre-compute trim grid
    # ------------------------------------------------------------------
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
                "desc": f"grid[{i}] {point['alt_ft']:.0f}ft/{point['vc_kts']:.0f}kts",
            }
            ok, thr, elev, ptrim, theta = native_trim(self.fdm, env)
            entry = {
                "env": env,
                "thr": thr,
                "elev": elev,
                "ptrim": ptrim,
                "theta": theta,
                "ok": bool(ok),
            }
            if ok:
                accepted.append(entry)
                print(
                    f"[ENV]   grid[{i}] OK  h={env['alt_ft']:.0f} Vc={env['vc_kts']:.0f} "
                    f"thr={thr:.3f} elev={elev:+.4f} θ={theta:+.2f}"
                )
            else:
                print(
                    f"[ENV]   grid[{i}] FAIL h={env['alt_ft']:.0f} Vc={env['vc_kts']:.0f} "
                    f"— skipped"
                )

        if not accepted:
            # Absolute fallback: original single-point ENV
            print("[ENV] WARNING: entire grid failed; falling back to default ENV")
            ok, thr, elev, ptrim, theta = native_trim(self.fdm, ENV)
            if not ok:
                raise RuntimeError("F16 trim failed to converge — cannot build training env")
            accepted = [{
                "env": dict(ENV),
                "thr": thr,
                "elev": elev,
                "ptrim": ptrim,
                "theta": theta,
                "ok": True,
            }]

        self._trim_cache = accepted
        # Prefer the original 15k/400 as index 0 when present
        self._nominal_idx = 0
        for i, e in enumerate(self._trim_cache):
            if (
                abs(e["env"]["alt_ft"] - 15000.0) < 1.0
                and abs(e["env"]["vc_kts"] - 400.0) < 1.0
            ):
                self._nominal_idx = i
                break

        print(
            f"[ENV] Trim cache ready: {len(self._trim_cache)} points "
            f"(nominal idx={self._nominal_idx}). "
            f"reset() samples from cache — no per-episode trim solve."
        )

    # ------------------------------------------------------------------
    # Curriculum control (called from training callback / outside)
    # ------------------------------------------------------------------
    def set_curriculum_phase(self, phase: float):
        """
        phase ∈ [0, 1]:
          0.0 → always sample the nominal 15k/400 trim
          1.0 → uniform over the full accepted grid
        Intermediate values mix: with probability `phase` sample the full
        grid, otherwise stick to nominal. Simple and effective.
        """
        self.curriculum_phase = float(np.clip(phase, 0.0, 1.0))

    def _sample_trim_entry(self):
        if not self.randomize_trim or len(self._trim_cache) == 1:
            return self._trim_cache[self._nominal_idx]

        # Curriculum: mostly nominal early, full grid later
        if self.np_random.random() > self.curriculum_phase:
            return self._trim_cache[self._nominal_idx]
        idx = int(self.np_random.integers(0, len(self._trim_cache)))
        return self._trim_cache[idx]

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
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
        self.target_altitude = float(env["alt_ft"])
        self.target_vc = float(env["vc_kts"])
        self.fault_trigger_time = float(self.np_random.uniform(*self.fault_time_range))
        self.fault_trigger_step = int(self.fault_trigger_time / self.dt)
        self.elevator_health = float(self.np_random.uniform(*self.fault_severity_range))
        self.fault_active = False
        self.detector_estimate = 1.0

        return self._get_obs(), {
            "trim_alt_ft": self.target_altitude,
            "trim_vc_kts": self.target_vc,
            "curriculum_phase": self.curriculum_phase,
        }

    def step(self, action):
        self.current_step += 1
        t = self.current_step * self.dt

        if self.current_step >= self.fault_trigger_step:
            self.fault_active = True
            self.detector_estimate = self.elevator_health
        physical_gamma = self.elevator_health if self.fault_active else 1.0

        cmds = self.baseline.update(self.fdm, self.target_altitude, self.target_vc)
        elev_raw = cmds["elev"]
        self.last_pid_action = float(np.clip(elev_raw, -1.0, 1.0))

        # Match unified: residual is bounded add-on AFTER 1/γ compensation
        residual = float(action[0]) if self.fault_active else 0.0
        residual = float(np.clip(residual, -0.2, 0.2))
        gamma_hat = float(np.clip(self.detector_estimate, 0.05, 1.0))
        comp = min(1.0 / gamma_hat, 4.0) if self.fault_active else 1.0
        deficit = max(1.0 - gamma_hat, 0.15) if self.fault_active else 0.0
        elev_comp = float(np.clip(elev_raw * comp + residual * deficit * 0.35, -1.0, 1.0))
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

        reward = -(
            2.5 * (alt_error ** 2) / 10000.0
            + 2.0 * (theta_rad ** 2)
            + 1.0 * (q_rad_s ** 2)
            + 0.5 * (residual ** 2)
        )
        reward += 3.0

        terminated = False
        # Floor scales with target so high-altitude points are not unfairly terminated
        hard_floor = max(3000.0, 0.4 * self.target_altitude)
        if abs(theta_rad) > math.radians(45.0) or abs(alt_error) > 3000.0 or st["h"] < hard_floor:
            reward -= 400.0
            terminated = True
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "fault_active": self.fault_active,
            "elevator_health": physical_gamma,
            "altitude": st["h"],
            "target_altitude": self.target_altitude,
            "target_vc": self.target_vc,
            "residual_correction": residual,
            "curriculum_phase": self.curriculum_phase,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        st = flight_state(self.fdm)
        u = self.fdm.get_property_value("velocities/u-fps")
        w = self.fdm.get_property_value("velocities/w-fps")
        q = math.radians(st["q"])
        theta = math.radians(st["theta"])
        return np.array([
            u, w, q, theta, st["h"], self.target_altitude,
            self.last_pid_action, self.detector_estimate,
        ], dtype=np.float32)
