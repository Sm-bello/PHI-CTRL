import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import jsbsim

from controller.baseline_pid_jsbsim import BaselinePIDJSBSim
from fault_injection.injector import FaultInjector


PROP_ELEV_CMD = "fcs/elevator-cmd-norm"
PROP_THROTTLE = "fcs/throttle-cmd-norm"


def find_trim(fdm, dt, target_alt=2000.0, target_vc=80.0, target_gamma=0.0,
              trim_eval_t=2.0, cost_warn=5.0, verbose=False):
    """
    2-D grid trim search (throttle x elevator) for c172p steady level flight.

    target_vc defaults to 80 kts, NOT 100-120 kts. Empirically (see
    jsbsim_phi_ctrl_env exploration log), 100-120 kts never converges for
    this airframe/weight within a reasonable throttle/elevator range --
    cost stays >= ~9 and the optimum sits on the grid boundary, meaning the
    aircraft is trading airspeed for climb rather than truly trimming.
    80 kts (close to this airframe's best-glide/economy speed) converges
    cleanly: cost ~1.75, comfortably inside an interior (non-boundary)
    optimum. If you need a faster cruise condition, re-run this search at
    your target speed FIRST and check on_grid_edge / cost before trusting
    the result for anything downstream -- do not assume convergence.

    JSBSim's own console output (aircraft config dump, mass properties on
    every run_ic()) is extremely verbose across a 100+ candidate grid
    search; suppressed by default via `verbose=False`.
    """
    import io, contextlib
    ctx = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

    with ctx:
        fdm.set_property_value("ic/h-sl-ft", target_alt)
        fdm.set_property_value("ic/vc-kts", target_vc)
        fdm.set_property_value("ic/gamma-deg", target_gamma)
        fdm.set_property_value("ic/phi-deg", 0.0)
        fdm.set_property_value("ic/psi-true-deg", 0.0)
        fdm.run_ic()

        best_cost = 1e9
        best_thr, best_elev = 0.60, 0.0

        throttles = np.linspace(0.00, 0.50, 16)
        elevators = np.linspace(-0.05, 0.15, 16)
        n_eval = int(trim_eval_t / dt)

        for thr in throttles:
            for elev in elevators:
                fdm.reset_to_initial_conditions(True)
                fdm.set_property_value(PROP_THROTTLE, float(thr))
                fdm.set_property_value(PROP_ELEV_CMD, float(elev))
                for _ in range(n_eval):
                    fdm.run()

                hdot = abs(fdm.get_property_value("velocities/h-dot-fps"))
                udot = abs(fdm.get_property_value("accelerations/udot-ft_sec2"))
                wdot = abs(fdm.get_property_value("accelerations/wdot-ft_sec2"))
                q_dps = abs(math.degrees(fdm.get_property_value("velocities/q-rad_sec")))
                cost = hdot + 0.05 * udot + 0.05 * wdot + 2.0 * q_dps

                if cost < best_cost:
                    best_cost = cost
                    best_thr, best_elev = float(thr), float(elev)

    # thr==0.0 (idle) is a genuine physical floor, not an arbitrary search
    # cutoff -- being pinned there just means idle power is enough, which is
    # a legitimate trim, not evidence the search range was too narrow.
    thr_on_meaningful_edge = np.isclose(best_thr, throttles[-1]) or (
        np.isclose(best_thr, throttles[0]) and not np.isclose(throttles[0], 0.0)
    )
    on_edge = (thr_on_meaningful_edge
               or np.isclose(best_elev, elevators[0]) or np.isclose(best_elev, elevators[-1]))
    if best_cost > cost_warn or on_edge:
        raise RuntimeError(
            f"[TRIM] Did not converge: cost={best_cost:.2f} (want << {cost_warn}), "
            f"thr={best_thr:.3f}, elev={best_elev:.3f}, on_grid_edge={on_edge}. "
            f"Widen the search ranges further or lower target_vc."
        )
    return best_thr, best_elev


class JSBSimPhiCtrlEnv(gym.Env):
    """
    PHI-CTRL residual-learning env, JSBSim c172p 6-DOF plant (longitudinal
    control only: elevator + fixed trim throttle -- ailerons/rudder held
    level, matching the original phi_ctrl_env.py's longitudinal-only scope).

    Observation (8,), SAME units/order as the original linear-plant env so a
    checkpoint trained there is at least architecturally comparable:
        [u (ft/s), w (ft/s), q (rad/s), theta (rad), h (ft),
         target_h (ft), last_pid_action (normalized elev), fault_estimate]

    Action (1,): additive residual on the normalized elevator command,
    applied strictly during the active fault window (safety shield, same
    policy as the original env). Bound set to [-0.5, 0.5] to match the
    checkpoint's ACTUAL trained action_space (confirmed via
    validate_checkpoint_native.py) -- NOT the [-0.3, 0.3] currently declared
    in gym_env/phi_ctrl_env.py, which does not match the loaded model and
    should be reconciled before relying on that file again.
    """
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self, max_episode_time_s=25.0, fault_time_range=(5.0, 8.0),
                 fault_severity_range=(0.3, 0.7), trim_target_vc=80.0,
                 trim_target_alt=2000.0, aircraft="c172p"):
        super().__init__()
        self.dt = 1.0 / 120.0
        self.max_steps = int(max_episode_time_s / self.dt)
        self.fault_time_range = fault_time_range
        self.fault_severity_range = fault_severity_range
        self.aircraft = aircraft
        self.trim_target_vc = trim_target_vc
        self.trim_target_alt = trim_target_alt

        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.pid = BaselinePIDJSBSim(dt=self.dt)
        self.injector = FaultInjector(nominal_B=np.eye(1))  # only used for its API shape here

        self.fdm = None
        self.trim_thr = None
        self.trim_elev = None
        self._build_fdm()

    def _build_fdm(self):
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_dt(self.dt)
        if not self.fdm.load_model(self.aircraft):
            raise RuntimeError(f"Failed to load aircraft: {self.aircraft}")
        self.fdm.disable_output()
        self.trim_thr, self.trim_elev = find_trim(
            self.fdm, self.dt, target_alt=self.trim_target_alt, target_vc=self.trim_target_vc
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.pid.reset()
        self.last_pid_action = 0.0

        self.fdm.set_property_value("ic/h-sl-ft", self.trim_target_alt)
        self.fdm.set_property_value("ic/vc-kts", self.trim_target_vc)
        self.fdm.set_property_value("ic/gamma-deg", 0.0)
        self.fdm.set_property_value("ic/phi-deg", 0.0)
        self.fdm.set_property_value("ic/psi-true-deg", 0.0)
        self.fdm.set_property_value(PROP_THROTTLE, self.trim_thr)
        self.fdm.set_property_value(PROP_ELEV_CMD, self.trim_elev)
        self.fdm.run_ic()

        for _ in range(int(3.0 / self.dt)):  # brief open-loop settle at trim
            self.fdm.set_property_value(PROP_THROTTLE, self.trim_thr)
            self.fdm.set_property_value(PROP_ELEV_CMD, self.trim_elev)
            self.fdm.run()

        self.h0 = self.fdm.get_property_value("position/h-sl-ft")
        self.target_altitude = self.h0 + self.np_random.choice([0.0, 30.0, -30.0])

        self.fault_trigger_time = self.np_random.uniform(*self.fault_time_range)
        self.fault_trigger_step = int(self.fault_trigger_time / self.dt)
        self.elevator_health = self.np_random.uniform(*self.fault_severity_range)
        self.fault_active = False
        self.detector_estimate = 1.0

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        t = self.current_step * self.dt

        if self.current_step >= self.fault_trigger_step:
            self.fault_active = True
            self.detector_estimate = self.elevator_health
        eff_gamma = self.elevator_health if self.fault_active else 1.0

        h = self.fdm.get_property_value("position/h-sl-ft")
        hdot = self.fdm.get_property_value("velocities/h-dot-fps")
        theta_deg = math.degrees(self.fdm.get_property_value("attitude/theta-rad"))
        q_dps = math.degrees(self.fdm.get_property_value("velocities/q-rad_sec"))

        pid_elev_raw, _ = self.pid.compute_action(
            {"h_ft": h, "hdot_fps": hdot, "theta_deg": theta_deg, "q_dps": q_dps},
            target_h=self.target_altitude,
        )
        self.last_pid_action = float(np.clip(pid_elev_raw, -1.0, 1.0))

        # --- Fault representation + residual, single clean design ---
        # The fault attenuates whatever command reaches the elevator surface
        # (actuator effectiveness loss), exactly mirroring injector.py's
        # B_eff scaling and the classical harness's proven approach. The
        # residual is added to the RAW (unclipped) pid command BEFORE this
        # attenuation, so the agent can push the pre-attenuation command
        # above nominal to recover authority -- up to the actuator's true
        # physical ceiling of +/-eff_gamma once clipped. Clip happens
        # exactly ONCE, after both the residual and the attenuation are
        # applied -- re-introducing a clip-then-scale-then-clip sequence
        # here would silently zero out the residual's effect, which is
        # the same double-saturation bug fixed earlier in
        # run_jsbsim_layer3_fixed_v3.py. Do not add a second clip above.
        residual = float(action[0]) if self.fault_active else 0.0
        combined_raw = pid_elev_raw + residual
        elev_out = np.clip(combined_raw * eff_gamma, -1.0, 1.0)

        self.fdm.set_property_value(PROP_ELEV_CMD, elev_out)
        self.fdm.set_property_value(PROP_THROTTLE, self.trim_thr)
        # Lateral surfaces held at trim (0) -- longitudinal-only, matching
        # the original env's scope.
        self.fdm.set_property_value("fcs/aileron-cmd-norm", 0.0)
        self.fdm.set_property_value("fcs/rudder-cmd-norm", 0.0)

        self.fdm.run()

        h_new = self.fdm.get_property_value("position/h-sl-ft")
        theta_new = self.fdm.get_property_value("attitude/theta-rad")
        q_new = self.fdm.get_property_value("velocities/q-rad_sec")
        u_new = self.fdm.get_property_value("velocities/u-fps")
        w_new = self.fdm.get_property_value("velocities/w-fps")

        alt_error = self.target_altitude - h_new
        reward = -(
            2.5 * (alt_error ** 2) / 100.0
            + 2.0 * (theta_new ** 2)
            + 1.0 * (q_new ** 2)
            + 0.5 * (residual ** 2)
        )
        reward += 3.0

        terminated = False
        if abs(theta_new) > math.radians(45.0) or abs(alt_error) > 500.0:
            reward -= 400.0
            terminated = True
        truncated = bool(self.current_step >= self.max_steps)

        self._u, self._w, self._q, self._theta, self._h = u_new, w_new, q_new, theta_new, h_new

        info = {
            "fault_active": self.fault_active,
            "elevator_health": eff_gamma,
            "altitude": h_new,
            "target_altitude": self.target_altitude,
            "residual_correction": residual,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        u = self.fdm.get_property_value("velocities/u-fps")
        w = self.fdm.get_property_value("velocities/w-fps")
        q = self.fdm.get_property_value("velocities/q-rad_sec")
        theta = self.fdm.get_property_value("attitude/theta-rad")
        h = self.fdm.get_property_value("position/h-sl-ft")
        return np.array([
            u, w, q, theta, h, self.target_altitude,
            self.last_pid_action, self.detector_estimate
        ], dtype=np.float32)
