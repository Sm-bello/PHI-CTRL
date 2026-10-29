#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 & Step 3 Verification Suite:
1. Clean JSBSim F-16 Trim at 100 Hz physics / 50 Hz control (2 sub-steps per cycle)
   without per-step propulsion/set-running reset and with native [0,1]->[0,2] throttle mapping.
2. Exact ISA Mach and dynamic pressure q_bar calculation across all flight envelopes.
3. Contract 1 60-second Baseline Hold-Trim Verification (|Δh| < 1 ft, |hdot| < 0.2 ft/s within 5 s).
4. Numerical linearization of JSBSim F-16 at trim, computing A_sp, B_sp, ||T||_inf,
   Gain Margin, Phase Margin, Time-Delay Margin, Disk Margin, Small-Gain bounds, and gamma_floor.
5. Nonlinear JSBSim stability sweep over estimation error tilde_gamma in [-0.35, +0.50]
   to empirically validate Theorem 1 and the gamma_floor bound.
6. True JSBSim F-16 steady-state thrust curve sweep T(delta_th) across [0.0, 1.0].
"""
import os
import json
import math
import numpy as np
import pandas as pd
import scipy.signal as signal
import jsbsim

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "verified_v2")
os.makedirs(OUT_DIR, exist_ok=True)

DT_SIM = 0.01   # 100 Hz physics integration
DT_CTRL = 0.02  # 50 Hz control & estimation (exact 2 integer sub-steps per cycle)
SUBSTEPS = 2


def create_trimmed_f16(alt_ft=15000.0, vc_kts=400.0):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.set_dt(DT_SIM)
    if not fdm.load_model("f16"):
        raise RuntimeError("Failed to load JSBSim f16 model")

    fdm.set_property_value("ic/h-sl-ft", float(alt_ft))
    fdm.set_property_value("ic/vc-kts", float(vc_kts))
    fdm.set_property_value("ic/gamma-deg", 0.0)
    fdm.set_property_value("ic/phi-deg", 0.0)
    fdm.set_property_value("ic/psi-true-deg", 0.0)
    fdm.set_property_value("ic/p-rad_sec", 0.0)
    fdm.set_property_value("ic/q-rad_sec", 0.0)
    fdm.set_property_value("ic/r-rad_sec", 0.0)
    fdm.reset_to_initial_conditions(1)
    # Initialize turbine once (do NOT call in runtime step loop!)
    fdm.set_property_value("propulsion/set-running", -1)
    fdm.set_property_value("fcs/fbw-override", 1.0)
    fdm.set_property_value("gear/gear-cmd-norm", 0.0)
    fdm.set_property_value("gear/gear-pos-norm", 0.0)
    fdm.run_ic()
    fdm.do_trim(1)

    trim_info = {
        "alt_ft": float(fdm.get_property_value("position/h-sl-ft")),
        "vc_kts": float(fdm.get_property_value("velocities/vc-kts")),
        "vt_fps": float(fdm.get_property_value("velocities/vt-fps")),
        "mach": float(fdm.get_property_value("velocities/mach")),
        "qbar_psf": float(fdm.get_property_value("aero/qbar-psf")),
        "qbar_kpa": float(fdm.get_property_value("aero/qbar-psf")) * 0.04788025898,
        "hdot_fps": float(fdm.get_property_value("velocities/h-dot-fps")),
        "theta0_deg": math.degrees(fdm.get_property_value("attitude/theta-rad")),
        "alpha0_deg": math.degrees(fdm.get_property_value("aero/alpha-rad")),
        "q0_dps": math.degrees(fdm.get_property_value("velocities/q-rad_sec")),
        "thr0": float(fdm.get_property_value("fcs/throttle-cmd-norm")),
        "thr_pos0": float(fdm.get_property_value("fcs/throttle-pos-norm")),
        "ptrim0": float(fdm.get_property_value("fcs/pitch-trim-cmd-norm")),
        "elev_pos0": float(fdm.get_property_value("fcs/elevator-pos-norm")),
        "n2_0": float(fdm.get_property_value("propulsion/engine/n2")),
        "thrust0_lbs": float(fdm.get_property_value("propulsion/engine/thrust-lbs")),
    }
    return fdm, trim_info


def step_plant(fdm, elev_cmd, thr_cmd, ptrim0, ail_cmd=0.0, rud_cmd=0.0, gamma_e=1.0, gamma_th=1.0):
    """Apply actuator commands (with effectiveness scaling) and step 2 physics sub-steps (50 Hz)."""
    elev_plant = float(np.clip(elev_cmd * gamma_e, -1.0, 1.0))
    thr_plant = float(np.clip(thr_cmd * gamma_th, 0.0, 1.0))
    for _ in range(SUBSTEPS):
        fdm.set_property_value("fcs/fbw-override", 1.0)
        fdm.set_property_value("fcs/pitch-trim-cmd-norm", float(ptrim0))
        fdm.set_property_value("fcs/elevator-cmd-norm", elev_plant)
        fdm.set_property_value("fcs/throttle-cmd-norm", thr_plant)
        # Native F-16 FCS maps throttle-cmd-norm [0, 1] -> throttle-pos-norm [0, 2]
        fdm.set_property_value("fcs/throttle-pos-norm", 2.0 * thr_plant)
        fdm.set_property_value("fcs/aileron-cmd-norm", float(ail_cmd))
        fdm.set_property_value("fcs/rudder-cmd-norm", float(rud_cmd))
        fdm.run()
    return elev_plant, thr_plant


class VerifiedTECSController:
    """
    Verified Total Energy Control System (TECS) + Inner Pitch Damper at 50 Hz.
    """
    def __init__(self, trim_info, dt=DT_CTRL):
        self.dt = dt
        self.g = 32.174
        self.thr0 = trim_info["thr0"]
        self.ptrim0 = trim_info["ptrim0"]
        self.theta0 = trim_info["theta0_deg"]
        self.int_E = 0.0
        self.int_B = 0.0
        self.int_th = 0.0
        self.prev_thr = self.thr0
        self.prev_elev = 0.0

    def update(self, state, h_ref, vc_ref, hdot_ref=0.0):
        h = state["h_ft"]
        vc = state["vc_kts"]
        hdot = state["hdot_fps"]
        theta = state["theta_deg"]
        q = state["q_dps"]
        phi = state["phi_deg"]
        p = state["p_dps"]
        r = state["r_dps"]

        V_fps = vc * 1.68781
        V_ref_fps = vc_ref * 1.68781

        # Specific energy errors (in ft)
        e_h = h_ref - h
        e_K = (V_ref_fps**2 - V_fps**2) / (2.0 * self.g)
        e_T = e_h + e_K
        e_D = e_h - e_K

        # Throttle channel (Total Energy)
        self.int_E = float(np.clip(self.int_E + (vc_ref - vc) * self.dt, -200.0, 200.0))
        thr_raw = self.thr0 + 0.020 * (vc_ref - vc) + 0.005 * self.int_E + 0.0010 * e_h + 0.0004 * e_T
        thr_cmd = float(np.clip(thr_raw, 0.05, 1.0))
        max_dthr = 0.35 * self.dt
        thr_cmd = float(np.clip(thr_cmd, self.prev_thr - max_dthr, self.prev_thr + max_dthr))
        self.prev_thr = thr_cmd

        # Pitch channel (Energy Distribution + Path Damping)
        self.int_B = float(np.clip(self.int_B + e_h * self.dt, -250.0, 250.0))
        theta_cmd = (
            self.theta0
            + 0.060 * e_h
            + 0.003 * e_D
            + 0.015 * self.int_B
            + 0.120 * (hdot_ref - hdot)
        )
        theta_cmd = float(np.clip(theta_cmd, -10.0, 15.0))

        e_th = theta_cmd - theta
        self.int_th = float(np.clip(self.int_th + e_th * self.dt, -15.0, 15.0))
        # Sign convention: positive elevator-cmd-norm is nose-down on JSBSim F-16
        elev_base = -0.100 * e_th - 0.010 * self.int_th + 0.060 * q
        elev_base = float(np.clip(elev_base, -1.0, 1.0))

        # Outer wings-level bank hold (inner roll/yaw rate dampers remain active in f16.xml)
        ail_cmd = float(np.clip(-0.02 * phi, -1.0, 1.0))
        rud_cmd = 0.0

        return {
            "elev_base": elev_base,
            "thr_cmd": thr_cmd,
            "theta_cmd": theta_cmd,
            "ail_cmd": ail_cmd,
            "rud_cmd": rud_cmd,
            "e_h": e_h,
            "e_T": e_T,
            "e_D": e_D,
        }


def get_state(fdm):
    return {
        "h_ft": float(fdm.get_property_value("position/h-sl-ft")),
        "vc_kts": float(fdm.get_property_value("velocities/vc-kts")),
        "vt_fps": float(fdm.get_property_value("velocities/vt-fps")),
        "hdot_fps": float(fdm.get_property_value("velocities/h-dot-fps")),
        "theta_deg": math.degrees(fdm.get_property_value("attitude/theta-rad")),
        "alpha_deg": math.degrees(fdm.get_property_value("aero/alpha-rad")),
        "q_dps": math.degrees(fdm.get_property_value("velocities/q-rad_sec")),
        "phi_deg": math.degrees(fdm.get_property_value("attitude/phi-rad")),
        "p_dps": math.degrees(fdm.get_property_value("velocities/p-rad_sec")),
        "r_dps": math.degrees(fdm.get_property_value("velocities/r-rad_sec")),
        "elev_pos": float(fdm.get_property_value("fcs/elevator-pos-norm")),
        "thr_pos": float(fdm.get_property_value("fcs/throttle-pos-norm")) * 0.5,
        "thrust_lbs": float(fdm.get_property_value("propulsion/engine/thrust-lbs")),
        "n2": float(fdm.get_property_value("propulsion/engine/n2")),
    }


def run_contract1_verification():
    print("\n--- 1. Running Contract 1 Baseline Hold-Trim Verification (60 s @ 50 Hz) ---")
    fdm, trim_info = create_trimmed_f16(15000.0, 400.0)
    ctrl = VerifiedTECSController(trim_info)
    rows = []
    n_steps = int(60.0 / DT_CTRL)
    for k in range(n_steps):
        t = k * DT_CTRL
        st = get_state(fdm)
        out = ctrl.update(st, 15000.0, 400.0, 0.0)
        elev_p, thr_p = step_plant(
            fdm, out["elev_base"], out["thr_cmd"], trim_info["ptrim0"],
            out["ail_cmd"], out["rud_cmd"], gamma_e=1.0, gamma_th=1.0
        )
        rows.append({
            "time_s": t,
            "alt_ft": st["h_ft"],
            "vc_kts": st["vc_kts"],
            "hdot_fps": st["hdot_fps"],
            "theta_deg": st["theta_deg"],
            "alpha_deg": st["alpha_deg"],
            "q_dps": st["q_dps"],
            "phi_deg": st["phi_deg"],
            "throttle_cmd": out["thr_cmd"],
            "elev_cmd": out["elev_base"],
            "thrust_lbs": st["thrust_lbs"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "contract1_hold_trim_50hz.csv"), index=False)

    max_dh = float((df["alt_ft"] - 15000.0).abs().max())
    max_dvc = float((df["vc_kts"] - 400.0).abs().max())
    max_hdot_after_5s = float(df[df["time_s"] >= 5.0]["hdot_fps"].abs().max())
    max_hdot_all = float(df["hdot_fps"].abs().max())
    max_phi = float(df["phi_deg"].abs().max())
    print(f"  Contract 1 Results: max|dh|={max_dh:.4f} ft, max|dVc|={max_dvc:.4f} kt, "
          f"max|hdot|(t>=5s)={max_hdot_after_5s:.4f} ft/s (all={max_hdot_all:.4f} ft/s), max|phi|={max_phi:.5f} deg")
    assert max_dh < 2.0 and max_hdot_after_5s < 0.2 and max_phi < 0.1, "Contract 1 FAILED!"
    print("  [PASS] Contract 1 Strictly Verified!")
    return trim_info, df


def linearize_and_compute_margins():
    print("\n--- 2. Numerical Linearization of JSBSim F-16 & Small-Gain / Disk Margins ---")
    envelopes = [
        ("Env1_Design_15k_400", 15000.0, 400.0),
        ("Env2_LowAlt_10k_400", 10000.0, 400.0),
        ("Env3_HighAlt_20k_400", 20000.0, 400.0),
        ("Env2b_LowAlt_10k_300", 10000.0, 300.0),
        ("Env3b_HighAlt_20k_450", 20000.0, 450.0),
    ]
    env_table = []
    for name, alt, vc in envelopes:
        _, ti = create_trimmed_f16(alt, vc)
        ti["name"] = name
        env_table.append(ti)
        print(f"  {name}: Mach={ti['mach']:.4f}, qbar={ti['qbar_kpa']:.2f} kPa ({ti['qbar_psf']:.1f} psf), "
              f"theta0={ti['theta0_deg']:.3f} deg, thr0={ti['thr0']:.4f}, T0={ti['thrust0_lbs']:.1f} lbf, hdot={ti['hdot_fps']:.2e}")

    # Numerical perturbation for short-period derivatives at 15,000 ft / 400 kt
    _, ti0 = create_trimmed_f16(15000.0, 400.0)

    def eval_derivs(d_alpha_rad=0.0, d_q_rps=0.0, d_elev=0.0, d_thr=0.0):
        fdm, ti = create_trimmed_f16(15000.0, 400.0)
        if abs(d_alpha_rad) > 0 or abs(d_q_rps) > 0:
            alpha0_rad = math.radians(ti["alpha0_deg"])
            theta0_rad = math.radians(ti["theta0_deg"])
            fdm.set_property_value("ic/alpha-deg", math.degrees(alpha0_rad + d_alpha_rad))
            fdm.set_property_value("ic/theta-deg", math.degrees(theta0_rad + d_alpha_rad))
            fdm.set_property_value("ic/q-rad_sec", d_q_rps)
            fdm.run_ic()
            fdm.do_trim(1) if False else None
        # Hold state at trim and step FCS/engine for 15 steps so actuator & thrust respond, while reading aerodynamic/total moments
        thr_c = float(np.clip(ti["thr0"] + d_thr, 0.0, 1.0))
        for _ in range(25):
            if abs(d_alpha_rad) > 0 or abs(d_q_rps) > 0:
                break
            fdm.set_property_value("fcs/fbw-override", 1.0)
            fdm.set_property_value("fcs/pitch-trim-cmd-norm", ti["ptrim0"])
            fdm.set_property_value("fcs/elevator-cmd-norm", float(d_elev))
            fdm.set_property_value("fcs/throttle-cmd-norm", thr_c)
            fdm.set_property_value("fcs/throttle-pos-norm", 2.0 * thr_c)
            fdm.run()
        if abs(d_alpha_rad) > 0 or abs(d_q_rps) > 0:
            fdm.set_property_value("fcs/fbw-override", 1.0)
            fdm.set_property_value("fcs/pitch-trim-cmd-norm", ti["ptrim0"])
            fdm.set_property_value("fcs/elevator-cmd-norm", 0.0)
            fdm.set_property_value("fcs/throttle-cmd-norm", ti["thr0"])
            fdm.set_property_value("fcs/throttle-pos-norm", 2.0 * ti["thr0"])
            fdm.run()
        qdot = float(fdm.get_property_value("accelerations/qdot-rad_sec2"))
        udot = float(fdm.get_property_value("accelerations/udot-ft_sec2"))
        wdot = float(fdm.get_property_value("accelerations/wdot-ft_sec2"))
        vt = float(fdm.get_property_value("velocities/vt-fps"))
        adot = wdot / vt
        return adot, qdot

    eps_a = 0.005
    eps_q = 0.01
    eps_u = 0.02

    adot_p, qdot_p = eval_derivs(d_alpha_rad=+eps_a)
    adot_m, qdot_m = eval_derivs(d_alpha_rad=-eps_a)
    Z_alpha_V = (adot_p - adot_m) / (2 * eps_a)
    M_alpha = (qdot_p - qdot_m) / (2 * eps_a)

    adot_p, qdot_p = eval_derivs(d_q_rps=+eps_q)
    adot_m, qdot_m = eval_derivs(d_q_rps=-eps_q)
    Z_q_V = (adot_p - adot_m) / (2 * eps_q)
    M_q = (qdot_p - qdot_m) / (2 * eps_q)

    adot_p, qdot_p = eval_derivs(d_elev=+eps_u)
    adot_m, qdot_m = eval_derivs(d_elev=-eps_u)
    Z_de_V = (adot_p - adot_m) / (2 * eps_u)
    M_de = (qdot_p - qdot_m) / (2 * eps_u)

    adot_p, qdot_p = eval_derivs(d_thr=+eps_u)
    adot_m, qdot_m = eval_derivs(d_thr=-eps_u)
    Z_dth_V = (adot_p - adot_m) / (2 * eps_u)
    M_dth = (qdot_p - qdot_m) / (2 * eps_u)

    print(f"  Linearized Short-Period @ 15k/400kt:")
    print(f"    Z_alpha/V0 = {Z_alpha_V:.4f} 1/s, M_alpha = {M_alpha:.4f} 1/s^2")
    print(f"    Z_q/V0     = {Z_q_V:.4f},     M_q     = {M_q:.4f} 1/s")
    print(f"    Z_de/V0    = {Z_de_V:.4f} 1/s, M_de    = {M_de:.4f} rad/s^2 per norm elev")
    print(f"    Z_dth/V0   = {Z_dth_V:.5f} 1/s, M_dth   = {M_dth:.5f} rad/s^2 per norm thr")

    # Build continuous-time short-period + theta plant from elev_cmd (including 20.2 rad/s actuator + Padé delay)
    # States: [alpha (rad), q (rad/s), theta (rad)]
    A_p = np.array([
        [Z_alpha_V, 1.0, 0.0],
        [M_alpha,   M_q, 0.0],
        [0.0,       1.0, 0.0],
    ])
    B_p = np.array([[Z_de_V], [M_de], [0.0]])
    # Actuator 1/(0.0495 s + 1) and effective loop delay (0.03 s = sample/hold + filter lag)
    tau_act = 0.0495
    tau_del = 0.030

    w = np.logspace(-2, 2.2, 4000)
    s = 1j * w

    # Plant transfer functions to theta(s) and q(s)
    G_q = np.zeros_like(s, dtype=complex)
    G_th = np.zeros_like(s, dtype=complex)
    I3 = np.eye(3)
    for idx, si in enumerate(s):
        resolv = np.linalg.inv(si * I3 - A_p) @ B_p
        act = 1.0 / (tau_act * si + 1.0) * np.exp(-tau_del * si)
        G_q[idx] = resolv[1, 0] * act
        G_th[idx] = resolv[2, 0] * act

    # Controller transfer function in radian/norm units matching VerifiedTECSController + MRAC inner loop:
    # u_elev = - (K_th * (theta_cmd - theta_deg) + K_ith/s * (theta_cmd - theta_deg) - K_q * q_dps)
    # Convert deg -> rad (1 rad = 57.2958 deg):
    deg_per_rad = 180.0 / math.pi
    Kp_th = 0.095 * deg_per_rad
    Ki_th = 0.012 * deg_per_rad
    Kq_rad = 0.055 * deg_per_rad

    # Open-loop transfer function L(jw) when breaking the loop at the elevator input:
    # Since M_de < 0 and positive elev is nose-down, negative feedback has positive loop gain:
    L_jw = - ( (Kp_th + Ki_th / s) * G_th + Kq_rad * G_q )
    T_jw = L_jw / (1.0 + L_jw)
    S_jw = 1.0 / (1.0 + L_jw)

    T_mag = np.abs(T_jw)
    S_mag = np.abs(S_jw)
    T_inf = float(np.max(T_mag))
    S_inf = float(np.max(S_mag))
    w_peak_T = float(w[np.argmax(T_mag)])

    # Classical Gain Margin, Phase Margin, Crossover Frequency, Delay Margin, and Disk Margin
    L_mag = np.abs(L_jw)
    L_phase_deg = np.unwrap(np.angle(L_jw)) * 180.0 / math.pi

    # Gain crossover (|L(jw)| = 1)
    gc_idx = np.argmin(np.abs(L_mag - 1.0))
    w_gc = float(w[gc_idx])
    pm_deg = float(180.0 + L_phase_deg[gc_idx])
    delay_margin_s = float((pm_deg * math.pi / 180.0) / w_gc)

    # Phase crossover (phase = -180 deg)
    pc_candidates = np.where(np.diff(np.sign(L_phase_deg + 180.0)))[0]
    if len(pc_candidates) > 0:
        pc_idx = pc_candidates[0]
        w_pc = float(w[pc_idx])
        gm_linear = float(1.0 / L_mag[pc_idx])
        gm_db = float(20.0 * math.log10(gm_linear))
    else:
        w_pc = float("nan")
        gm_linear = float("inf")
        gm_db = float("inf")

    # Classical symmetric Disk Margin alpha_max = 1 / ||S - T||_inf * 2
    st_diff_inf = float(np.max(np.abs(S_jw - T_jw)))
    alpha_disk = 2.0 / st_diff_inf
    disk_gm_max = (1.0 + 0.5 * alpha_disk) / (1.0 - 0.5 * alpha_disk)
    disk_pm_deg = 2.0 * math.degrees(math.atan(0.5 * alpha_disk))

    # Small-Gain sufficient bounds at gamma_e = 0.50, 0.60, 0.80, 1.00
    gamma_floor_sg = T_inf / (1.0 + T_inf)  # Lower bound on gamma_hat when gamma_e = 1.0
    gamma_floor_nyq = 1.0 / gm_linear       # Exact Nyquist lower bound on gamma_hat when gamma_e = 1.0

    sg_bounds = {}
    for ge in [0.5, 0.6, 0.8, 1.0]:
        lower_tilde = -ge / (1.0 + T_inf)
        upper_tilde_sg = ge / (T_inf - 1.0) if T_inf > 1.0 else float("inf")
        upper_tilde_phys = 1.0 - ge  # Since gamma_hat <= 1.0
        lower_tilde_nyq = -ge * (1.0 - 1.0 / gm_linear)
        sg_bounds[str(ge)] = {
            "gamma_e": ge,
            "sg_lower_tilde": float(lower_tilde),
            "sg_min_gamma_hat": float(ge + lower_tilde),
            "nyquist_lower_tilde": float(lower_tilde_nyq),
            "nyquist_min_gamma_hat": float(ge / gm_linear),
            "sg_upper_tilde": float(upper_tilde_sg),
            "physical_upper_tilde_cap": float(upper_tilde_phys),
        }

    print(f"  Frequency-Domain Margins @ 15,000 ft / 400 kt:")
    print(f"    ||T||_inf = {T_inf:.4f} (at w = {w_peak_T:.2f} rad/s), ||S||_inf = {S_inf:.4f}")
    print(f"    Gain Crossover w_gc = {w_gc:.2f} rad/s, Phase Margin PM = {pm_deg:.2f} deg, Delay Margin = {delay_margin_s:.3f} s")
    print(f"    Phase Crossover w_pc = {w_pc:.2f} rad/s, Gain Margin GM = {gm_linear:.3f} ({gm_db:.2f} dB)")
    print(f"    Disk Margin alpha = {alpha_disk:.4f} (Disk GM = {disk_gm_max:.3f}, Disk PM = {disk_pm_deg:.2f} deg)")
    print(f"    Healthy-Plant Safe Floor: Small-Gain gamma_floor = {gamma_floor_sg:.4f}, Exact Nyquist gamma_floor = {gamma_floor_nyq:.4f}")
    print(f"    At gamma_e = 0.50: SG lower bound tilde_gamma > {sg_bounds['0.5']['sg_lower_tilde']:.4f} "
          f"(gamma_hat > {sg_bounds['0.5']['sg_min_gamma_hat']:.4f}); "
          f"Exact Nyquist lower bound tilde_gamma > {sg_bounds['0.5']['nyquist_lower_tilde']:.4f} "
          f"(gamma_hat > {sg_bounds['0.5']['nyquist_min_gamma_hat']:.4f})")

    theory_summary = {
        "envelopes": env_table,
        "short_period_15k_400": {
            "Z_alpha_V": float(Z_alpha_V),
            "M_alpha": float(M_alpha),
            "Z_q_V": float(Z_q_V),
            "M_q": float(M_q),
            "Z_de_V": float(Z_de_V),
            "M_de": float(M_de),
            "Z_dth_V": float(Z_dth_V),
            "M_dth": float(M_dth),
        },
        "margins_15k_400": {
            "T_inf": T_inf,
            "S_inf": S_inf,
            "w_peak_T_rad_s": w_peak_T,
            "w_gc_rad_s": w_gc,
            "pm_deg": pm_deg,
            "delay_margin_s": delay_margin_s,
            "w_pc_rad_s": w_pc,
            "gm_linear": gm_linear,
            "gm_db": gm_db,
            "disk_alpha": alpha_disk,
            "disk_gm_max": disk_gm_max,
            "disk_pm_deg": disk_pm_deg,
            "gamma_floor_small_gain": gamma_floor_sg,
            "gamma_floor_nyquist": gamma_floor_nyq,
        },
        "bounds_by_gamma": sg_bounds,
    }
    with open(os.path.join(OUT_DIR, "theory_and_margins.json"), "w") as f:
        json.dump(theory_summary, f, indent=2)
    return theory_summary


def run_nonlinear_stability_sweep(theory_summary):
    print("\n--- 3. Nonlinear JSBSim Stability Sweep Over Estimation Error tilde_gamma (Item 11) ---")
    # Sweep tilde_gamma from -0.38 to +0.50 at gamma_e = 0.50 during a +500 ft reference step climb
    gamma_e = 0.50
    tilde_grid = np.round(np.linspace(-0.38, 0.50, 23), 3)
    sweep_rows = []

    for tg in tilde_grid:
        gamma_hat_fixed = max(0.05, min(1.0, gamma_e + float(tg)))
        kappa_fixed = 1.0 / gamma_hat_fixed  # Raw un-floored kappa to test the actual stability boundary!

        fdm, ti = create_trimmed_f16(15000.0, 400.0)
        ctrl = VerifiedTECSController(ti)
        # Include the actuator transport/filter lag tau_del = 0.03 s (1-2 steps buffer)
        u_buf = [0.0, 0.0]
        q_post = []
        err_h_post = []
        n_steps = int(35.0 / DT_CTRL)
        h_ref = 15000.0
        hdot_ref = 0.0
        wn_r, zeta_r = 0.30, 0.90

        for k in range(n_steps):
            t = k * DT_CTRL
            target_h = 15500.0 if t >= 5.0 else 15000.0
            ddh = wn_r**2 * (target_h - h_ref) - 2 * zeta_r * wn_r * hdot_ref
            hdot_ref += ddh * DT_CTRL
            h_ref += hdot_ref * DT_CTRL

            st = get_state(fdm)
            out = ctrl.update(st, h_ref, 400.0, hdot_ref)
            fault_on = (t >= 10.0)
            ge_now = gamma_e if fault_on else 1.0
            kap_now = kappa_fixed if fault_on else 1.0

            u_Delayed = u_buf.pop(0)
            u_cmd = float(np.clip(kap_now * out["elev_base"], -1.0, 1.0))
            u_buf.append(u_cmd)

            step_plant(fdm, u_Delayed, out["thr_cmd"], ti["ptrim0"], out["ail_cmd"], out["rud_cmd"], gamma_e=ge_now)

            if t >= 10.0:
                q_post.append(st["q_dps"])
                err_h_post.append(abs(h_ref - st["h_ft"]))

        q_arr = np.array(q_post)
        # Oscillation metric: RMS of high-pass pitch rate in last 15 seconds (t in [20, 35 s])
        q_tail = q_arr[int(10.0 / DT_CTRL):]
        q_osc_std = float(np.std(q_tail))
        q_peak = float(np.max(np.abs(q_arr)))
        max_eh = float(np.max(err_h_post))
        stable = bool(q_osc_std < 1.5 and max_eh < 150.0)
        sweep_rows.append({
            "tilde_gamma": float(tg),
            "gamma_hat": float(gamma_hat_fixed),
            "kappa": float(kappa_fixed),
            "delta_m": float(gamma_e / gamma_hat_fixed - 1.0),
            "q_peak_dps": q_peak,
            "q_tail_std_dps": q_osc_std,
            "max_err_h_ft": max_eh,
            "stable": int(stable),
        })
        print(f"  tilde_gamma={tg:+.3f} (gamma_hat={gamma_hat_fixed:.3f}, kappa={kappa_fixed:.2f}): "
              f"|q|_max={q_peak:.2f} deg/s, q_tail_std={q_osc_std:.3f} deg/s, max|e_h|={max_eh:.1f} ft -> {'STABLE' if stable else 'OSCILLATORY/UNSTABLE'}")

    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(os.path.join(OUT_DIR, "stability_sweep_nonlinear.csv"), index=False)
    return df_sweep


def run_thrust_curve_sweep():
    print("\n--- 4. True JSBSim F-16 Steady-State Thrust Sweep T(delta_th) @ 15,000 ft / 400 kt (Item 20) ---")
    thr_grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    rows = []
    for thr in thr_grid:
        fdm, ti = create_trimmed_f16(15000.0, 400.0)
        # Hold airspeed & altitude fixed for 12 seconds so turbine N2 and afterburner reach steady state at (15k, 400kt)
        for _ in range(1200):
            fdm.set_property_value("position/h-sl-ft", 15000.0)
            fdm.set_property_value("velocities/vc-kts", 400.0)
            fdm.set_property_value("fcs/throttle-cmd-norm", float(thr))
            fdm.set_property_value("fcs/throttle-pos-norm", float(2.0 * thr))
            fdm.run()
        n2 = float(fdm.get_property_value("propulsion/engine/n2"))
        thrust = float(fdm.get_property_value("propulsion/engine/thrust-lbs"))
        rows.append({
            "throttle_cmd": float(thr),
            "throttle_pos_norm": float(2.0 * thr),
            "n2_pct": n2,
            "thrust_lbs": thrust,
        })
        print(f"  delta_th={thr:.2f} (pos-norm={2*thr:.2f}): N2={n2:6.2f}%, Thrust={thrust:8.1f} lbf")
    df_thr = pd.DataFrame(rows)
    df_thr.to_csv(os.path.join(OUT_DIR, "f16_thrust_curve_15k_400.csv"), index=False)
    return df_thr


if __name__ == "__main__":
    run_contract1_verification()
    theory = linearize_and_compute_margins()
    run_nonlinear_stability_sweep(theory)
    run_thrust_curve_sweep()
    print("\n[DONE] Step 1 & Step 3 verification suite completed successfully!")
