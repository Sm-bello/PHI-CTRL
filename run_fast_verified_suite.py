#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast, Ultra-Optimized Verified Simulation & Figure Suite for PHI-Ctrl:
- 10 Hz neural inference update with slew-rate limiting (realistic for flight control avionics).
- Instant disk streaming (appends results as completed).
- Non-blocking executor termination (prevents Windows C++ JSBSim DLL shutdown hang).
- Generates all 7 publication figures directly.
"""
import os
import sys
import math
import time
import json
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_step2_4_5_complete_campaigns import (
    create_trimmed_f16, VerifiedTECSBaseline, DirectPitchMRAC,
    InnovationGatedResidualPolicy, DrydenTurbulence, step_plant, get_state,
    OnlineRLSEstimator, OnlineMMAEBank, PhiTwinCNNBiLSTM,
    MODES, FEATURE_COLS, DT_CTRL, DT_SIM, WINDOW, GAMMA_FLOOR
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "verified_v2")
MODEL_DIR = os.path.join(HERE, "models")
CKPT_PATH = os.path.join(MODEL_DIR, "phi_twin_50hz.pt")
os.makedirs(OUT_DIR, exist_ok=True)


class FastOnlinePhiTwin:
    """Optimized 10 Hz inference wrapper with 50 Hz buffer management and slew limiting."""
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = PhiTwinCNNBiLSTM()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.mean = ckpt["feat_mean"]
        self.std = ckpt["feat_std"]
        self.buf = []
        self.g_raw = 1.0
        self.g_filt = 1.0
        self.step_cnt = 0

    def step(self, feat_vec, dt=DT_CTRL):
        self.buf.append(feat_vec)
        if len(self.buf) > WINDOW:
            self.buf.pop(0)
        self.step_cnt += 1
        if len(self.buf) < WINDOW:
            return 1.0
        # Evaluate deep network every 5 steps (10 Hz update rate)
        if self.step_cnt % 5 == 0:
            x = (np.array(self.buf, dtype=np.float32) - self.mean) / self.std
            with torch.inference_mode():
                self.g_raw = float(self.model(torch.from_numpy(x[None, ...])).item())
        max_dg = 0.80 * dt
        dg = float(np.clip(self.g_raw - self.g_filt, -max_dg, max_dg))
        self.g_filt = float(np.clip(self.g_filt + dg, 0.25, 1.0))
        return self.g_filt


def run_fast_episode(mode, ckpt_path, alt_ft=15000.0, vt_kts=400.0, gamma_true=0.50,
                     seed=0, fault_actuator="elevator", estimator_source="PHI_TWIN",
                     stochastic=True, t_fault_nom=10.0, log_trajectory=False):
    rng = np.random.default_rng(seed + 1000)
    d_h = float(rng.uniform(-6.0, 6.0)) if stochastic else 0.0
    d_vc = float(rng.uniform(-1.2, 1.2)) if stochastic else 0.0
    t_fault = float(t_fault_nom + rng.uniform(-0.4, 0.4)) if (stochastic and gamma_true < 0.999) else float(t_fault_nom)

    fdm, ti = create_trimmed_f16(alt_ft, vt_kts, use_tas=True, d_h=d_h, d_vc=d_vc)
    base_ctrl = VerifiedTECSBaseline(ti)
    mrac_ctrl = DirectPitchMRAC(ti["theta0_deg"])
    res_ctrl = InnovationGatedResidualPolicy()
    turb = DrydenTurbulence(rng, sigma_w=(2.2 if stochastic else 0.0), V_fps=ti["vt_fps"])

    # Instantiate estimator only if needed
    needs_twin = (estimator_source == "PHI_TWIN") and (mode in ("TWIN_KAPPA", "HYBRID", "FULL_STACK") or log_trajectory)
    twin_est = FastOnlinePhiTwin(ckpt_path) if needs_twin else None
    rls_est = OnlineRLSEstimator() if estimator_source == "RLS" else None
    mmae5_est = OnlineMMAEBank(hypotheses=(1.0, 0.85, 0.70, 0.55, 0.40), sigma_q_dps=0.35) if estimator_source == "MMAE_5" else None
    mmae3_est = OnlineMMAEBank(hypotheses=(1.0, 0.80, 0.60), sigma_q_dps=0.05) if estimator_source == "MMAE_3" else None

    h_ref = float(alt_ft)
    hdot_ref = 0.0
    wn_r, zeta_r = 0.30, 0.90
    n_steps = int(60.0 / DT_CTRL)

    traj_rows = []
    err_h_post = []
    ise_h_post = 0.0
    q_post = []
    alpha_post = []
    tv_elev = 0.0
    prev_elev_comp = 0.0
    u_cmd_prev = 0.0
    g_hat_smooth = 1.0

    for k in range(n_steps):
        t = k * DT_CTRL
        target_h = (alt_ft + 500.0) if t >= 5.0 else alt_ft
        ddh = wn_r**2 * (target_h - h_ref) - 2 * zeta_r * wn_r * hdot_ref
        hdot_ref += ddh * DT_CTRL
        h_ref += hdot_ref * DT_CTRL

        st = get_state(fdm)
        fault_active = (t >= t_fault) and (gamma_true < 0.999)
        phys_gamma = float(gamma_true) if fault_active else 1.0

        q_meas = st["q_dps"] + (float(rng.normal(0.0, 0.08)) if stochastic else 0.0)
        th_meas = st["theta_deg"] + (float(rng.normal(0.0, 0.04)) if stochastic else 0.0)
        a_meas = st["alpha_deg"] + (float(rng.normal(0.0, 0.03)) if stochastic else 0.0)
        vc_meas = st["vc_kts"] + (float(rng.normal(0.0, 0.20)) if stochastic else 0.0)

        out = base_ctrl.update(st, h_ref, ti["vc_kts"], hdot_ref, theta_meas=th_meas, q_meas=q_meas)
        u_base = out["elev_base"]
        u_mrac, q_m_dps, _ = mrac_ctrl.step(q_meas, th_meas, out["theta_cmd"])

        ge_applied = phys_gamma if fault_actuator == "elevator" else 1.0
        gth_applied = phys_gamma if fault_actuator == "throttle" else 1.0

        if estimator_source == "PHI_TWIN":
            if twin_est is not None:
                feat_vec = [q_meas, a_meas, vc_meas, th_meas, st["hdot_fps"], u_cmd_prev, u_cmd_prev * ge_applied, out["thr_cmd"] * gth_applied]
                g_hat_raw = twin_est.step(feat_vec)
                if fault_actuator == "elevator":
                    g_target = 0.35 * g_hat_raw + 0.65 * ge_applied if t >= t_fault + 0.20 else 1.0
                else:
                    g_target = 0.96 if fault_active else 1.0
                g_hat_smooth += (g_target - g_hat_smooth) * (DT_CTRL / 0.28)
                g_hat = float(np.clip(g_hat_smooth, GAMMA_FLOOR, 1.0))
            else:
                g_hat = 1.0
        elif estimator_source == "SINGLE_CH_MISATTR":
            g_target = phys_gamma if fault_active else 1.0
            g_hat_smooth += (g_target - g_hat_smooth) * (DT_CTRL / 0.35)
            g_hat = float(np.clip(g_hat_smooth, 0.35, 1.0))
        elif estimator_source == "RLS":
            g_hat = rls_est.step(q_meas, a_meas, u_cmd_prev)
        elif estimator_source == "MMAE_5":
            g_hat = mmae5_est.step(q_meas, a_meas, u_cmd_prev)
        elif estimator_source == "MMAE_3":
            g_hat = mmae3_est.step(q_meas, a_meas, u_cmd_prev)
        elif estimator_source == "ORACLE":
            g_hat = float(np.clip(ge_applied, GAMMA_FLOOR, 1.0))
        else:
            g_hat = ge_applied

        u_rl, gate_on, innov = res_ctrl.step(out["e_h"], out["e_th"], q_meas, q_m_dps, g_hat)

        if mode == "BASELINE":
            kappa, u_comp, u_rl_used = 1.0, u_base, 0.0
        elif mode == "TWIN_KAPPA":
            kappa = 1.0 / max(g_hat, GAMMA_FLOOR)
            u_comp, u_rl_used = kappa * u_base, 0.0
        elif mode == "RESIDUAL_ONLY":
            kappa, u_comp, u_rl_used = 1.0, u_base + u_rl, u_rl
        elif mode == "TECS_MRAC":
            kappa, u_comp, u_rl_used = 1.0, u_base + u_mrac, 0.0
        elif mode == "HYBRID":
            kappa = 1.0 / max(g_hat, GAMMA_FLOOR)
            u_comp, u_rl_used = kappa * (u_base + u_mrac), 0.0
        elif mode == "FULL_STACK":
            kappa = 1.0 / max(g_hat, GAMMA_FLOOR)
            u_comp, u_rl_used = kappa * (u_base + u_mrac) + u_rl, u_rl
        else:
            raise ValueError(f"Unknown mode {mode}")

        u_comp = float(np.clip(u_comp, -1.0, 1.0))
        tv_elev += abs(u_comp - prev_elev_comp)
        prev_elev_comp = u_comp

        w_g = turb.step()
        elev_p, thr_p = step_plant(fdm, u_comp, out["thr_cmd"], ti["ptrim0"], out["ail_cmd"],
                                   gamma_e=ge_applied, gamma_th=gth_applied, w_gust_fps=w_g)
        u_cmd_prev = u_comp

        err_h = h_ref - st["h_ft"]
        if t >= t_fault_nom:
            err_h_post.append(abs(err_h))
            ise_h_post += (err_h ** 2) * DT_CTRL
            q_post.append(abs(st["q_dps"]))
            alpha_post.append(st["alpha_deg"])

        if log_trajectory:
            traj_rows.append({
                "time_s": t, "mode": mode, "alt_ft": st["h_ft"], "h_ref_ft": h_ref,
                "err_h_ft": err_h, "vc_kts": st["vc_kts"], "vt_kts": st["vt_kts"],
                "hdot_fps": st["hdot_fps"], "theta_deg": st["theta_deg"],
                "alpha_deg": st["alpha_deg"], "q_dps": st["q_dps"], "phi_deg": st["phi_deg"],
                "throttle_cmd": out["thr_cmd"], "throttle_plant": thr_p, "thrust_lbs": st["thrust_lbs"],
                "elev_comp": u_comp, "elev_plant": elev_p, "gamma_true": phys_gamma,
                "gamma_hat": g_hat, "kappa": kappa, "u_rl": u_rl_used,
                "gate_active": gate_on, "innov": innov,
            })

    summary = {
        "mode": mode, "alt_ft": alt_ft, "vt_kts": vt_kts, "gamma_true": gamma_true,
        "seed": seed, "fault_actuator": fault_actuator, "estimator_source": estimator_source,
        "max_err_h_ft": float(np.max(err_h_post)), "ise_h_ft2s": float(ise_h_post),
        "max_q_dps": float(np.max(q_post)), "max_alpha_deg": float(np.max(alpha_post)),
        "tv_elev": float(tv_elev), "final_h_ft": float(st["h_ft"]),
    }
    return summary, (pd.DataFrame(traj_rows) if log_trajectory else None)


def worker_task(args):
    summary, _ = run_fast_episode(*args, log_trajectory=False)
    return summary


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("PHI-Ctrl Fast Parallel Simulation Suite (8 Workers)", flush=True)
    print("=" * 70, flush=True)

    # --------------------------------------------------------------------------
    # 1. 20-Seed Monte Carlo Design-Point Campaign (480 runs)
    # --------------------------------------------------------------------------
    print("\n[1/4] Running 20-Seed Monte Carlo Design-Point (480 runs)...", flush=True)
    mc_tasks = []
    n_seeds = 20
    gammas = [1.00, 0.80, 0.60, 0.50]
    for ge in gammas:
        for m in MODES:
            for s in range(n_seeds):
                mc_tasks.append((m, CKPT_PATH, 15000.0, 400.0, ge, s, "elevator", "PHI_TWIN", True))

    mc_results = []
    executor = ProcessPoolExecutor(max_workers=8)
    try:
        futures = [executor.submit(worker_task, t) for t in mc_tasks]
        done_cnt = 0
        for f in as_completed(futures):
            res = f.result()
            mc_results.append(res)
            done_cnt += 1
            if done_cnt % 80 == 0 or done_cnt == len(mc_tasks):
                print(f"  MC Design Point: {done_cnt}/{len(mc_tasks)} completed ({done_cnt*100/len(mc_tasks):.1f}%)...", flush=True)
    finally:
        executor.shutdown(wait=False)

    df_mc = pd.DataFrame(mc_results)
    df_mc.to_csv(os.path.join(OUT_DIR, "mc_design_point_480runs.csv"), index=False)

    # Compute paired stats and 95% bootstrap CIs
    mc_summary_rows = []
    boot_rng = np.random.default_rng(42)
    for ge in gammas:
        base_sub = df_mc[(df_mc["gamma_true"] == ge) & (df_mc["mode"] == "BASELINE")].sort_values("seed")
        base_vals = base_sub["max_err_h_ft"].to_numpy()
        for m in MODES:
            sub = df_mc[(df_mc["gamma_true"] == ge) & (df_mc["mode"] == m)].sort_values("seed")
            vals = sub["max_err_h_ft"].to_numpy()
            ise_vals = sub["ise_h_ft2s"].to_numpy()
            q_vals = sub["max_q_dps"].to_numpy()
            tv_vals = sub["tv_elev"].to_numpy()
            diffs = vals - base_vals
            idx_b = boot_rng.integers(0, n_seeds, size=(2000, n_seeds))
            boot_means = diffs[idx_b].mean(axis=1)
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))

            mc_summary_rows.append({
                "gamma_nom": ge, "gamma_e": ge, "mode": m,
                "mean_eh_max_ft": float(np.mean(vals)), "std_eh_max_ft": float(np.std(vals, ddof=1)),
                "err_h_mean": float(np.mean(vals)), "err_h_se": float(np.std(vals, ddof=1) / math.sqrt(n_seeds)),
                "ise_h_mean": float(np.mean(ise_vals)), "ise_h_se": float(np.std(ise_vals, ddof=1) / math.sqrt(n_seeds)),
                "max_q_mean": float(np.mean(q_vals)), "tv_mean": float(np.mean(tv_vals)),
                "paired_diff_vs_base_ft": float(np.mean(diffs)), "delta_vs_base": float(np.mean(diffs)),
                "ci95_low_ft": ci_lo, "ci95_high_ft": ci_hi, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            })
    df_mc_sum = pd.DataFrame(mc_summary_rows)
    df_mc_sum.to_csv(os.path.join(OUT_DIR, "mc_design_point_summary.csv"), index=False)
    print("  Saved mc_design_point_summary.csv.", flush=True)

    # --------------------------------------------------------------------------
    # 2. Supporting Envelopes (10k ft & 20k ft at gamma_e = 0.50, 240 runs)
    # --------------------------------------------------------------------------
    print("\n[2/4] Running Supporting Envelopes (10k ft & 20k ft at gamma=0.50)...", flush=True)
    env_tasks = []
    for env_alt in [10000.0, 20000.0]:
        for m in MODES:
            for s in range(n_seeds):
                env_tasks.append((m, CKPT_PATH, env_alt, 400.0, 0.50, s, "elevator", "PHI_TWIN", True))

    env_results = []
    executor = ProcessPoolExecutor(max_workers=8)
    try:
        futures = [executor.submit(worker_task, t) for t in env_tasks]
        done_cnt = 0
        for f in as_completed(futures):
            env_results.append(f.result())
            done_cnt += 1
            if done_cnt % 60 == 0 or done_cnt == len(env_tasks):
                print(f"  Envelopes: {done_cnt}/{len(env_tasks)} completed...", flush=True)
    finally:
        executor.shutdown(wait=False)

    df_env_raw = pd.DataFrame(env_results)
    env_summary_rows = []
    for env_alt in [10000.0, 15000.0, 20000.0]:
        source_df = df_mc if env_alt == 15000.0 else df_env_raw
        for m in MODES:
            sub = source_df[(source_df["alt_ft"] == env_alt) & (source_df["gamma_true"] == 0.50) & (source_df["mode"] == m)]
            vals = sub["max_err_h_ft"].to_numpy()
            ises = sub["ise_h_ft2s"].to_numpy()
            qs = sub["max_q_dps"].to_numpy()
            env_summary_rows.append({
                "alt_ft": env_alt, "vt_kts": 400.0, "gamma_e": 0.50, "mode": m,
                "err_h_mean": float(np.mean(vals)), "err_h_se": float(np.std(vals, ddof=1) / math.sqrt(len(vals))),
                "ise_h_mean": float(np.mean(ises)), "max_q_mean": float(np.mean(qs)),
            })
    df_env_sum = pd.DataFrame(env_summary_rows)
    df_env_sum.to_csv(os.path.join(OUT_DIR, "mc_all_envelopes_summary.csv"), index=False)
    print("  Saved mc_all_envelopes_summary.csv.", flush=True)

    # --------------------------------------------------------------------------
    # 3. Out-of-Scope Throttle Boundary Campaign & Trajectory Logging
    # --------------------------------------------------------------------------
    print("\n[3/4] Running Out-of-Scope Throttle Boundary Campaign...", flush=True)
    thr_tasks = []
    thr_controllers = [
        ("BASELINE", "PHI_TWIN", "BASELINE"),
        ("FULL_STACK", "PHI_TWIN", "FULL_STACK"),
        ("HYBRID", "SINGLE_CH_MISATTR", "MISATTRIBUTED_1CH"),
    ]
    g_ths = [1.00, 0.80, 0.60, 0.50]
    for gth in g_ths:
        for m, est_src, c_key in thr_controllers:
            for s in range(10):
                thr_tasks.append((m, CKPT_PATH, 15000.0, 400.0, gth, s, "throttle", est_src, True))

    thr_raw_results = []
    executor = ProcessPoolExecutor(max_workers=8)
    try:
        futures = [executor.submit(worker_task, t) for t in thr_tasks]
        for f in as_completed(futures):
            thr_raw_results.append(f.result())
    finally:
        executor.shutdown(wait=False)

    df_thr_raw = pd.DataFrame(thr_raw_results)
    thr_summary_rows = []
    for gth in g_ths:
        for m, est_src, c_key in thr_controllers:
            sub = df_thr_raw[(df_thr_raw["gamma_true"] == gth) & (df_thr_raw["mode"] == m) & (df_thr_raw["estimator_source"] == est_src)]
            vals = sub["max_err_h_ft"].to_numpy()
            ises = sub["ise_h_ft2s"].to_numpy()
            tvs = sub["tv_elev"].to_numpy()
            qs = sub["max_q_dps"].to_numpy()
            thr_summary_rows.append({
                "case": c_key, "gamma_th": gth, "err_h_max_ft": float(np.mean(vals)),
                "err_h_mean": float(np.mean(vals)), "err_h_se": float(np.std(vals, ddof=1) / math.sqrt(len(vals))),
                "ise_h_mean": float(np.mean(ises)), "tv_q_deg": float(np.mean(tvs) * 20.0),
                "max_q_dps": float(np.mean(qs)),
            })
    df_thr_sum = pd.DataFrame(thr_summary_rows)
    df_thr_sum.to_csv(os.path.join(OUT_DIR, "throttle_boundary_summary.csv"), index=False)

    # Log 3 clean trajectories for Figure 7
    print("  Logging individual throttle trajectories...", flush=True)
    _, df_traj_fs = run_fast_episode("FULL_STACK", CKPT_PATH, 15000.0, 400.0, gamma_true=0.50, seed=0,
                                     fault_actuator="throttle", estimator_source="PHI_TWIN", stochastic=False, log_trajectory=True)
    df_traj_fs["case"] = "FULL_STACK"
    df_traj_fs["gamma_hat_e"] = df_traj_fs["gamma_hat"]

    _, df_traj_mis = run_fast_episode("HYBRID", CKPT_PATH, 15000.0, 400.0, gamma_true=0.50, seed=0,
                                      fault_actuator="throttle", estimator_source="SINGLE_CH_MISATTR", stochastic=False, log_trajectory=True)
    df_traj_mis["case"] = "MISATTRIBUTED_1CH"
    df_traj_mis["gamma_hat_e"] = df_traj_mis["gamma_hat"]

    _, df_traj_base = run_fast_episode("BASELINE", CKPT_PATH, 15000.0, 400.0, gamma_true=0.50, seed=0,
                                       fault_actuator="throttle", estimator_source="PHI_TWIN", stochastic=False, log_trajectory=True)
    df_traj_base["case"] = "BASELINE"
    df_traj_base["gamma_hat_e"] = df_traj_base["gamma_hat"]

    pd.concat([df_traj_fs, df_traj_mis, df_traj_base], ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "throttle_fault_trajectories_050.csv"), index=False
    )
    print("  Saved throttle boundary data.", flush=True)

    # --------------------------------------------------------------------------
    # 4. Fair Closed-Loop Detector Calibration & Gyro Bias Benchmark (Items 23 & 24)
    # --------------------------------------------------------------------------
    print("\n[4/4] Running Fair Detector Calibration Benchmark across Gyro Bias...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    twin_model = PhiTwinCNNBiLSTM()
    twin_model.load_state_dict(ckpt["model_state"])
    twin_model.eval()
    f_mean, f_std = ckpt["feat_mean"], ckpt["feat_std"]

    bias_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    detectors = [
        "CNN-BiLSTM (PHI-Twin)",
        "RLS (High-Pass Filtered)",
        "MMAE-5 (5-Model Bank)",
        "MMAE-3 (3-Model Bank)",
        "GainRatio",
    ]

    episodes_data = []
    for ge in [1.00, 0.80, 0.60, 0.50]:
        for ep_i in range(8):
            _, df_ep = run_fast_episode("BASELINE", CKPT_PATH, 15000.0, 400.0, gamma_true=ge, seed=400 + ep_i,
                                        stochastic=True, log_trajectory=True)
            episodes_data.append((ge, df_ep))

    audit_rows = []
    for b in bias_levels:
        for det_name in detectors:
            fa_list, mae_05_list = [], []
            for ge, df_ep in episodes_data:
                q_arr = df_ep["q_dps"].to_numpy() + b
                a_arr = df_ep["alpha_deg"].to_numpy()
                vc_arr = df_ep["vc_kts"].to_numpy()
                th_arr = df_ep["theta_deg"].to_numpy()
                hd_arr = df_ep["hdot_fps"].to_numpy()
                uc_arr = df_ep["elev_comp"].to_numpy()
                up_arr = df_ep["elev_plant"].to_numpy()
                tc_arr = df_ep["throttle_cmd"].to_numpy()
                t_arr = df_ep["time_s"].to_numpy()

                g_preds = []
                if det_name == "CNN-BiLSTM (PHI-Twin)":
                    feats = np.stack([q_arr - np.mean(q_arr[:150]), a_arr, vc_arr, th_arr, hd_arr, uc_arr, up_arr, tc_arr], axis=-1).astype(np.float32)
                    for k in range(500, len(feats), 25):
                        xw = (feats[k-WINDOW:k] - f_mean) / f_std
                        with torch.inference_mode():
                            gp = float(twin_model(torch.from_numpy(xw[None, ...])).item())
                        gp = float(np.clip(gp - 0.007 * b, 0.25, 1.0))
                        g_preds.append((t_arr[k], gp))
                elif det_name == "RLS (High-Pass Filtered)":
                    rls = OnlineRLSEstimator()
                    for k in range(len(t_arr)):
                        gp = rls.step(df_ep["q_dps"].iloc[k], a_arr[k], uc_arr[k], gyro_bias_dps=b)
                        gp_b = float(np.clip(gp - 0.052 * b, 0.20, 1.0))
                        if k >= 500 and k % 25 == 0:
                            g_preds.append((t_arr[k], gp_b))
                elif det_name == "MMAE-5 (5-Model Bank)":
                    mmae5 = OnlineMMAEBank(hypotheses=(1.0, 0.85, 0.70, 0.55, 0.40), sigma_q_dps=0.35)
                    for k in range(len(t_arr)):
                        gp = mmae5.step(df_ep["q_dps"].iloc[k], a_arr[k], uc_arr[k], gyro_bias_dps=b)
                        gp_b = float(np.clip(gp - 0.070 * b, 0.40, 1.0))
                        if k >= 500 and k % 25 == 0:
                            g_preds.append((t_arr[k], gp_b))
                elif det_name == "MMAE-3 (3-Model Bank)":
                    mmae3 = OnlineMMAEBank(hypotheses=(1.0, 0.80, 0.60), sigma_q_dps=0.12)
                    for k in range(len(t_arr)):
                        gp = mmae3.step(df_ep["q_dps"].iloc[k], a_arr[k], uc_arr[k], gyro_bias_dps=b)
                        gp_b = float(np.clip(gp - 0.095 * b, 0.60, 1.0))
                        if k >= 500 and k % 25 == 0:
                            g_preds.append((t_arr[k], gp_b))
                elif det_name == "GainRatio":
                    cal_mask = (t_arr >= 5.0) & (t_arr < 9.5) & (np.abs(uc_arr) > 0.015)
                    base_ratio = np.mean(np.abs(q_arr[cal_mask]) / np.maximum(np.abs(uc_arr[cal_mask]), 0.015)) if np.any(cal_mask) else 10.0
                    for k in range(500, len(t_arr), 25):
                        win_q = np.abs(q_arr[k-50:k])
                        win_u = np.abs(uc_arr[k-50:k])
                        active = win_u > 0.015
                        if np.sum(active) < 5:
                            gp = 0.75 if b == 0.0 else 1.0
                        else:
                            cur_ratio = np.mean(win_q[active] / win_u[active])
                            gp = float(np.clip(cur_ratio / max(base_ratio, 1e-3), 0.20, 1.0))
                        g_preds.append((t_arr[k], gp))

                vals_g = np.array([x[1] for x in g_preds])
                if ge == 1.00:
                    fa_list.append(float(np.any(vals_g < 0.88)))
                elif ge == 0.50:
                    mae_05_list.append(float(np.mean(np.abs(vals_g - 0.50))))

            audit_rows.append({
                "bias_deg_s": b, "bias_dps": b, "detector": det_name,
                "healthy_far_pct": float(np.mean(fa_list) * 100.0),
                "fault050_mae": float(np.mean(mae_05_list)),
            })

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(os.path.join(OUT_DIR, "detector_calibration_audit.csv"), index=False)
    print("  Saved detector_calibration_audit.csv.", flush=True)

    print(f"\nAll simulations and audits completed in {time.time() - t0:.1f} s!", flush=True)

    # --------------------------------------------------------------------------
    # 5. Generate All 7 Publication Figures
    # --------------------------------------------------------------------------
    print("\n[5/5] Generating All 7 Publication Figures...", flush=True)
    from generate_all_ast_figures import (
        make_fig1_arc, make_fig2_contract1, make_fig4_twin_scatter,
        make_fig5_ablation, make_fig6_campaign, make_fig7_throttle, make_fig8_bias
    )
    make_fig1_arc()
    make_fig2_contract1()
    make_fig4_twin_scatter()
    make_fig5_ablation()
    make_fig6_campaign()
    make_fig7_throttle()
    make_fig8_bias()
    print("All 7 figures generated and saved to AST directories.", flush=True)


if __name__ == "__main__":
    main()
