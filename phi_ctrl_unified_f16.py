#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL Unified Orchestrator — F-16 (V2)
=========================================
Layer 0/1 (gate-PASSING plant + EnergyHold) + Layer 2 (MRAC, diagnostic
observer, MMAE effectiveness bank, optional PPO residual).

Terminology (mandatory)
-----------------------
  γ  = elevator *remaining effectiveness* in [0, 1]
  γ = 1.0  → healthy
  γ = 0.8  → 20% effectiveness *loss* (80% remaining)
  γ = 0.5  → 50% effectiveness *loss* (50% remaining)

Design decisions
----------------
  - POSITIVE elev_cmd = NOSE DOWN on this airframe.
  - Single clip before plant write; physical γ applied last.
  - Observer is DIAGNOSTIC-ONLY (OBSERVER_CLOSED_LOOP=False).
  - Detector path: GainRatioDetector + ElevEffectivenessBank (MMAE-style).
  - RL residual applies only when fault is active AND bank estimates
    degradation (gamma_hat < residual enable threshold) — not blindly.
  - BASELINE has zero fault knowledge (honest comparison).
  - Staged gate: BASELINE pre-fault must pass before augmented cases.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plant.jsbsim_plant_f16 import (
    DT, ENV, PROP_AIL, PROP_RUD,
    native_trim, force_ic, ownership, set_throttle, set_elev, set_pitch_trim,
    flight_state,
)
from controller.energy_hold_f16 import EnergyHold
from controller.mrac.adaptive_controller import MRACAdaptiveController
from sensor_fusion.observer import BiasCompensator
from detector.fault_detector import GainRatioDetector
from detector.mmae_bank import ElevEffectivenessBank
from detector.phi_twin_cnn_bilstm import PhiTwinCNNLSTM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import jsbsim
except ImportError as exc:
    raise ImportError("jsbsim not found. Install: conda install -c conda-forge jsbsim") from exc


# ---------------------------------------------------------------------------
# Constants — γ is *remaining* effectiveness
# ---------------------------------------------------------------------------
AIRCRAFT_NAME = "f16"
OUTPUT_DIR = HERE / "results_f16"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SETTLE_S = 18.0
FAULT_START_TIME = 15.0
SIM_DURATION = 60.0

# Default study point: 50% remaining effectiveness (= 50% loss).
# Override with --gamma 0.8 for a 20% loss study.
DEFAULT_GAMMA_REMAINING = 0.50

OBSERVER_CLOSED_LOOP = False  # diagnostic-only
BASELINE_GATE_MAX_DH_FT = 800.0
BASELINE_GATE_MAX_HDOT_FPS = 20.0
INTERLOCK_THETA_DEG = 30.0
INTERLOCK_PHI_DEG = 30.0
BAILOUT_THETA_DEG = 60.0
BAILOUT_PHI_DEG = 60.0
BAILOUT_MIN_ALT_FT = 5000.0

# Residual enable: only when bank believes remaining effectiveness dropped
RESIDUAL_ENABLE_GAMMA_HAT = 0.92
RESIDUAL_MAX = 0.20          # hard cap on residual contribution (was 0.5 — saturates plant)
RESIDUAL_RATE_MAX = 0.05     # per step at 120 Hz ≈ 6/s slew
RESIDUAL_GAIN = 0.35         # scale raw policy output before authority weighting


class TeeLogger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.logfile.close()


class FlightLogger:
    FIELDS = [
        "step", "time_s", "case_name", "alt_ft", "vc_kts",
        "theta_deg", "phi_deg", "hdot_fps", "q_dps",
        "elev_raw", "elev_comp", "elev_plant", "throttle", "speedbrake",
        "case_gamma_estimate", "physical_gamma", "comp_factor",
        "mrac_elev", "observer_bias_q", "observer_bias_theta",
        "detector_severity", "detector_confidence",
        "mmae_gamma_hat", "mmae_best_gamma", "phi_twin_gamma_hat", "phi_twin_conf",
        "residual_enabled", "rl_residual", "interlock_tripped",
    ]

    def __init__(self, filepath):
        self._fp = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self.step = 0

    def log(self, **kwargs):
        self.step += 1
        row = {k: kwargs.get(k, "") for k in self.FIELDS}
        row["step"] = self.step
        self._writer.writerow(row)

    def close(self):
        self._fp.close()
        print(f"[LOG] Wrote {self.step} rows")


def loss_percent_from_gamma(g: float) -> float:
    """γ remaining → percent loss (for human-readable logs)."""
    return 100.0 * (1.0 - float(g))


def check_baseline_gate(csv_path, fault_start_time=FAULT_START_TIME):
    df = pd.read_csv(csv_path)
    if df.empty:
        return False, ["no data logged"]
    pre = df[df["time_s"] < fault_start_time]
    if pre.empty:
        return False, ["no pre-fault data"]
    h0 = pre["alt_ft"].iloc[0]
    max_dh = (pre["alt_ft"] - h0).abs().max()
    max_hdot = pre["hdot_fps"].abs().max()
    reasons = []
    if max_dh > BASELINE_GATE_MAX_DH_FT:
        reasons.append(f"pre-fault max|dh|={max_dh:.1f}ft > {BASELINE_GATE_MAX_DH_FT}ft")
    if max_hdot > BASELINE_GATE_MAX_HDOT_FPS:
        reasons.append(f"pre-fault max|hdot|={max_hdot:.1f}fps > {BASELINE_GATE_MAX_HDOT_FPS}fps")
    return (len(reasons) == 0), reasons


def run_case(
    case_name,
    csv_path,
    gamma_remaining,
    use_mracs=False,
    use_observer=False,
    use_detector=False,
    use_rl=False,
    obs_mats=None,
    rl_model_path=None,
):
    print(f"\n{'=' * 70}\n  CASE : {case_name}\n{'=' * 70}")
    print(
        f"  physical γ_remaining={gamma_remaining:.2f} "
        f"({loss_percent_from_gamma(gamma_remaining):.0f}% effectiveness loss) "
        f"after t={FAULT_START_TIME:.0f}s"
    )

    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model(AIRCRAFT_NAME):
        raise RuntimeError(f"Failed to load aircraft: {AIRCRAFT_NAME}")

    ok, trim_thr, trim_elev, trim_ptrim, trim_theta = native_trim(fdm)
    if not ok:
        print(f"[TRIM] FAILED for case {case_name}")
        return True, {}

    print(f"[SETTLE] {SETTLE_S:.0f}s closed-loop settle...")
    baseline = EnergyHold(trim_thr, trim_elev, trim_ptrim, trim_theta, DT)
    for _ in range(int(SETTLE_S / DT)):
        cmds = baseline.update(fdm, ENV["alt_ft"], ENV["vc_kts"])
        set_elev(fdm, cmds["elev"])
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()
    st0 = flight_state(fdm)
    h0 = st0["h"]
    print(f"[SETTLE] h={h0:.0f}ft Vc={st0['vc']:.1f}kts theta={st0['theta']:+.2f}deg")

    baseline.elev0 = st0["elev_cmd"] if abs(st0["elev_cmd"]) > 1e-6 else trim_elev
    baseline.thr0 = max(st0["thr"] if st0["thr"] else trim_thr, 0.0)
    baseline.theta0 = st0["theta"]
    baseline.prev_elev = baseline.elev0
    baseline.prev_thr = baseline.thr0
    baseline.reset()

    logger = FlightLogger(csv_path)

    mrac = MRACAdaptiveController(dt=DT) if use_mracs else None
    mrac_prev_u, mrac_du_max, mrac_u_max = 0.0, 0.08, 0.25

    observer = None
    if use_observer and obs_mats is not None:
        A, B, C, L = obs_mats
        observer = BiasCompensator(A=A, B=B, L=L, C=C, dt=DT)

    gain_det = (
        GainRatioDetector(
            dt=DT, calib_window=1.0, roll_window=0.5,
            cmd_threshold=0.001, fault_threshold=0.95,
        )
        if use_detector
        else None
    )
    mmae = ElevEffectivenessBank(gammas=(1.0, 0.8, 0.6), process_var=0.05, dt=DT) if use_detector else None
    if mmae is not None:
        mmae.reset()
    phi_twin = None
    if use_detector:
        twin_path = HERE / "models" / "phi_twin_cnn_bilstm.pt"
        phi_twin = PhiTwinCNNLSTM(checkpoint=twin_path)
        if not phi_twin.available:
            phi_twin = None
        else:
            phi_twin.reset()

    rl_model = None
    if use_rl and rl_model_path and Path(rl_model_path).exists():
        try:
            from stable_baselines3 import PPO
            rl_model = PPO.load(rl_model_path, device="cpu")
            fname = Path(rl_model_path).name
            if "f16" not in fname.lower():
                print(
                    f"[RL][WARNING] '{fname}' may not be F-16-trained. "
                    f"Prefer models/phi_ctrl_residual_f16.zip."
                )
            else:
                print(f"[RL] Loaded F-16 checkpoint '{fname}'.")
            print(
                f"[RL] Residual enable gated by MMAE: gamma_hat < {RESIDUAL_ENABLE_GAMMA_HAT}"
            )
        except Exception as e:
            print(f"[RL] Load failed, inert: {e}")
    elif use_rl:
        print(f"[RL] No model at {rl_model_path} — residual inert.")

    crash = False
    interlock_tripped = False
    n = int(SIM_DURATION / DT)
    t_offset = fdm.get_sim_time()
    prev_elev_cmd = trim_elev
    residual_on_count = 0
    prev_rl_res = 0.0

    for i in range(n):
        t = fdm.get_sim_time() - t_offset
        fault_active = t >= FAULT_START_TIME
        physical_gamma = float(gamma_remaining) if fault_active else 1.0

        st = flight_state(fdm)
        theta_deg_now = st["theta"]
        phi_deg_now = math.degrees(fdm.get_property_value("attitude/phi-rad"))

        if not interlock_tripped and (use_mracs or use_observer or use_rl or use_detector):
            if abs(theta_deg_now) > INTERLOCK_THETA_DEG or abs(phi_deg_now) > INTERLOCK_PHI_DEG:
                interlock_tripped = True
                print(
                    f"[INTERLOCK] TRIPPED t={t:.2f}s θ={theta_deg_now:.1f} φ={phi_deg_now:.1f} "
                    f"— augmentation off, raw baseline only."
                )

        eff_mracs = use_mracs and not interlock_tripped
        eff_observer_cl = use_observer and not interlock_tripped and OBSERVER_CLOSED_LOOP
        eff_rl = use_rl and not interlock_tripped
        eff_detector = use_detector and not interlock_tripped

        # --- Diagnostic observer (never closed-loop unless flag True) ---
        obs_bias = {"q": 0.0, "theta": 0.0}
        if observer is not None:
            raw5 = np.array([
                st["vc"] * 1.68781, 0.0,
                math.radians(st["q"]), math.radians(st["theta"]), st["h"],
            ])
            u_ctrl = np.array([prev_elev_cmd, 0.0])
            try:
                _y, bias = observer.compensate(raw5, u_ctrl, fault_active=fault_active)
                obs_bias = {"q": float(bias[2]), "theta": float(bias[3])}
            except Exception:
                pass
            _ = eff_observer_cl  # reserved for future closed-loop; intentionally unused

        # --- Integrity: MMAE (physics) + optional PHI-Twin CNN-BiLSTM ---
        case_gamma_estimate = 1.0
        det_state = {"severity": 1.0, "confidence": 1.0}
        mmae_gamma_hat = 1.0
        mmae_best = 1.0
        twin_g, twin_c = 1.0, 0.0

        if gain_det is not None:
            fs = gain_det.update(q_dps=st["q"], elev_cmd=prev_elev_cmd)
            det_state = {"severity": fs.severity, "confidence": fs.confidence}
            if eff_detector and fs.is_faulty:
                case_gamma_estimate = fs.severity

        if mmae is not None:
            bank = mmae.update(elev_cmd=prev_elev_cmd, q_dps=st["q"])
            mmae_gamma_hat = bank["gamma_hat"]
            mmae_best = bank["best_gamma"]
            if eff_detector:
                case_gamma_estimate = min(case_gamma_estimate, mmae_gamma_hat)

        if phi_twin is not None and eff_detector:
            twin_out = phi_twin.update({
                "q_dps": st["q"],
                "theta_deg": st["theta"],
                "hdot_fps": st["hdot"],
                "vc_kts": st["vc"],
                "alpha_deg": st.get("alpha", 0.0),
                "elevator_cmd": prev_elev_cmd,
                "elevator_pos": prev_elev_cmd,
                "throttle": st.get("thr", 0.0),
            })
            twin_g = twin_out["gamma_hat"]
            twin_c = twin_out["confidence"]
            if twin_out["source"] == "cnn_bilstm" and twin_c > 0.3:
                # Fuse: prefer learned γ̂ when confident, else physics bank
                case_gamma_estimate = (1.0 - twin_c) * case_gamma_estimate + twin_c * twin_g

        cmds = baseline.update(fdm, ENV["alt_ft"], ENV["vc_kts"])
        elev_raw = cmds["elev"]

        # --- MRAC ---
        mrac_elev = 0.0
        if mrac is not None and eff_mracs:
            target_theta_rad = math.radians(cmds["pitch_cmd_deg"])
            mrac_state = np.array([
                st["vc"] * 1.68781, 0.0,
                math.radians(st["q"]), math.radians(st["theta"]), st["h"],
            ])
            try:
                u_out = mrac.compute_action(mrac_state, target_theta=target_theta_rad)
                u_out = float(u_out[0]) if hasattr(u_out, "__len__") else float(u_out)
            except Exception:
                u_out = 0.0
            u_out = float(np.clip(u_out, -mrac_u_max, mrac_u_max))
            du = float(np.clip(u_out - mrac_prev_u, -mrac_du_max, mrac_du_max))
            mrac_elev = mrac_prev_u + du
            mrac_prev_u = mrac_elev
            mrac_elev = -mrac_elev

        # --- RL residual: additive correction ON TOP of 1/γ compensation ---
        # Critical fix: never drop comp_factor when residual is on (that was
        # the FULL_STACK dive: residual +0.5 replaced authority recovery).
        rl_res = 0.0
        residual_enabled = False
        if rl_model is not None and eff_rl and fault_active:
            g_for_gate = case_gamma_estimate
            if phi_twin is not None:
                g_for_gate = min(g_for_gate, twin_g)
            if mmae is not None:
                g_for_gate = min(g_for_gate, mmae_gamma_hat)
            residual_enabled = g_for_gate < RESIDUAL_ENABLE_GAMMA_HAT
            if mmae is None and phi_twin is None:
                residual_enabled = True
            if residual_enabled:
                residual_on_count += 1
                try:
                    obs8 = np.array([
                        st["vc"] * 1.68781,
                        0.0,
                        math.radians(st["q"]),
                        math.radians(st["theta"]),
                        st["h"],
                        ENV["alt_ft"],
                        prev_elev_cmd,
                        case_gamma_estimate,
                    ], dtype=np.float32)
                    action, _ = rl_model.predict(obs8, deterministic=True)
                    raw_a = float(np.clip(float(action[0]), -0.5, 0.5))
                    # Scale by authority deficit so residual only fills lost γ
                    deficit = float(np.clip(1.0 - case_gamma_estimate, 0.0, 1.0))
                    desired = raw_a * RESIDUAL_GAIN * max(deficit, 0.15)
                    desired = float(np.clip(desired, -RESIDUAL_MAX, RESIDUAL_MAX))
                    # Rate limit (prevents 0→0.5 jump in one step)
                    du = float(np.clip(desired - prev_rl_res, -RESIDUAL_RATE_MAX, RESIDUAL_RATE_MAX))
                    rl_res = prev_rl_res + du
                except Exception as e:
                    if i == int(FAULT_START_TIME / DT):
                        print(f"[RL] Inference error at t={t:.1f}s: {e}")
                    rl_res = prev_rl_res * 0.9
            else:
                # Soft bleed residual off when gate closes
                rl_res = prev_rl_res * 0.85
            prev_rl_res = rl_res
        else:
            prev_rl_res *= 0.85
            rl_res = prev_rl_res

        # BASELINE: no compensation knowledge
        if case_name == "BASELINE":
            case_gamma_estimate = 1.0
            comp_factor = 1.0
            rl_res = 0.0
        else:
            comp_factor = min(1.0 / case_gamma_estimate if case_gamma_estimate > 0.01 else 1.0, 4.0)

        # Always apply authority compensation; residual is a bounded add-on
        classical = elev_raw + mrac_elev
        elev_comp = float(np.clip(classical * comp_factor + rl_res, -1.0, 1.0))
        elev_plant = elev_comp * physical_gamma
        prev_elev_cmd = elev_comp

        set_elev(fdm, elev_plant)
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

        st_new = flight_state(fdm)
        logger.log(
            time_s=round(t, 6), case_name=case_name,
            alt_ft=st_new["h"], vc_kts=st_new["vc"],
            theta_deg=st_new["theta"],
            phi_deg=math.degrees(fdm.get_property_value("attitude/phi-rad")),
            hdot_fps=st_new["hdot"], q_dps=st_new["q"],
            elev_raw=elev_raw, elev_comp=elev_comp, elev_plant=elev_plant,
            throttle=cmds["throttle"], speedbrake=cmds["speedbrake"],
            case_gamma_estimate=case_gamma_estimate,
            physical_gamma=physical_gamma, comp_factor=comp_factor,
            mrac_elev=mrac_elev,
            observer_bias_q=obs_bias["q"], observer_bias_theta=obs_bias["theta"],
            detector_severity=det_state["severity"],
            detector_confidence=det_state["confidence"],
            mmae_gamma_hat=mmae_gamma_hat, mmae_best_gamma=mmae_best,
            phi_twin_gamma_hat=twin_g, phi_twin_conf=twin_c,
            residual_enabled=int(residual_enabled),
            rl_residual=rl_res, interlock_tripped=int(interlock_tripped),
        )

        phi_deg = math.degrees(fdm.get_property_value("attitude/phi-rad"))
        theta_deg = st_new["theta"]
        if (
            st_new["h"] < BAILOUT_MIN_ALT_FT
            or abs(phi_deg) > BAILOUT_PHI_DEG
            or abs(theta_deg) > BAILOUT_THETA_DEG
        ):
            print(
                f"  *** BAILOUT t={t:.2f}s h={st_new['h']:.0f} "
                f"θ={theta_deg:.1f} φ={phi_deg:.1f} ***"
            )
            crash = True
            break

    logger.close()
    df = pd.read_csv(csv_path)
    max_dh = (df["alt_ft"] - h0).abs().max()
    metrics = {
        "case": case_name,
        "max_abs_dh": float(max_dh),
        "min_theta": float(df["theta_deg"].min()),
        "max_theta": float(df["theta_deg"].max()),
        "max_roll": float(df["phi_deg"].abs().max()),
        "interlock": bool(interlock_tripped),
        "crash": bool(crash),
        "residual_steps": int(residual_on_count),
        "gamma_remaining": float(gamma_remaining),
    }
    print(
        f"\nMETRICS ({case_name}): max|dh|={max_dh:.1f}ft  "
        f"θ=[{metrics['min_theta']:.1f},{metrics['max_theta']:.1f}]  "
        f"roll_max={metrics['max_roll']:.1f}  interlock={interlock_tripped}  "
        f"crash={crash}  residual_steps={residual_on_count}"
    )
    return crash, metrics


def build_diagnostic_observer_mats():
    """Diagnostic-only approximate matrices — NOT F-16 specific.
    Never fed back into control while OBSERVER_CLOSED_LOOP=False."""
    A = np.array([
        [-0.02, 0.05, 0.0, -32.2, 0.0],
        [-0.10, -0.60, 700.0, 0.0, 0.0],
        [0.0, -0.01, -0.9, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 700.0, 0.0],
    ])
    B = np.array([[0.0, 8.0], [-30.0, 0.0], [-15.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    C = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=float)
    L = np.zeros((5, 3))
    try:
        from scipy.signal import place_poles
        L = place_poles(A.T, C.T, [-0.8, -1.0, -1.2, -2.5, -3.0]).gain_matrix.T
    except Exception as e:
        print(f"[WARN] Pole placement failed ({e}), L=zeros")
    return A, B, C, L


def plot_all_cases(csv_paths, out_dir, gamma_remaining):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"PHI-CTRL F-16 Ablation (γ_remaining={gamma_remaining:.2f}, "
        f"{loss_percent_from_gamma(gamma_remaining):.0f}% loss)",
        fontsize=13, fontweight="bold",
    )
    colors = {
        "BASELINE": "#2563eb", "TECS_MRAC": "#dc2626",
        "HYBRID": "#16a34a", "FULL_STACK": "#9333ea",
    }
    panels = [
        ("alt_ft", "Altitude (ft)"), ("vc_kts", "Airspeed (kts)"),
        ("hdot_fps", "Vertical Speed (fps)"),
        ("theta_deg", "Pitch (deg)"), ("phi_deg", "Roll (deg)"),
        ("throttle", "Throttle Cmd"),
        ("elev_plant", "Elevator to Plant"),
        ("mmae_gamma_hat", "MMAE γ̂"),
        ("rl_residual", "RL Residual"),
    ]
    for ax, (col, title) in zip(axes.flat, panels):
        ax.axvline(FAULT_START_TIME, color="k", ls="--", lw=1.0, alpha=0.4)
        for case_name, path in csv_paths.items():
            df = pd.read_csv(path)
            if df.empty or col not in df.columns:
                continue
            ax.plot(
                df["time_s"], df[col],
                color=colors.get(case_name, "#000"), lw=1.4,
                label=case_name, alpha=0.85,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "phi_ctrl_unified_f16_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[PLOT] Saved: {out_path}")


def write_ablation_table(metrics_list, out_dir, gamma_remaining):
    path = out_dir / "ablation_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case", "gamma_remaining", "loss_percent", "max_abs_dh",
                "min_theta", "max_theta", "max_roll", "interlock", "crash",
                "residual_steps",
            ],
        )
        w.writeheader()
        for m in metrics_list:
            w.writerow({
                "case": m["case"],
                "gamma_remaining": m["gamma_remaining"],
                "loss_percent": loss_percent_from_gamma(m["gamma_remaining"]),
                "max_abs_dh": m["max_abs_dh"],
                "min_theta": m["min_theta"],
                "max_theta": m["max_theta"],
                "max_roll": m["max_roll"],
                "interlock": m["interlock"],
                "crash": m["crash"],
                "residual_steps": m["residual_steps"],
            })
    print(f"[ABLATION] Wrote {path}")
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (honest — residual not assumed superior)")
    print("=" * 70)
    print(
        f"{'Case':<12} {'max|Δh|':>10} {'θ min':>8} {'θ max':>8} "
        f"{'crash':>6} {'resid_steps':>12}"
    )
    for m in metrics_list:
        print(
            f"{m['case']:<12} {m['max_abs_dh']:10.1f} {m['min_theta']:8.1f} "
            f"{m['max_theta']:8.1f} {str(m['crash']):>6} {m['residual_steps']:12d}"
        )
    print("=" * 70)
    print(
        "Note: FULL_STACK residual is MMAE-gated. If max|Δh| exceeds BASELINE, "
        "report as ablation finding and retrain with --curriculum."
    )


def main():
    parser = argparse.ArgumentParser(description="PHI-CTRL F-16 unified ablation")
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA_REMAINING,
        help="Elevator remaining effectiveness after fault (e.g. 0.8 = 20%% loss, 0.5 = 50%% loss)",
    )
    parser.add_argument(
        "--rl-model", default=str(HERE / "models" / "phi_ctrl_residual_f16.zip"),
    )
    args = parser.parse_args()
    gamma_remaining = float(np.clip(args.gamma, 0.05, 1.0))

    tee = TeeLogger(str(OUTPUT_DIR / "console_log_f16_unified.txt"))
    print("PHI-CTRL Unified Orchestrator — F-16 (V2)")
    print(f"OBSERVER_CLOSED_LOOP={OBSERVER_CLOSED_LOOP} (diagnostic-only)")
    print(
        f"Fault: γ_remaining={gamma_remaining:.2f} "
        f"({loss_percent_from_gamma(gamma_remaining):.0f}% effectiveness loss) "
        f"at t≥{FAULT_START_TIME:.0f}s"
    )
    twin_ckpt = HERE / "models" / "phi_twin_cnn_bilstm.pt"
    print(
        "Detector: MMAE physics bank"
        + (" + PHI-Twin CNN-BiLSTM" if twin_ckpt.exists() else " (CNN-BiLSTM ckpt not found — physics only)")
    )
    print(f"RL residual enable threshold: gamma_hat < {RESIDUAL_ENABLE_GAMMA_HAT}")

    obs_mats = build_diagnostic_observer_mats()
    csv_paths = {}
    metrics_list = []

    baseline_csv = OUTPUT_DIR / "unified_baseline_log.csv"
    crash, m = run_case(
        "BASELINE", str(baseline_csv), gamma_remaining, obs_mats=obs_mats,
    )
    csv_paths["BASELINE"] = str(baseline_csv)
    if m:
        metrics_list.append(m)
    passed, reasons = check_baseline_gate(str(baseline_csv))

    if not passed or crash:
        print(f"\n{'#' * 70}\n[GATE] BASELINE FAILED: {reasons + (['crashed'] if crash else [])}")
        print("[GATE] Stopping — fix baseline before augmented cases.")
        if metrics_list:
            write_ablation_table(metrics_list, OUTPUT_DIR, gamma_remaining)
        plot_all_cases(csv_paths, OUTPUT_DIR, gamma_remaining)
        tee.close()
        sys.exit(1)

    print("\n[GATE] BASELINE PASSED. Proceeding to TECS_MRAC, HYBRID, FULL_STACK.")

    cases = {
        "TECS_MRAC": {"mrac": True, "obs": False, "det": False, "rl": False},
        "HYBRID": {"mrac": False, "obs": False, "det": True, "rl": False},
        "FULL_STACK": {"mrac": True, "obs": True, "det": True, "rl": True},
    }
    for case_name, flags in cases.items():
        csv_path = OUTPUT_DIR / f"unified_{case_name.lower()}_log.csv"
        csv_paths[case_name] = str(csv_path)
        _, m = run_case(
            case_name, str(csv_path), gamma_remaining,
            use_mracs=flags["mrac"], use_observer=flags["obs"],
            use_detector=flags["det"], use_rl=flags["rl"],
            obs_mats=obs_mats, rl_model_path=args.rl_model,
        )
        if m:
            metrics_list.append(m)

    write_ablation_table(metrics_list, OUTPUT_DIR, gamma_remaining)
    plot_all_cases(csv_paths, OUTPUT_DIR, gamma_remaining)
    print("\n[DONE] All cases complete.")
    tee.close()


if __name__ == "__main__":
    main()
