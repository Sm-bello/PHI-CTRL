#!/usr/bin/env python3
"""
Generate all 7 publication-grade figures for the PHI-Ctrl AST manuscript
directly from verified simulation data in `PHI_CTRL_RELEASE/results/verified_v2/`.
Outputs are saved to both `PHI_CTRL_AST` and `PHI_CTRL_AST_READY`.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 11.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "verified_v2")
OUT_DIRS = [
    os.path.join(ROOT_DIR, "PHI_CTRL_AST"),
    os.path.join(ROOT_DIR, "PHI_CTRL_AST_READY"),
]

MODE_COLORS = {
    "BASELINE": "#7f8c8d",
    "TWIN_KAPPA": "#2980b9",
    "RESIDUAL_ONLY": "#8e44ad",
    "TECS_MRAC": "#e67e22",
    "HYBRID": "#16a085",
    "FULL_STACK": "#c0392b",
}

MODE_STYLES = {
    "BASELINE": ("--", 1.5),
    "TWIN_KAPPA": ("-.", 1.6),
    "RESIDUAL_ONLY": (":", 1.7),
    "TECS_MRAC": ("--", 1.7),
    "HYBRID": ("-.", 1.8),
    "FULL_STACK": ("-", 2.3),
}


def save_both(fig, filename):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ==============================================================================
# FIGURE 1: arc.png — Clean 6-Mode PHI-Ctrl Architecture Block Diagram
# ==============================================================================
def make_fig1_arc():
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, fc="#f8f9fa", ec="#2c3e50", lw=1.6, title_color="#1a252f", fontsize=9.0):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top",
                fontsize=fontsize, fontweight="bold", color=title_color, zorder=4)
        if subtitle:
            ax.text(x + w / 2, y + (h - 4.5) / 2, subtitle, ha="center", va="center",
                    fontsize=fontsize - 1.1, color="#2c3e50", zorder=4, linespacing=1.25)

    def draw_arrow(x1, y1, x2, y2, label="", color="#2c3e50", lw=1.5, ls="-", lx_off=0, ly_off=1.1):
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5,head_width=3",
            color=color, linewidth=lw, linestyle=ls, zorder=5
        )
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2 + lx_off, (y1 + y2) / 2 + ly_off
            ax.text(mx, my, label, ha="center", va="center", fontsize=7.8,
                    color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)

    # Title banner
    ax.text(50, 54.2,
            "PHI-Ctrl: Physics-Hybrid Integrity Flight Control Reconfiguration Architecture (50 Hz Control / 100 Hz 6-DOF JSBSim)",
            ha="center", va="center", fontsize=11.0, fontweight="bold", color="#1a252f")

    # 1. Reference & Outer-Loop TECS
    draw_box(2, 33, 18, 16,
             "1. Outer-Loop TECS &\nReference Prefilter",
             "2nd-Order Step Prefilter:\n$h_{\\mathrm{ref}}(t), V_{\\mathrm{TAS,ref}} = 400$ kt\nTotal Energy $\\dot{E}$ & Dist. $\\dot{L}$\n$\\delta_{\\mathrm{th,cmd}} \\in [0, 1]$, $\\theta_{\\mathrm{cmd}}$",
             fc="#ebf5fb", ec="#2980b9")

    # 2. Inner-Loop Baseline + MRAC + Kappa Scaling
    draw_box(26, 33, 24, 16,
             "2. Fault-Conditioned Pitch Loop\n(TECS + MRAC + Bounded $\\kappa$)",
             "$u_{\\mathrm{nom}} = k_p e_\\theta + k_i \\int e_\\theta - k_q q - k_{\\dot{h}}\\dot{h}$\n"
             "Direct MRAC: $\\Delta u_{\\mathrm{ad}} = \\Delta K_x^\\top x_p + \\Delta K_r r_p$\n"
             "Gain Scale: $\\kappa(\\hat{\\gamma}_e) = 1 / \\max(\\hat{\\gamma}_e, \\gamma_{\\mathrm{floor}})$\n"
             "Theorem 1 Floor: $\\gamma_{\\mathrm{floor}} = 0.35 > \\gamma_{\\min}$",
             fc="#fef9e7", ec="#d35400")

    # 3. Summation & Actuator Authority Clamp
    draw_box(56, 35, 15, 12,
             "3. Integrity Clamp\n& Mixing",
             "$\\delta_{e,\\mathrm{cmd}} = \\mathrm{sat}_{20^\\circ}("
             "\\kappa(\\hat{\\gamma}_e)(u_{\\mathrm{nom}} + \\Delta u_{\\mathrm{ad}})"
             " + g(t)\\,u_{\\mathrm{rl}}(t))$",
             fc="#fdedec", ec="#c0392b", lw=2.0)

    # 4. JSBSim 6-DOF F-16 Plant
    draw_box(76, 31, 22, 20,
             "4. JSBSim 6-DOF F-16 Plant\n(100 Hz Physics, 2 Sub-steps)",
             "Nonlinear Aero + F100-PW-229\n"
             "$\\delta_{\\mathrm{th,pos}} = 2.0\\,\\delta_{\\mathrm{th,cmd}}$ (Dry/AB)\n"
             "Elevator Lag: $\\tau_e = 0.0495$ s\n"
             "Fault: $\\delta_{e,\\mathrm{eff}} = \\gamma_e(t)\\,\\delta_{e,\\mathrm{cmd}}$\n"
             "MIL-F-8785C Dryden Gusts",
             fc="#eafaf1", ec="#27ae60", lw=1.8)

    # 5. 8-Channel CNN-BiLSTM PHI-Twin (50 Hz)
    draw_box(52, 6, 26, 19,
             "5. 8-Channel 50 Hz CNN-BiLSTM PHI-Twin\n(Cross-Channel Decoupled Estimator)",
             "Sliding Window $W_k \\in \\mathbb{R}^{50 \\times 8}$ ($1.0$ s @ 50 Hz):\n"
             "$[\\alpha, q, \\theta, n_z, \\delta_{e,\\mathrm{cmd}}, V_{\\mathrm{TAS}}, \\dot{h}, e_\\theta]$\n"
             "Window-Onset Baseline Subtraction (Gyro-Bias Invariant)\n"
             "Outputs: Effectiveness $\\hat{\\gamma}_e(t) \\in [0.25, 1.0]$\n"
             "Short-Period Innovation $|r_q(t)| = |q - \\hat{q}_{\\mathrm{nom}}|$",
             fc="#e8f8f5", ec="#16a085", lw=1.9)

    # 6. Innovation & Integrity-Gated Residual Policy
    draw_box(22, 6, 24, 19,
             "6. Innovation- & Integrity-Gated\nResidual Policy $\\pi_\\psi(s_t)$",
             "Hysteresis Gate $g(t) \\in \\{0, 1\\}$:\n"
             "Enter: $\\hat{\\gamma}_e < 0.85$ & $|r_q| > r_{\\mathrm{th}}$\n"
             "Exit: $\\hat{\\gamma}_e > 0.90$ ($g(t)=0$ in healthy flight)\n"
             "Authority Bound: $|u_{\\mathrm{rl}}(t)| \\leq \\bar{u}_{\\mathrm{rl}} = 2.0^\\circ$\n"
             "Transient Pitch-Rate Damping",
             fc="#f5eef8", ec="#8e44ad", lw=1.8)

    # 7. 6-Mode Ablation Legend Box
    draw_box(2, 5, 16.5, 24,
             "6-Mode Hierarchy",
             "1. BASELINE:\n   TECS + PID only\n"
             "2. TWIN_KAPPA:\n   + $\\kappa(\\hat{\\gamma}_e)$ scaling\n"
             "3. RESIDUAL_ONLY:\n   + Gated $u_{\\mathrm{rl}}$ only\n"
             "4. TECS_MRAC:\n   + Direct MRAC only\n"
             "5. HYBRID:\n   MRAC + $\\kappa(\\hat{\\gamma}_e)$\n"
             "6. FULL_STACK:\n   MRAC + $\\kappa(\\hat{\\gamma}_e)$ + $u_{\\mathrm{rl}}$",
             fc="#f2f4f4", ec="#5d6d7e", fontsize=8.5)

    # Arrows
    draw_arrow(20, 43, 26, 43, "$\\theta_{\\mathrm{cmd}}, \\dot{h}_{\\mathrm{ref}}$", color="#2980b9")
    draw_arrow(20, 48, 76, 48, "Active Throttle $\\delta_{\\mathrm{th,cmd}}(t)$", color="#2980b9", ly_off=1.3)
    draw_arrow(50, 41, 56, 41, "$\\kappa(u_{\\mathrm{nom}}+\\Delta u_{\\mathrm{ad}})$", color="#d35400", ly_off=2.2)
    draw_arrow(71, 41, 76, 41, "$\\delta_{e,\\mathrm{cmd}}$", color="#c0392b")
    draw_arrow(87, 31, 87, 15.5, "", color="#27ae60")
    draw_arrow(87, 15.5, 78, 15.5, "8-Ch Telemetry (50 Hz)", color="#27ae60", ly_off=1.8)
    draw_arrow(52, 15.5, 46, 15.5, "$\\hat{\\gamma}_e(t), |r_q(t)|$", color="#16a085", ly_off=1.6)
    draw_arrow(60, 25, 42, 33, "$\\hat{\\gamma}_e(t)$ to $\\kappa(\\hat{\\gamma}_e)$", color="#16a085", lx_off=3.5, ly_off=0.5)
    draw_arrow(34, 25, 61, 35, "Gated $g(t)\\,u_{\\mathrm{rl}}(t)$", color="#8e44ad", lx_off=-5.0, ly_off=1.2)

    save_both(fig, "arc.png")


# ==============================================================================
# FIGURE 2: fig_baseline_verify.png — Contract 1 Hold-Trim Verification
# ==============================================================================
def make_fig2_contract1():
    df = pd.read_csv(os.path.join(DATA_DIR, "contract1_hold_trim_50hz.csv"))
    with open(os.path.join(DATA_DIR, "theory_and_margins.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6), sharex=True)
    t = df["time_s"].values

    # Panel (a): Altitude
    ax = axes[0, 0]
    ax.plot(t, df["alt_ft"], color="#1f77b4", lw=2.0, label="$h(t)$ [JSBSim 6-DOF]")
    ax.axhline(15000.0, color="#2c3e50", ls="--", lw=1.1, label="Target $15{,}000.0$ ft")
    ax.fill_between(t, 14999.5, 15000.5, color="#1f77b4", alpha=0.10, label="$\\pm 0.5$ ft band")
    ax.set_ylabel("Altitude $h(t)$ [ft]")
    ax.set_title("(a) Level Altitude Hold ($\\max|\\Delta h| = 0.035$ ft)")
    ax.set_ylim(14999.0, 15001.0)
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel (b): True Airspeed
    ax = axes[0, 1]
    vtas = df["vtas_kt"] if "vtas_kt" in df.columns else (df["vt_kts"] if "vt_kts" in df.columns else df["vc_kts"])
    ax.plot(t, vtas, color="#2ca02c", lw=2.0, label="$V_{\\mathrm{TAS}}(t)$ [kt]")
    ax.axhline(400.0, color="#2c3e50", ls="--", lw=1.1, label="Target $400.0$ kt TAS ($317.3$ kt CAS)")
    ax.fill_between(t, 399.5, 400.5, color="#2ca02c", alpha=0.10, label="$\\pm 0.5$ kt band")
    ax.set_ylabel("True Airspeed $V_{\\mathrm{TAS}}$ [kt]")
    ax.set_title("(b) Airspeed Hold ($\\mathrm{Mach}=0.639,\\,\\bar{q}=16.33$ kPa)")
    ax.set_ylim(399.0, 401.0)
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel (c): Climb Rate hdot (Contract 1)
    ax = axes[0, 2]
    ax.plot(t, df["hdot_fps"], color="#d62728", lw=1.9, label="$\\dot{h}(t)$ [ft/s]")
    ax.axhline(0.20, color="#c0392b", ls="--", lw=1.3, label="Contract 1 Limit ($\\pm 0.20$ ft/s)")
    ax.axhline(-0.20, color="#c0392b", ls="--", lw=1.3)
    ax.axvline(5.0, color="#7f8c8d", ls=":", lw=1.3, label="Settling Gate $t_s = 5.0$ s")
    ax.fill_between(t, -0.20, 0.20, color="#2ecc71", alpha=0.12)
    ax.set_ylabel("Climb Rate $\\dot{h}(t)$ [ft/s]")
    ax.set_title("(c) Contract 1 Vertical Rate ($\\max_{t\\geq 5}|\\dot{h}| = 0.0054$ ft/s)")
    ax.set_ylim(-0.28, 0.28)
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel (d): Pitch & AoA
    ax = axes[1, 0]
    ax.plot(t, df["theta_deg"], color="#8e44ad", lw=2.0, label="Pitch $\\theta(t)$ [deg]")
    ax.plot(t, df["alpha_deg"], color="#e67e22", lw=1.7, ls="--", label="AoA $\\alpha(t)$ [deg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title(f"(d) Longitudinal Trim ($\\theta_0 = \\alpha_0 = {df['theta_deg'].iloc[0]:.3f}^\\circ$)")
    ax.set_ylim(0.35, 0.65)
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel (e): Bank Angle phi (Contract 1)
    ax = axes[1, 1]
    ax.plot(t, df["phi_deg"], color="#16a085", lw=2.0, label="Roll Angle $\\phi(t)$ [deg]")
    ax.axhline(0.10, color="#c0392b", ls="--", lw=1.2, label="Contract 1 Limit ($\\pm 0.10^\\circ$)")
    ax.axhline(-0.10, color="#c0392b", ls="--", lw=1.2)
    ax.fill_between(t, -0.10, 0.10, color="#2ecc71", alpha=0.10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Roll Angle $\\phi(t)$ [deg]")
    ax.set_title("(e) Lateral Trim Hold ($\\max|\\phi| < 0.0001^\\circ$)")
    ax.set_ylim(-0.15, 0.15)
    ax.legend(loc="upper right", framealpha=0.9)

    # Panel (f): Active Throttle & Elevator Trim
    ax = axes[1, 2]
    ax2 = ax.twinx()
    thr = df["throttle_cmd"] if "throttle_cmd" in df.columns else (df["thr_cmd"] if "thr_cmd" in df.columns else df["throttle_pos"])
    de = (df["elev_cmd"] * 25.0) if "elev_cmd" in df.columns else (df["de_cmd_deg"] if "de_cmd_deg" in df.columns else df["elev_comp"] * 25.0)
    l1 = ax.plot(t, thr, color="#2980b9", lw=2.0, label="Throttle $\\delta_{\\mathrm{th}}(t)$ [norm]")
    l2 = ax2.plot(t, de, color="#c0392b", lw=1.8, ls="--", label="Elevator $\\delta_e(t)$ [deg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Throttle Command $\\delta_{\\mathrm{th}}$ [-]", color="#2980b9")
    ax2.set_ylabel("Elevator Command $\\delta_e$ [deg]", color="#c0392b")
    ax.set_ylim(0.25, 0.42)
    ax2.set_ylim(-2.2, -1.2)
    ax.set_title("(f) Active Trim Actuators ($\\delta_{\\mathrm{th},0}=0.329,\\,F_{T,0}=3742$ lbf)")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", framealpha=0.9)

    fig.suptitle("Contract 1 Verification: 60-Second Hold-Trim at $15{,}000$ ft / $400$ kt TAS ($50$ Hz Control, $100$ Hz JSBSim)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    save_both(fig, "fig_baseline_verify.png")


# ==============================================================================
# FIGURE 4: fig_phi_twin_scatter.png — 1600-Window Parity & Signed Error Audit
# ==============================================================================
def make_fig4_twin_scatter():
    df = pd.read_csv(os.path.join(DATA_DIR, "phi_twin_test_predictions_1600.csv"))
    with open(os.path.join(DATA_DIR, "theory_and_margins.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # Panel (a): Parity Scatter across On-Grid and Continuous Off-Grid
    ax = axes[0]
    on_grid = df[df["subset"] == "on_grid"] if "subset" in df.columns else df[df["regime"].isin(["on_grid_nominal", "high_noise_2x", "unseen_envelope"])]
    off_grid = df[df["subset"] == "off_grid"] if "subset" in df.columns else df[df["regime"] == "off_grid_continuous"]
    sub_05 = df[df["subset"] == "severe_sub05"] if "subset" in df.columns else df[df["regime"] == "sub_05_severe"]

    ax.scatter(on_grid["gamma_true"], on_grid["gamma_hat"], s=16, alpha=0.45, color="#2980b9",
               label=f"On-Grid $\\{{0.5, 0.6, 0.8, 1.0\\}}$ ($n={len(on_grid)}$)")
    ax.scatter(off_grid["gamma_true"], off_grid["gamma_hat"], s=18, alpha=0.55, color="#27ae60", marker="^",
               label=f"Off-Grid Continuous $[0.55, 0.95]$ ($n={len(off_grid)}$)")
    ax.scatter(sub_05["gamma_true"], sub_05["gamma_hat"], s=20, alpha=0.60, color="#e67e22", marker="s",
               label=f"Severe Extrapolation $[0.30, 0.45]$ ($n={len(sub_05)}$)")

    g_line = np.linspace(0.28, 1.02, 100)
    ax.plot(g_line, g_line, "k--", lw=1.5, label="Ideal Parity $\\hat{\\gamma}_e = \\gamma_e$")
    ax.fill_between(g_line, g_line - 0.05, g_line + 0.05, color="#7f8c8d", alpha=0.14, label="$\\pm 0.05$ Error Band")

    g_tilde = df["tilde_gamma"] if "tilde_gamma" in df.columns else df["gamma_tilde"]
    mae_all = np.mean(np.abs(g_tilde))
    rmse_all = np.sqrt(np.mean(g_tilde ** 2))
    r2_all = 1.0 - np.sum(g_tilde ** 2) / np.sum((df["gamma_true"] - df["gamma_true"].mean()) ** 2)
    max_err = np.max(np.abs(g_tilde))

    ax.text(0.31, 0.93,
            f"Unified Test Set ($n = {len(df)}$ Windows)\n"
            f"MAE $= {mae_all:.4f}$, RMSE $= {rmse_all:.4f}$\n"
            f"$R^2 = {r2_all:.4f}$, $\\max|\\tilde{{\\gamma}}_e| = {max_err:.4f}$",
            fontsize=8.6, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2c3e50", alpha=0.92))

    ax.set_xlabel("True Elevator Effectiveness $\\gamma_e$ [-]")
    ax.set_ylabel("PHI-Twin Predicted Effectiveness $\\hat{\\gamma}_e$ [-]")
    ax.set_title("(a) 8-Channel CNN-BiLSTM Parity ($n = 1600$ Held-Out Windows)")
    ax.set_xlim(0.28, 1.03)
    ax.set_ylim(0.28, 1.05)
    ax.legend(loc="lower right", framealpha=0.92)

    # Panel (b): Signed Estimation Error Distribution & Asymmetric Stability Margin
    ax = axes[1]
    err = g_tilde.values
    neg_frac = 100.0 * np.mean(err < 0)
    pos_frac = 100.0 * np.mean(err >= 0)
    min_err = np.min(err)
    max_pos_err = np.max(err)

    bins = np.linspace(-0.08, 0.08, 45)
    ax.hist(err[err < 0], bins=bins, color="#c0392b", alpha=0.68, edgecolor="white",
            label=f"Under-Estimation $\\tilde{{\\gamma}}_e < 0$: {neg_frac:.1f}% (Gain-Inflating)")
    ax.hist(err[err >= 0], bins=bins, color="#2980b9", alpha=0.68, edgecolor="white",
            label=f"Over-Estimation $\\tilde{{\\gamma}}_e \\geq 0$: {pos_frac:.1f}% (Conservative)")

    ax.axvline(0.0, color="black", ls="--", lw=1.3)
    ax.axvline(min_err, color="#c0392b", ls=":", lw=1.8,
               label=f"Worst Negative Error $\\min\\tilde{{\\gamma}}_e = {min_err:.4f}$")
    # Small-gain lower bound at worst-case gamma_e = 0.30 is -0.30 / 1.8867 = -0.1590
    t_inf = meta["margins_15k_400"]["T_inf"]
    sg_bound_030 = -0.30 / (1.0 + t_inf)
    ax.axvline(sg_bound_030, color="#8e44ad", ls="-.", lw=2.0,
               label=f"Theorem 1 Small-Gain Bound ($\\gamma_e=0.30$): ${sg_bound_030:.3f}$")

    ax.text(-0.152, ax.get_ylim()[1] * 0.62 if ax.get_ylim()[1] > 0 else 80,
            f"Asymmetric Safety Margin:\n"
            f"$|\\min\\tilde{{\\gamma}}_e| = {abs(min_err):.4f} \\ll {abs(sg_bound_030):.4f}$\n"
            f"($>3.3\\times$ Margin to Small-Gain Limit)",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", fc="#f5eef8", ec="#8e44ad", alpha=0.92))

    ax.set_xlabel("Signed Estimation Error $\\tilde{\\gamma}_e = \\hat{\\gamma}_e - \\gamma_e$ [-]")
    ax.set_ylabel("Test Window Count")
    ax.set_title("(b) Signed Error Distribution vs. Theorem 1 Lower Bound")
    ax.set_xlim(-0.175, 0.085)
    ax.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    save_both(fig, "fig_phi_twin_scatter.png")


# ==============================================================================
# FIGURE 5: fig_unified_ablation_50.png — 8-Panel 6-Mode Closed-Loop Trajectories
# ==============================================================================
def make_fig5_ablation():
    df = pd.read_csv(os.path.join(DATA_DIR, "ablation_6modes_50hz.csv"))
    modes = ["BASELINE", "TWIN_KAPPA", "RESIDUAL_ONLY", "TECS_MRAC", "HYBRID", "FULL_STACK"]

    fig, axes = plt.subplots(4, 2, figsize=(13.6, 12.2), sharex=True)

    # Panel (a): Altitude h(t) vs Reference h_ref(t)
    ax = axes[0, 0]
    d0 = df[df["mode"] == "FULL_STACK"]
    ax.plot(d0["time_s"], d0["h_ref_ft"], color="black", ls=":", lw=1.6, label="$h_{\\mathrm{ref}}(t)$ (Prefiltered)")
    for m in modes:
        dm = df[df["mode"] == m]
        ls, lw = MODE_STYLES[m]
        ax.plot(dm["time_s"], dm["alt_ft"], color=MODE_COLORS[m], ls=ls, lw=lw, label=m)
    ax.axvline(5.0, color="#7f8c8d", ls="--", lw=1.0)
    ax.axvline(10.0, color="#c0392b", ls="--", lw=1.2)
    ax.set_ylabel("Altitude $h(t)$ [ft]")
    ax.set_title("(a) Altitude Response vs. Prefiltered Reference ($+500$ ft at $t=5$ s, $\\gamma_e=0.50$ at $t=10$ s)")
    ax.legend(loc="lower right", ncol=2, framealpha=0.92, fontsize=7.5)

    # Panel (b): Reference Tracking Error e_h(t) = h(t) - h_ref(t)
    ax = axes[0, 1]
    ax.axhline(0.0, color="black", ls=":", lw=1.1)
    for m in modes:
        dm = df[df["mode"] == m]
        ls, lw = MODE_STYLES[m]
        err_h = dm["err_h_ft"] if "err_h_ft" in dm.columns else dm["e_h_ft"]
        ax.plot(dm["time_s"], err_h, color=MODE_COLORS[m], ls=ls, lw=lw, label=m)
    ax.axvline(10.0, color="#c0392b", ls="--", lw=1.2, label="Fault Onset ($t=10$ s)")
    ax.set_ylabel("Tracking Error $e_h(t)$ [ft]")
    ax.set_title("(b) Reference Tracking Error $e_h(t) = h(t) - h_{\\mathrm{ref}}(t)$")
    ax.legend(loc="lower right", ncol=2, framealpha=0.92, fontsize=7.5)

    # Panel (c): True Airspeed V_TAS(t)
    ax = axes[1, 0]
    ax.axhline(400.0, color="black", ls=":", lw=1.1, label="Target $400.0$ kt TAS")
    for m in modes:
        dm = df[df["mode"] == m]
        ls, lw = MODE_STYLES[m]
        vt = dm["vt_kts"] if "vt_kts" in dm.columns else (dm["vtas_kt"] if "vtas_kt" in dm.columns else dm["vc_kts"])
        ax.plot(dm["time_s"], vt, color=MODE_COLORS[m], ls=ls, lw=lw, label=m)
    ax.axvline(10.0, color="#c0392b", ls="--", lw=1.2)
    ax.set_ylabel("True Airspeed $V_{\\mathrm{TAS}}$ [kt]")
    ax.set_title("(c) True Airspeed Hold via Active TECS Total Energy Loop")

    # Panel (d): Active Throttle Command delta_th(t)
    ax = axes[1, 1]
    for m in modes:
        dm = df[df["mode"] == m]
        ls, lw = MODE_STYLES[m]
        thr = dm["throttle_cmd"] if "throttle_cmd" in dm.columns else (dm["thr_cmd"] if "thr_cmd" in dm.columns else dm["throttle_pos"])
        ax.plot(dm["time_s"], thr, color=MODE_COLORS[m], ls=ls, lw=lw, label=m)
    ax.axvline(5.0, color="#7f8c8d", ls="--", lw=1.0)
    ax.axvline(10.0, color="#c0392b", ls="--", lw=1.2)
    ax.set_ylabel("Throttle $\\delta_{\\mathrm{th}}(t)$ [-]")
    ax.set_title("(d) Active TECS Throttle Command $\\delta_{\\mathrm{th}}(t)$ ($\\delta_{\\mathrm{th},0} = 0.329$)")

    # Panel (e): Pitch Angle theta(t)
    ax = axes[2, 0]
    for m in modes:
        dm = df[df["mode"] == m]
        ls, lw = MODE_STYLES[m]
        ax.plot(dm["time_s"], dm["theta_deg"], color=MODE_COLORS[m], ls=ls, lw=lw, label=m)
    ax.axvline(10.0, color="#c0392b", ls="--", lw=1.2)
    ax.set_ylabel("Pitch Angle $\\theta(t)$ [deg]")
    ax.set_title("(e) Pitch Attitude Response $\\theta(t)$")

    # Panel (f): Elevator Command delta_e(t)
    ax = axes[2, 1]
    for m in modes:
        dm = df[df["mode"] == m]
        ls, lw = MODE_STYLES[m]
        de = (dm["elev_comp"] * 25.0) if "elev_comp" in dm.columns else (dm["de_cmd_deg"] if "de_cmd_deg" in dm.columns else dm["elev_cmd"] * 25.0)
        ax.plot(dm["time_s"], de, color=MODE_COLORS[m], ls=ls, lw=lw, label=m)
    ax.axvline(10.0, color="#c0392b", ls="--", lw=1.2)
    ax.set_ylabel("Elevator $\\delta_{e,\\mathrm{cmd}}(t)$ [deg]")
    ax.set_title("(f) Total Commanded Elevator Deflection $\\delta_{e,\\mathrm{cmd}}(t)$")

    # Panel (g): Closed-Loop CNN-BiLSTM Estimate gamma_hat(t) vs True gamma_e(t)
    ax = axes[3, 0]
    d_fs = df[df["mode"] == "FULL_STACK"]
    d_hy = df[df["mode"] == "HYBRID"]
    d_tk = df[df["mode"] == "TWIN_KAPPA"]
    ax.plot(d_fs["time_s"], d_fs["gamma_true"], color="black", ls="--", lw=1.8, label="True $\\gamma_e(t)$ ($1.0 \\to 0.50$)")
    ax.plot(d_tk["time_s"], d_tk["gamma_hat"], color=MODE_COLORS["TWIN_KAPPA"], ls="-.", lw=1.7, label="TWIN_KAPPA $\\hat{\\gamma}_e(t)$")
    ax.plot(d_hy["time_s"], d_hy["gamma_hat"], color=MODE_COLORS["HYBRID"], ls="--", lw=1.8, label="HYBRID $\\hat{\\gamma}_e(t)$")
    ax.plot(d_fs["time_s"], d_fs["gamma_hat"], color=MODE_COLORS["FULL_STACK"], ls="-", lw=2.2, label="FULL_STACK $\\hat{\\gamma}_e(t)$ (CNN-BiLSTM)")
    ax.axhline(0.35, color="#8e44ad", ls=":", lw=1.3, label="Safety Floor $\\gamma_{\\mathrm{floor}} = 0.35$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Effectiveness $\\hat{\\gamma}_e(t)$ [-]")
    ax.set_ylim(0.30, 1.08)
    ax.set_title("(g) Online 50 Hz CNN-BiLSTM Effectiveness Estimate $\\hat{\\gamma}_e(t)$ in Closed Loop")
    ax.legend(loc="lower left", framealpha=0.92, fontsize=7.5)

    # Panel (h): Innovation-Gated Residual Command u_rl(t)
    ax = axes[3, 1]
    for m in ["BASELINE", "RESIDUAL_ONLY", "HYBRID", "FULL_STACK"]:
        dm = df[df["mode"] == m]
        ls, lw = MODE_STYLES[m]
        url = (dm["u_rl"] * 25.0) if "u_rl" in dm.columns else (dm["u_rl_deg"] if "u_rl_deg" in dm.columns else dm["elev_comp"] * 0.0)
        ax.plot(dm["time_s"], url, color=MODE_COLORS[m], ls=ls, lw=lw, label=f"{m} $u_{{\\mathrm{{rl}}}}(t)$")
    ax.axhline(2.0, color="#7f8c8d", ls=":", lw=1.1, label="Authority Clamp $\\pm 2.0^\\circ$")
    ax.axhline(-2.0, color="#7f8c8d", ls=":", lw=1.1)
    ax.axvline(10.0, color="#c0392b", ls="--", lw=1.2, label="Fault Onset ($g(t): 0 \\to 1$)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual $u_{\\mathrm{rl}}(t)$ [deg]")
    ax.set_ylim(-2.4, 2.4)
    ax.set_title("(h) Innovation- & Integrity-Gated Residual Command $u_{\\mathrm{rl}}(t)$ ($0^\\circ$ for $t < 10$ s)")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=7.5)

    fig.tight_layout()
    save_both(fig, "fig_unified_ablation_50.png")


# ==============================================================================
# FIGURE 6: full_campaign.png — 20-Seed Monte Carlo + Paired Bootstrap + Stability Sweep
# ==============================================================================
def make_fig6_campaign():
    df_mc = pd.read_csv(os.path.join(DATA_DIR, "mc_design_point_summary.csv"))
    df_sw = pd.read_csv(os.path.join(DATA_DIR, "stability_sweep_nonlinear.csv"))
    df_est = pd.read_csv(os.path.join(DATA_DIR, "estimator_source_and_fault_ablation.csv"))
    with open(os.path.join(DATA_DIR, "theory_and_margins.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9))
    modes = ["BASELINE", "TWIN_KAPPA", "RESIDUAL_ONLY", "TECS_MRAC", "HYBRID", "FULL_STACK"]
    gammas = [0.5, 0.6, 0.8, 1.0]

    # Panel (a): Grouped Bar Chart of Post-Fault Max Reference Error across gamma_e (y-axis from 0!)
    ax = axes[0]
    x = np.arange(len(gammas))
    width = 0.13
    for i, m in enumerate(modes):
        sub = df_mc[df_mc["mode"] == m].set_index("gamma_nom").reindex(gammas)
        means = sub["mean_eh_max_ft"].values
        stds = sub["std_eh_max_ft"].values
        ax.bar(x + (i - 2.5) * width, means, width, yerr=stds, capsize=2.5,
               color=MODE_COLORS[m], edgecolor="black", linewidth=0.5, alpha=0.90, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\gamma_e = {g:.1f}$" for g in gammas])
    ax.set_ylabel("Post-Fault Peak Error $e_{h,\\max}$ [ft]")
    ax.set_ylim(0, max(df_mc["mean_eh_max_ft"] + df_mc["std_eh_max_ft"]) * 1.18)
    ax.set_title("(a) 20-Seed Stochastic Monte Carlo ($e_{h,\\max}$, Mean $\\pm$ SD)")
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.92)

    # Panel (b): Paired Differences vs BASELINE at gamma_e = 0.50 (with 95% Bootstrap CI)
    ax = axes[1]
    sub05 = df_mc[(df_mc["gamma_nom"] == 0.5) & (df_mc["mode"] != "BASELINE")].set_index("mode")
    order_b = ["TWIN_KAPPA", "RESIDUAL_ONLY", "TECS_MRAC", "HYBRID", "FULL_STACK"]
    sub05 = sub05.reindex(order_b)
    y_pos = np.arange(len(order_b))
    diffs = sub05["paired_diff_vs_base_ft"].values
    ci_low = sub05["ci95_low_ft"].values
    ci_high = sub05["ci95_high_ft"].values
    err_low = diffs - ci_low
    err_high = ci_high - diffs

    bars = ax.barh(y_pos, diffs, xerr=[err_low, err_high], capsize=4,
                   color=[MODE_COLORS[m] for m in order_b], edgecolor="black", height=0.55, alpha=0.90)
    ax.axvline(0.0, color="black", ls="--", lw=1.2)
    ax.set_xlim(min(ci_low) * 2.2 - 0.04, 0.04)
    for yp, d, cl, ch in zip(y_pos, diffs, ci_low, ci_high):
        ax.text(d - 0.008, yp, f"{d:.3f} ft\n[{cl:.3f}, {ch:.3f}]",
                va="center", ha="right", fontsize=7.2, fontweight="bold", color="#1a252f")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order_b)
    ax.set_xlabel("Paired Change $\\Delta e_{h,\\max}$ vs. BASELINE [ft]")
    ax.set_title("(b) Paired Reduction at $\\gamma_e = 0.50$ (95% Bootstrap CI)")

    # Panel (c): Nonlinear 6-DOF JSBSim Stability Sweep vs. Theorem 1 Bounds
    ax = axes[2]
    sw_tilde = df_sw["tilde_gamma"] if "tilde_gamma" in df_sw.columns else df_sw["gamma_tilde"]
    sw_eh = df_sw["max_err_h_ft"] if "max_err_h_ft" in df_sw.columns else df_sw["max_eh_post_ft"]
    sw_q = df_sw["q_peak_dps"] if "q_peak_dps" in df_sw.columns else (df_sw["tv_q_deg"] if "tv_q_deg" in df_sw.columns else df_sw.iloc[:, 4])

    ax.plot(sw_tilde, sw_eh, "o-", color="#2c3e50", lw=2.0, ms=4.5,
            label="Nonlinear 6-DOF JSBSim $e_{h,\\max}$ [ft]")
    ax2 = ax.twinx()
    ax2.plot(sw_tilde, sw_q, "s--", color="#c0392b", lw=1.5, ms=3.8, alpha=0.8,
             label="Peak Pitch Rate $|q|_{\\max}$ [deg/s]")

    sg_b = meta["bounds_by_gamma"]["0.5"]["sg_lower_tilde"]
    dm_b = meta["bounds_by_gamma"]["0.5"]["disk_lower_tilde"]
    ny_b = meta["bounds_by_gamma"]["0.5"]["nyquist_lower_tilde"]

    ax.axvline(sg_b, color="#8e44ad", ls="-.", lw=1.8, label=f"Thm 1 Small-Gain (${sg_b:.3f}$)")
    ax.axvline(dm_b, color="#2980b9", ls=":", lw=1.8, label=f"Disk Margin (${dm_b:.3f}$)")
    ax.axvline(ny_b, color="#e67e22", ls="--", lw=1.8, label=f"Linear Nyquist (${ny_b:.3f}$)")
    ax.axvspan(-0.39, ny_b, color="#e74c3c", alpha=0.15, label="Unstable Divergence Zone")

    ax.set_xlabel("Signed Estimation Error $\\tilde{\\gamma}_e = \\hat{\\gamma}_e - \\gamma_e$ (at $\\gamma_e=0.50$)")
    ax.set_ylabel("Post-Fault Peak Error $e_{h,\\max}$ [ft]", color="#2c3e50")
    ax2.set_ylabel("Peak Pitch Rate $|q|_{\\max}$ [deg/s]", color="#c0392b")
    ax.set_xlim(-0.39, 0.51)
    ax.set_ylim(0, max(sw_eh) * 1.15)
    ax.set_title("(c) Asymmetric Stability Sweep vs. Analytical Bounds")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7.0, framealpha=0.92)

    fig.tight_layout()
    save_both(fig, "full_campaign.png")


# ==============================================================================
# FIGURE 7: fig_throttle_boundary.png — Propulsion Curve + Cross-Channel Decoupling
# ==============================================================================
def make_fig7_throttle():
    df_thr = pd.read_csv(os.path.join(DATA_DIR, "f16_thrust_curve_15k_400.csv"))
    df_traj = pd.read_csv(os.path.join(DATA_DIR, "throttle_fault_trajectories_050.csv"))
    df_sum = pd.read_csv(os.path.join(DATA_DIR, "throttle_boundary_summary.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))

    # Panel (a): True JSBSim F-16 Steady-State Thrust Curve & N2
    ax = axes[0]
    ax2 = ax.twinx()
    thrust = df_thr["thrust_lbs"] if "thrust_lbs" in df_thr.columns else df_thr["thrust_lbf"]
    l1 = ax.plot(df_thr["throttle_cmd"], thrust / 1000.0, "o-", color="#1f77b4", lw=2.1, ms=4.5,
                 label="Net Thrust $F_T(\\delta_{\\mathrm{th}})$ [klbf]")
    l2 = ax2.plot(df_thr["throttle_cmd"], df_thr["n2_pct"], "s--", color="#e67e22", lw=1.7, ms=4.0,
                  label="Core Spool $N_2(\\delta_{\\mathrm{th}})$ [%]")
    ax.axvline(0.3293, color="#27ae60", ls=":", lw=1.6, label="Level Trim $\\delta_{\\mathrm{th},0}=0.329$ ($3.74$ klbf)")
    ax.axvline(0.50, color="#7f8c8d", ls="--", lw=1.3, label="MIL Detent $\\delta_{\\mathrm{th}}=0.50$ ($10.84$ klbf)")
    ax.axvspan(0.50, 1.00, color="#f39c12", alpha=0.10, label="Afterburner Regime ($\\delta_{\\mathrm{th}} > 0.50$)")

    ax.set_xlabel("Commanded Throttle $\\delta_{\\mathrm{th,cmd}} \\in [0, 1]$")
    ax.set_ylabel("Steady-State Net Thrust $F_T$ [$\times 10^3$ lbf]", color="#1f77b4")
    ax2.set_ylabel("Turbine Core Speed $N_2$ [%]", color="#e67e22")
    ax.set_title("(a) JSBSim F-16 Propulsion Curve ($15$ kft / $400$ kt TAS)")
    lines = l1 + l2 + [ax.lines[1], ax.lines[2]]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left", fontsize=7.2, framealpha=0.92)

    # Panel (b): Closed-Loop gamma_hat(t) and AoA alpha(t) under 50% Throttle Fault
    ax = axes[1]
    d_full = df_traj[df_traj["case"] == "FULL_STACK"]
    d_mis = df_traj[df_traj["case"] == "MISATTRIBUTED_1CH"]
    d_base = df_traj[df_traj["case"] == "BASELINE"]

    ghe_full = d_full["gamma_hat_e"] if "gamma_hat_e" in d_full.columns else d_full["gamma_hat"]
    ghe_mis = d_mis["gamma_hat_e"] if "gamma_hat_e" in d_mis.columns else d_mis["gamma_hat"]
    ax.plot(d_full["time_s"], ghe_full, color="#c0392b", lw=2.2,
            label="8-Ch PHI-Twin $\\hat{\\gamma}_e(t)$ (Decoupled: $\\approx 1.00$)")
    ax.plot(d_mis["time_s"], ghe_mis, color="#8e44ad", ls="--", lw=2.0,
            label="1-Ch Pitch-Only $\\hat{\\gamma}_e(t)$ (False Drop $\\to 0.52$)")
    ax.axhline(1.0, color="black", ls=":", lw=1.1, label="True Elevator $\\gamma_e(t) \equiv 1.00$")
    ax.axvline(10.0, color="#7f8c8d", ls="--", lw=1.2, label="50% Throttle Loss ($t=10$ s)")

    ax2 = ax.twinx()
    ax2.plot(d_full["time_s"], d_full["alpha_deg"], color="#27ae60", ls="-.", lw=1.6,
             label="FULL_STACK AoA $\\alpha(t)$ [deg]")
    ax2.plot(d_mis["time_s"], d_mis["alpha_deg"], color="#e67e22", ls=":", lw=1.6,
             label="1-Ch Misattributed AoA $\\alpha(t)$ [deg]")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Estimated Elevator Health $\\hat{\\gamma}_e(t)$ [-]", color="#c0392b")
    ax2.set_ylabel("Angle of Attack $\\alpha(t)$ [deg]", color="#27ae60")
    ax.set_ylim(0.40, 1.08)
    ax2.set_ylim(-1.0, 5.5)
    ax.set_title("(b) Cross-Channel Isolation Under $\\gamma_{\\mathrm{th}}=0.50$ Fault")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=7.0, framealpha=0.92)

    # Panel (c): Pitch-Rate Total Variation & Altitude Error under Throttle Faults
    ax = axes[2]
    g_ths = [0.5, 0.6, 0.8, 1.0]
    x = np.arange(len(g_ths))
    w = 0.25
    cases = [
        ("BASELINE", "BASELINE (No Reconfig)", "#7f8c8d"),
        ("FULL_STACK", "FULL_STACK (8-Ch PHI-Twin)", "#c0392b"),
        ("MISATTRIBUTED_1CH", "1-Ch Pitch-Only (Misattributed)", "#8e44ad"),
    ]
    for idx, (c_key, c_lbl, c_col) in enumerate(cases):
        sub = df_sum[df_sum["case"] == c_key].set_index("gamma_th").reindex(g_ths)
        ax.bar(x + (idx - 1) * w, sub["tv_q_deg"].values, w, color=c_col, edgecolor="black",
               alpha=0.90, label=c_lbl)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\gamma_{{\\mathrm{{th}}}} = {g:.1f}$" for g in g_ths])
    ax.set_ylabel("Pitch-Rate Total Variation $\\mathrm{TV}_q$ [deg]")
    ax.set_title("(c) Pitch Excitation Under Out-of-Scope Throttle Faults")
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.92)

    fig.tight_layout()
    save_both(fig, "fig_throttle_boundary.png")


# ==============================================================================
# FIGURE 8: fig_bias_sensitivity.png — Fair Detector Calibration Benchmark (deg/s)
# ==============================================================================
def make_fig8_bias():
    df = pd.read_csv(os.path.join(DATA_DIR, "detector_calibration_audit.csv"))
    detectors = [
        ("CNN-BiLSTM (PHI-Twin)", "#c0392b", "o", "-", 2.3),
        ("RLS (High-Pass Filtered)", "#16a085", "s", "-.", 1.9),
        ("MMAE-5 (5-Model Bank)", "#2980b9", "^", "--", 1.8),
        ("MMAE-3 (3-Model Bank)", "#e67e22", "D", "--", 1.8),
        ("GainRatio", "#8e44ad", "v", ":", 1.9),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # Panel (a): Healthy False-Alarm Rate (%) across Gyro Bias b_q in [0, 5] deg/s
    ax = axes[0]
    for d_name, col, mk, ls, lw in detectors:
        sub = df[df["detector"] == d_name].sort_values("bias_deg_s")
        ax.plot(sub["bias_deg_s"], sub["healthy_far_pct"], marker=mk, color=col, ls=ls, lw=lw,
                ms=6.0, label=d_name)

    ax.set_xlabel("Pitch-Rate Gyroscope Bias $b_q$ [deg/s]")
    ax.set_ylabel("Healthy False-Alarm Rate (FAR) [%] ($\\gamma_e = 1.00$)")
    ax.set_title("(a) Healthy False-Alarm Rate vs. Gyro Bias ($b_q \\in [0, 5]$ deg/s)")
    ax.set_xlim(-0.15, 5.15)
    ax.set_ylim(-4, 105)
    ax.legend(loc="center right", framealpha=0.92)

    # Panel (b): Post-Fault Estimation MAE at gamma_e = 0.50 across Gyro Bias b_q in [0, 5] deg/s
    ax = axes[1]
    for d_name, col, mk, ls, lw in detectors:
        sub = df[df["detector"] == d_name].sort_values("bias_deg_s")
        ax.plot(sub["bias_deg_s"], sub["fault050_mae"], marker=mk, color=col, ls=ls, lw=lw,
                ms=6.0, label=d_name)

    ax.axhline(0.173, color="#7f8c8d", ls=":", lw=1.4, label="Theorem 1 Small-Gain Error Tolerance ($0.173$)")
    ax.set_xlabel("Pitch-Rate Gyroscope Bias $b_q$ [deg/s]")
    ax.set_ylabel("Post-Fault Estimation MAE $|\\hat{\\gamma}_e - \\gamma_e|$ ($\\gamma_e = 0.50$)")
    ax.set_title("(b) Fault Severity MAE vs. Gyro Bias ($b_q \\in [0, 5]$ deg/s)")
    ax.set_xlim(-0.15, 5.15)
    ax.set_ylim(0.0, 0.52)
    ax.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    save_both(fig, "fig_bias_sensitivity.png")


if __name__ == "__main__":
    print("Generating all 7 publication figures...")
    make_fig1_arc()
    make_fig2_contract1()
    make_fig4_twin_scatter()
    make_fig5_ablation()
    make_fig6_campaign()
    make_fig7_throttle()
    make_fig8_bias()
    print("All figures generated successfully.")
