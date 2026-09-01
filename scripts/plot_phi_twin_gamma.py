#!/usr/bin/env python3
"""
PHI-Twin true γ vs γ̂ scatter (publication figure).

Run from repo root (PHI_CTRL_RELEASE / PHI_CTRL_WORKING):

  python plot_phi_twin_gamma.py

Or:

  python scripts/plot_phi_twin_gamma.py

Outputs:
  results_f16/phi_twin_gamma_scatter.png
  prints MAE(γ)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
# allow running from repo root or from scripts/
ROOT = HERE if (HERE / "detector").is_dir() else HERE.parent
sys.path.insert(0, str(ROOT))

from detector.phi_twin_cnn_bilstm import FEATURE_COLS
from detector.phi_twin_model import CNNLSTMGamma


def main():
    ckpt_path = ROOT / "models" / "phi_twin_cnn_bilstm.pt"
    data_path = ROOT / "data" / "phi_ctrl_f16_fault" / "episodes.csv"
    out_dir = ROOT / "results_f16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "phi_twin_gamma_scatter.png"

    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")
    if not data_path.exists():
        raise SystemExit(f"Missing dataset: {data_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    df = pd.read_csv(data_path)
    W = int(ckpt.get("window", 40))
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)

    model = CNNLSTMGamma(n_feat=len(FEATURE_COLS))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    yt, yp = [], []
    for _eid, g in df.groupby("episode_id"):
        g = g.sort_values("time_s")
        X = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = g["gamma_remaining"].to_numpy(dtype=np.float32)
        if len(X) < W:
            continue
        # non-overlapping windows for a clean scatter
        for i in range(0, len(X) - W + 1, W):
            x = (X[i : i + W] - mean) / np.maximum(std, 1e-6)
            t = torch.from_numpy(x[None, ...])
            with torch.no_grad():
                gh, _ = model(t)
            yp.append(float(gh.item()))
            yt.append(float(y[i + W - 1]))

    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    if len(yt) == 0:
        raise SystemExit("No windows built — check dataset length vs window size.")

    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    print(f"windows={len(yt)}  MAE(γ)={mae:.4f}  RMSE(γ)={rmse:.4f}")

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(yt, yp, s=10, alpha=0.35, c="#2563eb", edgecolors="none")
    lo = min(0.45, float(yt.min()) - 0.02, float(yp.min()) - 0.02)
    hi = max(1.02, float(yt.max()) + 0.02, float(yp.max()) + 0.02)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal γ̂ = γ")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True γ (remaining effectiveness)")
    ax.set_ylabel("PHI-Twin γ̂")
    ax.set_title(f"PHI-Twin: true vs estimated γ\nMAE={mae:.3f}  n={len(yt)}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
