#!/usr/bin/env python3
"""
Train PHI-Twin CNN-BiLSTM on PHI-CTRL F-16 fault dataset.

Requires a *good* dataset from generate_fault_dataset_f16.py V2
(must have fault_active==1 rows and non-crashed episodes).

Usage:
  python scripts/train_phi_twin_detector.py --data data/phi_ctrl_f16_fault
  python scripts/train_phi_twin_detector.py --data data/phi_ctrl_f16_fault --epochs 30 --window 40

Output:
  models/phi_twin_cnn_bilstm.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from detector.phi_twin_model import CNNLSTMGamma
from detector.phi_twin_cnn_bilstm import FEATURE_COLS


class WindowDataset(Dataset):
    def __init__(self, windows, gamma, fault):
        self.x = torch.from_numpy(windows.astype(np.float32))
        self.g = torch.from_numpy(gamma.astype(np.float32))
        self.f = torch.from_numpy(fault.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.g[i], self.f[i]


def build_windows(df: pd.DataFrame, window: int, stride: int):
    """Sliding windows per episode; label = last-step gamma & fault_active."""
    xs, gs, fs = [], [], []
    for eid, g in df.groupby("episode_id"):
        g = g.sort_values("time_s")
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        gamma = g["gamma_remaining"].to_numpy(dtype=np.float32)
        fault = g["fault_active"].to_numpy(dtype=np.float32)
        if len(feats) < window:
            continue
        for start in range(0, len(feats) - window + 1, stride):
            end = start + window
            xs.append(feats[start:end])
            gs.append(gamma[end - 1])
            fs.append(fault[end - 1])
    if not xs:
        raise RuntimeError(
            "No windows built. Dataset likely still broken (all crashes / no fault rows). "
            "Re-run: python scripts/generate_fault_dataset_f16.py --smoke"
        )
    return np.stack(xs), np.asarray(gs), np.asarray(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(HERE / "data" / "phi_ctrl_f16_fault"))
    ap.add_argument("--out", type=str, default=str(HERE / "models" / "phi_twin_cnn_bilstm.pt"))
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    data_dir = Path(args.data)
    ep_path = data_dir / "episodes.csv"
    if not ep_path.exists():
        raise SystemExit(f"Missing {ep_path}. Generate dataset first.")

    df = pd.read_csv(ep_path)
    print(f"[data] rows={len(df)} episodes={df.episode_id.nunique()}")
    print(f"[data] fault_active rate={df.fault_active.mean():.3f}")
    print(f"[data] gamma unique={sorted(df.gamma_remaining.unique())}")

    if df.fault_active.mean() < 0.05:
        raise SystemExit(
            "Dataset has almost no fault_active rows — do NOT train on V1 crash data.\n"
            "Regenerate with fixed generator:\n"
            "  python scripts/generate_fault_dataset_f16.py --episodes-per-gamma 40 "
            "--gammas 1.0 0.8 0.6 0.5 --out data/phi_ctrl_f16_fault"
        )

    X, y_g, y_f = build_windows(df, args.window, args.stride)
    # normalize features
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    Xn = (X - mean) / std

    n = len(Xn)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    n_val = max(1, int(args.val_frac * n))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    train_ds = WindowDataset(Xn[tr_idx], y_g[tr_idx], y_f[tr_idx])
    val_ds = WindowDataset(Xn[val_idx], y_g[val_idx], y_f[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTMGamma(n_feat=len(FEATURE_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    print(f"[train] windows={n} train={len(tr_idx)} val={len(val_idx)} device={device}")

    best_val = 1e9
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, gb, fb in train_loader:
            xb, gb, fb = xb.to(device), gb.to(device), fb.to(device)
            opt.zero_grad()
            g_hat, f_logit = model(xb)
            loss = mse(g_hat, gb) + 0.5 * bce(f_logit, fb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(xb)
        tr_loss /= max(1, len(tr_idx))

        model.eval()
        va_loss = 0.0
        mae = 0.0
        with torch.no_grad():
            for xb, gb, fb in val_loader:
                xb, gb, fb = xb.to(device), gb.to(device), fb.to(device)
                g_hat, f_logit = model(xb)
                loss = mse(g_hat, gb) + 0.5 * bce(f_logit, fb)
                va_loss += float(loss.item()) * len(xb)
                mae += float(torch.abs(g_hat - gb).sum().item())
        va_loss /= max(1, len(val_idx))
        mae /= max(1, len(val_idx))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_mae_gamma": mae})
        print(f"  epoch {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}  MAE(γ)={mae:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "window": args.window,
                    "feature_cols": FEATURE_COLS,
                    "history": history,
                },
                out,
            )
            print(f"    saved {out}")

    print("[done] Best checkpoint:", args.out)
    print("Wire-up: unified loads models/phi_twin_cnn_bilstm.pt when present.")


if __name__ == "__main__":
    main()
