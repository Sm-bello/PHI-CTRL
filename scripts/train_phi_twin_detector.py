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
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    # Episode-level split BEFORE window generation to prevent leakage from overlapping windows.
    episodes = np.array(sorted(df["episode_id"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    n_test_ep = max(1, int(round(args.test_frac * len(episodes))))
    n_val_ep = max(1, int(round(args.val_frac * len(episodes))))
    if n_test_ep + n_val_ep >= len(episodes):
        n_test_ep = max(1, len(episodes) // 10)
        n_val_ep = max(1, len(episodes) // 10)
    test_eps = set(episodes[:n_test_ep])
    val_eps = set(episodes[n_test_ep:n_test_ep + n_val_ep])
    train_eps = set(episodes[n_test_ep + n_val_ep:])

    def split_df(eps):
        return df[df["episode_id"].isin(eps)].copy()

    Xtr, ytr_g, ytr_f = build_windows(split_df(train_eps), args.window, args.stride)
    Xva, yva_g, yva_f = build_windows(split_df(val_eps), args.window, args.stride)
    Xte, yte_g, yte_f = build_windows(split_df(test_eps), args.window, args.stride)

    # Fit normalization on TRAIN ONLY, then apply unchanged statistics to val/test.
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    Xte = (Xte - mean) / std

    train_ds = WindowDataset(Xtr, ytr_g, ytr_f)
    val_ds = WindowDataset(Xva, yva_g, yva_f)
    test_ds = WindowDataset(Xte, yte_g, yte_f)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTMGamma(n_feat=len(FEATURE_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    print(f"[split] episodes train={len(train_eps)} val={len(val_eps)} test={len(test_eps)}")
    print(f"[train] windows train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} device={device}")

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
        tr_loss /= max(1, len(train_ds))

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
        va_loss /= max(1, len(val_ds))
        mae /= max(1, len(val_ds))
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

    # Held-out episode test evaluation using the best checkpoint.
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_abs = []
    test_fault_true = []
    test_fault_prob = []
    with torch.no_grad():
        for xb, gb, fb in test_loader:
            xb = xb.to(device)
            g_hat, f_logit = model(xb)
            test_abs.extend(torch.abs(g_hat.cpu() - gb).numpy().tolist())
            test_fault_true.extend(fb.numpy().tolist())
            test_fault_prob.extend(torch.sigmoid(f_logit).cpu().numpy().tolist())
    test_mae = float(np.mean(test_abs)) if test_abs else float("nan")
    pred_fault = np.asarray(test_fault_prob) >= 0.5
    true_fault = np.asarray(test_fault_true) >= 0.5
    precision = float((pred_fault & true_fault).sum() / max(1, pred_fault.sum()))
    recall = float((pred_fault & true_fault).sum() / max(1, true_fault.sum()))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    ckpt["split"] = {
        "seed": args.seed,
        "train_episode_ids": sorted(map(int, train_eps)),
        "val_episode_ids": sorted(map(int, val_eps)),
        "test_episode_ids": sorted(map(int, test_eps)),
        "normalization_fit": "train_only",
    }
    ckpt["test_metrics"] = {
        "gamma_mae": test_mae,
        "fault_precision": precision,
        "fault_recall": recall,
        "fault_f1": float(f1),
        "n_test_windows": len(test_ds),
    }
    torch.save(ckpt, args.out)
    metrics_path = Path(args.out).with_suffix(".test_metrics.json")
    metrics_path.write_text(json.dumps(ckpt["test_metrics"], indent=2), encoding="utf-8")
    print(f"[test] held-out episode gamma MAE={test_mae:.5f} fault F1={f1:.3f}")
    print("[done] Best checkpoint:", args.out)
    print("[done] Test metrics:", metrics_path)


if __name__ == "__main__":
    main()
