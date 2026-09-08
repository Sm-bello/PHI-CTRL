#!/usr/bin/env python3
"""PHI-Twin offline MAE / scatter — does not require FEATURE_COLS export."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for cand in [HERE, HERE.parent]:
    sys.path.insert(0, str(cand))

DEFAULT_FEATURES = [
    "theta_deg", "q_dps", "vc_kts", "h_ft", "elev_cmd", "thr",
    "hdot_fps", "alpha_deg",
]

def resolve_features(ckpt, df_cols):
    if isinstance(ckpt, dict):
        for key in ("feature_cols", "FEATURE_COLS", "features"):
            if key in ckpt and ckpt[key]:
                cols = list(ckpt[key])
                if all(c in df_cols for c in cols):
                    return cols
    aliases = {
        "theta_deg": ["theta_deg", "theta", "pitch_deg"],
        "q_dps": ["q_dps", "q", "pitch_rate_dps"],
        "vc_kts": ["vc_kts", "vc", "airspeed_kts"],
        "h_ft": ["h_ft", "h", "alt_ft", "altitude_ft"],
        "elev_cmd": ["elev_cmd", "elevator", "elev"],
        "thr": ["thr", "throttle", "throttle_pos"],
        "hdot_fps": ["hdot_fps", "hdot", "climb_rate_fps"],
        "alpha_deg": ["alpha_deg", "alpha", "aoa_deg"],
    }
    cols = []
    for feat in DEFAULT_FEATURES:
        if feat in df_cols:
            cols.append(feat)
            continue
        for alt in aliases.get(feat, []):
            if alt in df_cols:
                cols.append(alt)
                break
    return cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phi_ctrl_f16_fault")
    ap.add_argument("--model", default="models/phi_twin_cnn_bilstm.pt")
    ap.add_argument("--out", default="results/detector_metrics")
    ap.add_argument("--window", type=int, default=40)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_path = data_dir / "episodes.csv"
    if not ep_path.is_file():
        print("Missing", ep_path)
        return

    df = pd.read_csv(ep_path)
    print("Loaded", ep_path, "rows=", len(df))
    print("Columns:", list(df.columns))

    if "gamma_remaining" in df.columns:
        dist = df.groupby("gamma_remaining").size().reset_index(name="n_rows")
        dist.to_csv(out_dir / "gamma_distribution.csv", index=False)
        print(dist)

    model_path = Path(args.model)
    if not model_path.is_file():
        print("No model at", model_path)
        return

    import torch
    try:
        from detector.phi_twin_model import CNNLSTMGamma
    except Exception as e:
        print("CNNLSTMGamma import failed:", e)
        return

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    W = int(ckpt.get("window", args.window))
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    n_feat = int(mean.shape[0])

    feat_cols = resolve_features(ckpt, set(df.columns))
    if len(feat_cols) != n_feat:
        skip = {"episode_id", "time_s", "gamma_remaining", "gamma", "fault", "label"}
        numeric = [c for c in df.columns if c not in skip and np.issubdtype(df[c].dtype, np.number)]
        feat_cols = numeric[:n_feat]
        print("WARN: feature list length mismatch; using", feat_cols)

    print("Using features:", feat_cols, "n_feat=", n_feat, "window=", W)
    model = CNNLSTMGamma(n_feat=n_feat)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    if isinstance(state, dict) and any(str(k).startswith("model.") for k in state):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    id_col = "episode_id" if "episode_id" in df.columns else None
    t_col = "time_s" if "time_s" in df.columns else None
    y_col = "gamma_remaining" if "gamma_remaining" in df.columns else "gamma"

    yt, yp = [], []
    groups = df.groupby(id_col) if id_col else [(0, df)]
    for eid, g in groups:
        if t_col:
            g = g.sort_values(t_col)
        X = g[feat_cols].to_numpy(np.float32)
        y = g[y_col].to_numpy(np.float32)
        if len(X) < W:
            continue
        step = max(W // 2, 1)
        for i in range(0, len(X) - W + 1, step):
            x = (X[i:i + W] - mean) / np.maximum(std, 1e-6)
            with torch.no_grad():
                out = model(torch.from_numpy(x[None, ...]))
            gh = out[0] if isinstance(out, (tuple, list)) else out
            yp.append(float(gh.detach().cpu().reshape(-1)[0].item()))
            yt.append(float(y[i + W - 1]))

    yt, yp = np.array(yt), np.array(yp)
    if len(yt) == 0:
        print("No windows evaluated — check feature columns vs training.")
        return

    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} n={len(yt)}")

    rows = []
    for gval in sorted(set(np.round(yt, 2))):
        m = np.isclose(yt, gval, atol=0.05)
        if m.sum():
            rows.append({"gamma": float(gval), "n": int(m.sum()),
                         "mae": float(np.mean(np.abs(yp[m] - yt[m])))})
    pd.DataFrame(rows).to_csv(out_dir / "mae_by_gamma.csv", index=False)
    pd.DataFrame([{"mae": mae, "rmse": rmse, "n_windows": len(yt)}]).to_csv(
        out_dir / "summary.csv", index=False)
    pd.DataFrame({"gamma_true": yt, "gamma_hat": yp}).to_csv(
        out_dir / "scatter_points.csv", index=False)
    print("Wrote metrics to", out_dir)

if __name__ == "__main__":
    main()
