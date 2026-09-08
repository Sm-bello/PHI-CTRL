#!/usr/bin/env python3
"""Pairwise deltas vs BASELINE + bootstrap 95% CI for campaign runs.csv"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def bootstrap_mean_diff(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    diffs = [rng.choice(b, len(b), True).mean() - rng.choice(a, len(a), True).mean() for _ in range(n_boot)]
    d = np.array(diffs)
    return float(np.mean(b) - np.mean(a)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--metric", default="max_abs_dh")
    args = ap.parse_args()
    df = pd.read_csv(Path(args.in_dir) / "runs.csv")
    rows = []
    for key_vals, gdf in df.groupby(["alt_ft", "vc_kts", "gamma_true", "noise"]):
        base = gdf[gdf["mode"] == "BASELINE"][args.metric].values
        for mode, mdf in gdf.groupby("mode"):
            if mode == "BASELINE":
                continue
            d, lo, hi = bootstrap_mean_diff(base, mdf[args.metric].values)
            rows.append({
                "alt_ft": key_vals[0], "vc_kts": key_vals[1], "gamma": key_vals[2],
                "noise": key_vals[3], "mode": mode,
                "baseline_mean": float(np.nanmean(base)),
                "mode_mean": float(np.nanmean(mdf[args.metric])),
                "delta_mode_minus_baseline": d, "ci95_lo": lo, "ci95_hi": hi,
                "improves": bool(d < 0 and hi < 0) if np.isfinite(hi) else False,
            })
    out = Path(args.in_dir) / "effect_sizes.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Wrote", out)

if __name__ == "__main__":
    main()
