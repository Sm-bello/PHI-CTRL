#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL: Detector bias-sensitivity on REAL closed-loop telemetry.

Supersedes eval_detector_bias_sensitivity.py's synthetic-signal generator,
which used hand-picked constants that turned out to be far more energetic
than the real plant (real healthy elevator_cmd std ~0.008 vs the synthetic
generator's 0.15 -- an order of magnitude off, confirmed against
data/phi_ctrl_f16_fault/episodes.csv). This version runs the same two
real, unmodified detector classes (GainRatioDetector, ElevEffectivenessBank)
directly over the real q_dps / elevator_cmd / gamma_remaining columns from
the 160-episode dataset already used to train the CNN-BiLSTM twin, adding
only a constant synthetic bias to q_dps to emulate a rate-gyro sensor
fault. No synthetic elevator-command or process-noise model is used.

CAVEAT (carry into the manuscript): the dataset is logged at 20 Hz
(data/phi_ctrl_f16_fault/manifest.json: log_hz=20) while the online
detectors run at 120 Hz inside the real closed loop. This is the same
temporal-rate mismatch the manuscript already discloses for the CNN-BiLSTM
twin's online path (Section on Integrity Twin, final paragraph) -- report
this experiment under the identical caveat: an execution/robustness test
at the dataset's native rate, not a validated 120 Hz result.

USAGE
-----
  python eval_detector_bias_sensitivity_real.py \
      --data data/phi_ctrl_f16_fault/episodes.csv \
      --out results/detector_bias_sensitivity_real

OUTPUT
------
  <out>/bias_sensitivity_real_runs.csv
  <out>/bias_sensitivity_real_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for cand in [HERE, HERE.parent]:
    if (cand / "detector").is_dir():
        sys.path.insert(0, str(cand))
        break

from detector.fault_detector import GainRatioDetector
from detector.mmae_bank import ElevEffectivenessBank

BIASES_DEG_S = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]


@dataclass
class Row:
    detector: str
    episode_id: str
    gamma_target: float
    bias_dps: float
    is_healthy_episode: int
    false_alarm: int
    detect_delay_s: float
    severity_mae: float


def run_gain_ratio(t, q, elev, gamma, fault_active, dt) -> dict:
    det = GainRatioDetector(dt=dt, calib_window=1.0, roll_window=0.5)
    false_alarm = 0
    detect_t = None
    onset_t = None
    sev_err_acc, sev_err_n = 0.0, 0
    for i in range(len(t)):
        st = det.update(q[i], elev[i])
        if fault_active[i] == 0:
            if st.is_faulty and det.is_calibrated:
                false_alarm = 1
        else:
            if onset_t is None:
                onset_t = t[i]
            if st.is_faulty and detect_t is None:
                detect_t = t[i] - onset_t
            sev_err_acc += abs(st.severity - gamma[i])
            sev_err_n += 1
    return dict(false_alarm=false_alarm,
                detect_delay_s=(detect_t if detect_t is not None else float("nan")),
                severity_mae=(sev_err_acc / sev_err_n) if sev_err_n else float("nan"))


def run_mmae(t, q, elev, gamma, fault_active, dt) -> dict:
    bank = ElevEffectivenessBank(dt=dt)
    false_alarm = 0
    detect_t = None
    onset_t = None
    sev_err_acc, sev_err_n = 0.0, 0
    for i in range(len(t)):
        out = bank.update(elev[i], q[i])
        flagged = bank.residual_enable(out["gamma_hat"])
        if fault_active[i] == 0:
            if flagged:
                false_alarm = 1
        else:
            if onset_t is None:
                onset_t = t[i]
            if flagged and detect_t is None:
                detect_t = t[i] - onset_t
            sev_err_acc += abs(out["gamma_hat"] - gamma[i])
            sev_err_n += 1
    return dict(false_alarm=false_alarm,
                detect_delay_s=(detect_t if detect_t is not None else float("nan")),
                severity_mae=(sev_err_acc / sev_err_n) if sev_err_n else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/phi_ctrl_f16_fault/episodes.csv")
    ap.add_argument("--out", type=str, default="results/detector_bias_sensitivity_real")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    dt = float(df.groupby("episode_id")["time_s"].apply(lambda s: s.diff().median()).median())
    print(f"Inferred dt from data: {dt:.5f}s ({1.0/dt:.1f} Hz)")

    rows: List[Row] = []
    episode_ids = df["episode_id"].unique()
    for bias in BIASES_DEG_S:
        for eid in episode_ids:
            ep = df[df["episode_id"] == eid].sort_values("time_s")
            t = ep["time_s"].to_numpy()
            q = ep["q_dps"].to_numpy() + bias
            elev = ep["elevator_cmd"].to_numpy()
            gamma = ep["gamma_remaining"].to_numpy()
            fault_active = ep["fault_active"].to_numpy()
            is_healthy = int((fault_active == 0).all())
            gamma_target = float(gamma.min())

            for name, fn in (("GainRatioDetector", run_gain_ratio),
                              ("ElevEffectivenessBank", run_mmae)):
                out = fn(t, q, elev, gamma, fault_active, dt)
                rows.append(Row(
                    detector=name, episode_id=str(eid), gamma_target=gamma_target,
                    bias_dps=bias, is_healthy_episode=is_healthy,
                    false_alarm=out["false_alarm"], detect_delay_s=out["detect_delay_s"],
                    severity_mae=out["severity_mae"],
                ))

    runs_path = out_dir / "bias_sensitivity_real_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"Wrote {runs_path}")

    import collections
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r.detector, r.bias_dps, r.gamma_target)].append(r)

    summary = []
    for (det, bias, gamma), rs in sorted(groups.items()):
        healthy_rows = [x for x in rs if x.is_healthy_episode]
        fault_rows = [x for x in rs if not x.is_healthy_episode]
        fa = float(np.mean([x.false_alarm for x in healthy_rows])) if healthy_rows else float("nan")
        dd = [x.detect_delay_s for x in fault_rows if x.detect_delay_s == x.detect_delay_s]
        sm = [x.severity_mae for x in fault_rows if x.severity_mae == x.severity_mae]
        summary.append({
            "detector": det, "bias_dps": bias, "gamma_target": gamma,
            "n_episodes": len(rs),
            "false_alarm_rate_if_healthy": fa,
            "mean_detect_delay_s": float(np.mean(dd)) if dd else float("nan"),
            "mean_severity_mae": float(np.mean(sm)) if sm else float("nan"),
        })

    sum_path = out_dir / "bias_sensitivity_real_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {sum_path}")

    print("\nFalse-alarm rate on HEALTHY episodes (gamma_target=1.0), by bias magnitude:")
    print(f"{'detector':<24}{'bias_dps':>10}{'false_alarm_rate':>18}")
    for s in summary:
        if s["gamma_target"] >= 0.999:
            print(f"{s['detector']:<24}{s['bias_dps']:>10.1f}{s['false_alarm_rate_if_healthy']:>18.2f}")

    print("\nSeverity MAE on FAULTED episodes, by gamma_target and bias magnitude:")
    print(f"{'detector':<24}{'gamma':>6}{'bias_dps':>10}{'severity_mae':>14}{'detect_delay_s':>16}")
    for s in summary:
        if s["gamma_target"] < 0.999:
            print(f"{s['detector']:<24}{s['gamma_target']:>6.1f}{s['bias_dps']:>10.1f}"
                  f"{s['mean_severity_mae']:>14.4f}{s['mean_detect_delay_s']:>16.3f}")


if __name__ == "__main__":
    main()
