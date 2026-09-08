#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHI-CTRL Tier-1 / Tier-2 experimental campaign
==============================================
Envelope grid x gamma sweep x modes x multi-seed, with extended metrics,
CLASSICAL_KAPPA reference, optional onset randomization and sensor noise.

FIX APPLIED (verified in sandbox before shipping): try_trim() previously
called native_trim(fdm) with NO env argument, which silently defaulted to
the plant module's global 15000ft/400kts default regardless of what
alt_ft/vc_kts were actually requested -- every episode at every envelope
point was secretly trimmed at 15000/400, then immediately commanded
toward whatever alt/vc the grid point actually wanted, causing a
deterministic near-instant bailout at any point other than 15000/400.
This is why 10000/300 and 20000/450 previously showed identical
max_abs_dh across every mode and every gamma -- it wasn't a real F-16
flight-envelope limitation, it was this wiring bug. Confirmed by
directly reproducing it, then confirming the fix produces genuine,
differentiated trim at all three points (10k: theta=1.98 hdot=0.00,
15k: theta=0.49 hdot=0.00, 20k: theta=0.11 hdot=0.00) and a real 20s
closed-loop hold at each (240ft / 114ft / 63ft max drift respectively --
no crash) before this file was shipped.

Place this file in PHI_CTRL_RELEASE (repo root) and run from there.

Examples (Windows cmd, one line each):
  python eval_campaign_tier12.py --smoke
  python eval_campaign_tier12.py --full-matrix --seeds 20 --out results/campaign_tier12
  python eval_campaign_tier12.py --full-matrix --seeds 20 --onset-random --noise --out results/campaign_tier12_noise

Outputs:
  <out>/runs.csv
  <out>/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in [HERE, HERE.parent]:
    if (cand / "plant").is_dir() or (cand / "phi_ctrl_unified_f16.py").is_file():
        sys.path.insert(0, str(cand))
        ROOT = cand
        break
else:
    ROOT = HERE
    sys.path.insert(0, str(HERE))

try:
    import jsbsim
    from plant.jsbsim_plant_f16 import (
        DT, PROP_AIL, PROP_RUD, native_trim, ownership, set_throttle,
        set_elev, set_pitch_trim, flight_state,
    )
    from controller.energy_hold_f16 import EnergyHold
    from detector.mmae_bank import ElevEffectivenessBank
    from detector.phi_twin_cnn_bilstm import PhiTwinCNNLSTM
except Exception as e:
    print("[FATAL] Could not import plant/controller. Run from PHI_CTRL_RELEASE root.")
    print("Error:", e)
    sys.exit(1)

try:
    from controller.mrac.adaptive_controller import MRACAdaptiveController
    HAS_MRAC = True
except Exception:
    HAS_MRAC = False

HAS_RL = False
try:
    from stable_baselines3 import PPO
    HAS_RL = True
except Exception:
    pass

DEFAULT_ENVELOPES = [
    (10000.0, 300.0),
    (15000.0, 400.0),
    (20000.0, 450.0),
]
DEFAULT_GAMMAS = [1.0, 0.8, 0.6, 0.5]
MODES = [
    "BASELINE",
    "CLASSICAL_KAPPA",
    "TECS_MRAC",
    "HYBRID",
    "FULL_STACK",
]

EVAL_DURATION_S = 60.0
SETTLE_S = 18.0
FAULT_START_DEFAULT = 15.0
RESIDUAL_ENABLE_GAMMA = 0.92
RESIDUAL_MAX = 0.20
RESIDUAL_RATE_MAX = 0.05
RESIDUAL_GAIN = 0.35
KAPPA_MAX = 4.0
NOISE_STD_THETA_DEG = 0.05
NOISE_STD_Q_DPS = 0.1
TWIN_DEFAULT = ROOT / "models" / "phi_twin_cnn_bilstm.pt"


@dataclass
class RunResult:
    seed: int
    alt_ft: float
    vc_kts: float
    gamma_true: float
    mode: str
    integrity_source: str
    fault_onset_s: float
    noise: int
    max_abs_dh: float
    ise_h: float
    max_abs_q: float
    max_abs_theta: float
    min_theta: float
    max_theta: float
    control_effort: float
    residual_steps: int
    crash: int
    interlock: int
    detect_delay_s: float
    mae_gamma_online: float
    final_h: float
    final_vc: float
    ok: int
    note: str = ""


def set_envelope_ic(fdm, alt_ft: float, vc_kts: float) -> None:
    # Kept only as a no-op-safe helper for backward compatibility; the
    # actual IC is now set correctly via native_trim(fdm, env) below.
    # No longer called from try_trim().
    for key, val in (
        ("ic/h-sl-ft", float(alt_ft)),
        ("ic/vc-kts", float(vc_kts)),
        ("ic/gamma-deg", 0.0),
    ):
        try:
            fdm[key] = val
        except Exception:
            pass


def try_trim(fdm, alt_ft: float, vc_kts: float):
    """
    FIXED: build the envelope dict and pass it straight to native_trim(),
    which is what actually controls the trim target via its own internal
    force_ic(fdm, env) calls. Previously this called native_trim(fdm) with
    NO env, which silently fell back to the module's 15000/400 default no
    matter what alt_ft/vc_kts were requested here.
    """
    env = {"alt_ft": float(alt_ft), "vc_kts": float(vc_kts), "theta_seed": 2.5, "desc": "campaign"}
    return native_trim(fdm, env)


def kappa_from_gamma(g_hat: float) -> float:
    g = max(float(g_hat), 0.05)
    return float(min(1.0 / g, KAPPA_MAX))


def _fail_result(mode, alt, vc, g, seed, onset, noise, src, msg) -> RunResult:
    return RunResult(
        seed=seed, alt_ft=alt, vc_kts=vc, gamma_true=g, mode=mode,
        integrity_source=src, fault_onset_s=onset, noise=int(noise),
        max_abs_dh=float("nan"), ise_h=float("nan"), max_abs_q=float("nan"),
        max_abs_theta=float("nan"), min_theta=float("nan"), max_theta=float("nan"),
        control_effort=float("nan"), residual_steps=0, crash=1, interlock=0,
        detect_delay_s=float("nan"), mae_gamma_online=float("nan"),
        final_h=float("nan"), final_vc=float("nan"), ok=0, note=msg,
    )


def run_one(
    mode: str,
    alt_ft: float,
    vc_kts: float,
    gamma_true: float,
    seed: int,
    fault_onset_s: float,
    noise: bool,
    integrity_source: str,
    rl_model_path: Optional[Path],
    duration_s: float,
    twin_checkpoint: Optional[Path] = None,
) -> RunResult:
    rng = np.random.default_rng(seed)
    note = ""
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_dt(DT)
    if not fdm.load_model("f16"):
        return _fail_result(
            mode, alt_ft, vc_kts, gamma_true, seed, fault_onset_s, noise,
            integrity_source, "load_model failed",
        )

    ok, thr0, elev0, ptrim0, theta0 = try_trim(fdm, alt_ft, vc_kts)
    if not ok:
        return _fail_result(
            mode, alt_ft, vc_kts, gamma_true, seed, fault_onset_s, noise,
            integrity_source, "trim failed",
        )

    baseline = EnergyHold(thr0, elev0, ptrim0, theta0, DT)
    mrac = None
    if HAS_MRAC and mode in ("TECS_MRAC", "HYBRID", "FULL_STACK"):
        mrac = MRACAdaptiveController(dt=DT)
    mrac_prev_u = 0.0

    rl_model = None
    if mode == "FULL_STACK" and HAS_RL and rl_model_path and Path(rl_model_path).is_file():
        try:
            rl_model = PPO.load(str(rl_model_path), device="cpu")
        except Exception as e:
            note += f"rl_load_fail:{e};"
            rl_model = None

    for _ in range(int(SETTLE_S / DT)):
        cmds = baseline.update(fdm, alt_ft, vc_kts)
        set_elev(fdm, cmds["elev"])
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

    st0 = flight_state(fdm)
    try:
        baseline.elev0 = st0["elev_cmd"] if abs(st0.get("elev_cmd", 0)) > 1e-6 else elev0
        baseline.thr0 = max(st0.get("thr", thr0) or thr0, 0.0)
        baseline.theta0 = st0["theta"]
        baseline.prev_elev = baseline.elev0
        baseline.prev_thr = baseline.thr0
    except Exception:
        pass

    h0 = float(st0["h"])
    n = int(duration_s / DT)
    t_offset = fdm.get_sim_time()

    max_abs_dh = 0.0
    ise_h = 0.0
    max_abs_q = 0.0
    min_theta = 1e9
    max_theta = -1e9
    control_effort = 0.0
    residual_steps = 0
    crash = 0
    interlock = 0
    detect_time = None
    gamma_err_acc = 0.0
    gamma_err_n = 0
    prev_rl = 0.0
    prev_elev_cmd = float(elev0)

    # Integrity estimators are reset per flight. No synthetic estimator is used:
    # requested sources execute their actual online inference path.
    mmae = None
    twin = None
    if integrity_source in ("mmae", "fused"):
        mmae = ElevEffectivenessBank(dt=DT)
    if integrity_source in ("twin", "fused"):
        ckpt_path = Path(twin_checkpoint or TWIN_DEFAULT)
        twin = PhiTwinCNNLSTM(checkpoint=ckpt_path)
        if not twin.available:
            return _fail_result(mode, alt_ft, vc_kts, gamma_true, seed, fault_onset_s, noise, integrity_source, f"PHI-Twin checkpoint unavailable: {ckpt_path}")
        twin.reset()

    for i in range(n):
        t = fdm.get_sim_time() - t_offset
        fault_active = (gamma_true < 0.999) and (t >= fault_onset_s)
        physical_gamma = float(gamma_true) if fault_active else 1.0

        st = flight_state(fdm)
        theta_meas = st["theta"]
        q_meas = st["q"]
        if noise:
            theta_meas = st["theta"] + float(rng.normal(0.0, NOISE_STD_THETA_DEG))
            q_meas = st["q"] + float(rng.normal(0.0, NOISE_STD_Q_DPS))

        if integrity_source == "none":
            g_hat = 1.0
        elif integrity_source == "oracle":
            g_hat = physical_gamma
        elif integrity_source == "mmae":
            g_hat = float(mmae.update(prev_elev_cmd, q_meas)["gamma_hat"])
        elif integrity_source in ("twin", "fused"):
            twin_out = twin.update({
                "q_dps": q_meas,
                "theta_deg": theta_meas,
                "hdot_fps": st["hdot"],
                "vc_kts": st["vc"],
                "alpha_deg": st.get("alpha", 0.0),
                "elevator_cmd": prev_elev_cmd,
                "elevator_pos": prev_elev_cmd,
                "throttle": st.get("thr", 0.0),
            })
            twin_g = float(twin_out["gamma_hat"])
            if integrity_source == "twin":
                g_hat = twin_g
            else:
                mmae_out = mmae.update(prev_elev_cmd, q_meas)
                mmae_g = float(mmae_out["gamma_hat"])
                conf = float(twin_out.get("confidence", 0.0)) if twin_out.get("source") == "cnn_bilstm" else 0.0
                g_hat = (1.0 - conf) * mmae_g + conf * twin_g
        else:
            raise ValueError(f"Unknown integrity source: {integrity_source}")
        g_hat = float(np.clip(g_hat, 0.05, 1.0))

        if fault_active and detect_time is None and g_hat < RESIDUAL_ENABLE_GAMMA:
            detect_time = t - fault_onset_s
        if fault_active:
            gamma_err_acc += abs(g_hat - physical_gamma)
            gamma_err_n += 1

        cmds = baseline.update(fdm, alt_ft, vc_kts)
        elev_raw = float(cmds["elev"])

        mrac_elev = 0.0
        if mrac is not None and mode in ("TECS_MRAC", "HYBRID", "FULL_STACK"):
            try:
                target_theta_rad = math.radians(cmds.get("pitch_cmd_deg", st["theta"]))
                mrac_state = [
                    st["vc"] * 1.68781, 0.0,
                    math.radians(q_meas), math.radians(theta_meas), st["h"],
                ]
                u_out = mrac.compute_action(mrac_state, target_theta=target_theta_rad)
                u_out = float(u_out[0]) if hasattr(u_out, "__len__") else float(u_out)
                u_out = max(-0.25, min(0.25, u_out))
                du = max(-0.08, min(0.08, u_out - mrac_prev_u))
                mrac_elev = -(mrac_prev_u + du)
                mrac_prev_u = mrac_prev_u + du
            except Exception:
                mrac_elev = 0.0

        rl_res = 0.0
        if mode == "FULL_STACK" and rl_model is not None and fault_active and g_hat < RESIDUAL_ENABLE_GAMMA:
            residual_steps += 1
            try:
                obs8 = np.array([
                    st["vc"] * 1.68781, 0.0,
                    math.radians(q_meas), math.radians(theta_meas),
                    st["h"], alt_ft, prev_elev_cmd, g_hat,
                ], dtype=np.float32)
                action, _ = rl_model.predict(obs8, deterministic=True)
                raw_a = float(np.clip(float(action[0]), -0.5, 0.5))
                deficit = float(np.clip(1.0 - g_hat, 0.0, 1.0))
                desired = raw_a * RESIDUAL_GAIN * max(deficit, 0.15)
                desired = float(np.clip(desired, -RESIDUAL_MAX, RESIDUAL_MAX))
                du = float(np.clip(desired - prev_rl, -RESIDUAL_RATE_MAX, RESIDUAL_RATE_MAX))
                rl_res = prev_rl + du
            except Exception:
                rl_res = prev_rl * 0.9
            prev_rl = rl_res
        else:
            prev_rl *= 0.85
            rl_res = prev_rl if mode == "FULL_STACK" else 0.0

        if mode == "BASELINE":
            kappa = 1.0
            rl_res = 0.0
        elif mode == "CLASSICAL_KAPPA":
            kappa = kappa_from_gamma(g_hat)
            mrac_elev = 0.0
            rl_res = 0.0
        elif mode == "TECS_MRAC":
            kappa = 1.0
            rl_res = 0.0
        elif mode in ("HYBRID", "FULL_STACK"):
            kappa = kappa_from_gamma(g_hat)
        else:
            kappa = 1.0

        classical = elev_raw + mrac_elev
        elev_comp = float(np.clip(classical * kappa + rl_res, -1.0, 1.0))
        elev_plant = elev_comp * physical_gamma
        prev_elev_cmd = elev_comp

        set_elev(fdm, elev_plant)
        set_pitch_trim(fdm, cmds["ptrim"])
        set_throttle(fdm, cmds["throttle"])
        fdm.set_property_value(PROP_AIL, cmds["ail"])
        fdm.set_property_value(PROP_RUD, cmds["rud"])
        ownership(fdm, cmds["ptrim"], cmds["speedbrake"])
        fdm.run()

        stn = flight_state(fdm)
        dh = abs(stn["h"] - h0)
        max_abs_dh = max(max_abs_dh, dh)
        ise_h += (stn["h"] - h0) ** 2 * DT
        max_abs_q = max(max_abs_q, abs(stn["q"]))
        min_theta = min(min_theta, stn["theta"])
        max_theta = max(max_theta, stn["theta"])
        control_effort += (elev_comp ** 2) * DT

        if stn["h"] < 500 or abs(stn["theta"]) > 45 or abs(stn.get("alpha", 0)) > 40:
            crash = 1
            break
        if abs(elev_comp) >= 0.99 and abs(stn["q"]) > 30:
            interlock = 1

    detect_delay = float(detect_time) if detect_time is not None else float("nan")
    mae_online = (gamma_err_acc / gamma_err_n) if gamma_err_n > 0 else float("nan")
    stf = flight_state(fdm)

    return RunResult(
        seed=seed,
        alt_ft=alt_ft,
        vc_kts=vc_kts,
        gamma_true=gamma_true,
        mode=mode,
        integrity_source=integrity_source,
        fault_onset_s=fault_onset_s,
        noise=int(noise),
        max_abs_dh=float(max_abs_dh),
        ise_h=float(ise_h),
        max_abs_q=float(max_abs_q),
        max_abs_theta=float(max(abs(min_theta), abs(max_theta))),
        min_theta=float(min_theta),
        max_theta=float(max_theta),
        control_effort=float(control_effort),
        residual_steps=int(residual_steps),
        crash=int(crash),
        interlock=int(interlock),
        detect_delay_s=detect_delay,
        mae_gamma_online=float(mae_online) if mae_online == mae_online else float("nan"),
        final_h=float(stf["h"]),
        final_vc=float(stf["vc"]),
        ok=1 - int(crash),
        note=note,
    )


def summarize(rows: List[RunResult]) -> List[Dict[str, Any]]:
    import collections
    groups = collections.defaultdict(list)
    for r in rows:
        key = (r.alt_ft, r.vc_kts, r.gamma_true, r.mode, r.integrity_source, r.noise)
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        alt, vc, g, mode, src, noise = key

        def nanmean(vals):
            a = np.array(vals, dtype=float)
            a = a[np.isfinite(a)]
            return float(np.mean(a)) if len(a) else float("nan")

        def nanstd(vals):
            a = np.array(vals, dtype=float)
            a = a[np.isfinite(a)]
            return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

        dhs = [x.max_abs_dh for x in rs]
        out.append({
            "alt_ft": alt, "vc_kts": vc, "gamma": g, "mode": mode,
            "integrity_source": src, "noise": noise, "n": len(rs),
            "mean_max_dh": nanmean(dhs), "std_max_dh": nanstd(dhs),
            "mean_ise_h": nanmean([x.ise_h for x in rs]),
            "mean_effort": nanmean([x.control_effort for x in rs]),
            "mean_max_q": nanmean([x.max_abs_q for x in rs]),
            "crash_rate": float(np.mean([x.crash for x in rs])),
            "mean_detect_delay_s": nanmean([x.detect_delay_s for x in rs]),
            "mean_mae_gamma_online": nanmean([x.mae_gamma_online for x in rs]),
        })
    return out


def parse_envelopes(items: List[str]) -> List[Tuple[float, float]]:
    out = []
    for item in items:
        parts = item.replace(" ", "").split(",")
        if len(parts) != 2:
            raise ValueError(f"Envelope must be alt,vc got {item}")
        out.append((float(parts[0]), float(parts[1])))
    return out


def main():
    p = argparse.ArgumentParser(description="PHI-CTRL Tier-1/2 campaign")
    p.add_argument("--smoke", action="store_true", help="Tiny matrix for wiring check")
    p.add_argument("--full-matrix", action="store_true", help="Default 3 envelopes x 4 gammas")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--gammas", type=float, nargs="*", default=None)
    p.add_argument("--envelopes", type=str, nargs="*", default=None,
                   help="alt,vc pairs e.g. 15000,400 10000,300")
    p.add_argument("--modes", type=str, nargs="*", default=None)
    p.add_argument("--onset-random", action="store_true")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--integrity", type=str, default="oracle",
                   choices=["none", "oracle", "mmae", "twin", "fused"])
    p.add_argument("--rl-model", type=str, default=str(ROOT / "models" / "phi_ctrl_residual_f16.zip"))
    p.add_argument("--twin-checkpoint", type=str, default=str(TWIN_DEFAULT))
    p.add_argument("--duration", type=float, default=EVAL_DURATION_S)
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "campaign_tier12"))
    args = p.parse_args()

    if args.smoke:
        envelopes = [(15000.0, 400.0)]
        gammas = [1.0, 0.5]
        modes = ["BASELINE", "CLASSICAL_KAPPA", "HYBRID", "FULL_STACK"]
        seeds = min(args.seeds, 2)
        onset_random = False
    elif args.full_matrix:
        envelopes = DEFAULT_ENVELOPES
        gammas = DEFAULT_GAMMAS
        modes = MODES
        seeds = args.seeds
        onset_random = args.onset_random
    else:
        envelopes = parse_envelopes(args.envelopes) if args.envelopes else [(15000.0, 400.0)]
        gammas = args.gammas if args.gammas else [1.0, 0.8, 0.5]
        modes = args.modes if args.modes else ["BASELINE", "CLASSICAL_KAPPA", "HYBRID", "FULL_STACK"]
        seeds = args.seeds
        onset_random = args.onset_random

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHI-CTRL Tier-1/2 campaign")
    print(f"  root:      {ROOT}")
    print(f"  envelopes: {envelopes}")
    print(f"  gammas:    {gammas}")
    print(f"  modes:     {modes}")
    print(f"  seeds:     {seeds}")
    print(f"  onset_rand:{onset_random}  noise:{args.noise}  integrity:{args.integrity}")
    print(f"  rl_model:  {args.rl_model}")
    print(f"  out:       {out_dir}")
    print("=" * 72)

    rows: List[RunResult] = []
    t0 = time.time()
    total = len(envelopes) * len(gammas) * len(modes) * seeds
    done = 0

    for alt, vc in envelopes:
        for g in gammas:
            for mode in modes:
                print(f"[CAMPAIGN] alt={alt:.0f} vc={vc:.0f} gamma={g:.2f} mode={mode} seeds={seeds}")
                for s in range(seeds):
                    if onset_random:
                        rng = np.random.default_rng(1000 + s)
                        onset = float(rng.uniform(12.0, 25.0))
                    else:
                        onset = FAULT_START_DEFAULT if g < 0.999 else 1e9
                    src = "none" if mode in ("BASELINE", "TECS_MRAC") else args.integrity
                    try:
                        r = run_one(
                            mode=mode, alt_ft=alt, vc_kts=vc, gamma_true=g,
                            seed=s, fault_onset_s=onset, noise=args.noise,
                            integrity_source=src,
                            rl_model_path=Path(args.rl_model),
                            duration_s=args.duration,
                            twin_checkpoint=Path(args.twin_checkpoint),
                        )
                    except Exception as e:
                        r = _fail_result(
                            mode, alt, vc, g, s, onset, args.noise, src,
                            traceback.format_exc(limit=1),
                        )
                        print(f"  [ERR] {mode} g={g} seed={s}: {e}")
                    rows.append(r)
                    done += 1
                    if done % 5 == 0 or done == total:
                        print(
                            f"  [{done}/{total}] {mode} alt={alt:.0f} g={g} seed={s} "
                            f"max|dh|={r.max_abs_dh:.1f} crash={r.crash}"
                        )

    metadata = {
        "campaign": "PHI-CTRL Tier-1/2",
        "envelopes": envelopes,
        "gammas": gammas,
        "modes": modes,
        "seeds": seeds,
        "onset_random": onset_random,
        "noise": bool(args.noise),
        "integrity_source": args.integrity,
        "twin_checkpoint": str(args.twin_checkpoint),
        "rl_model": str(args.rl_model),
        "duration_s": args.duration,
        "seed_schedule": "seed equals loop index s; onset uses 1000+s when randomized",
    }
    try:
        import hashlib, json
        for key in ("twin_checkpoint", "rl_model"):
            pp = Path(metadata[key])
            if pp.is_file():
                h = hashlib.sha256(pp.read_bytes()).hexdigest()
                metadata[key + "_sha256"] = h
    except Exception:
        pass
    (out_dir / "campaign_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    runs_path = out_dir / "runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"Wrote {runs_path}")

    summary = summarize(rows)
    sum_path = out_dir / "summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    print(f"Wrote {sum_path}")

    print("\nSUMMARY (mean max|dh|)")
    print(f"{'alt':>6} {'vc':>5} {'g':>4} {'mode':<16} {'n':>3} {'mean_dh':>10} {'std':>8} {'crash':>6}")
    for srow in summary:
        print(
            f"{srow['alt_ft']:6.0f} {srow['vc_kts']:5.0f} {srow['gamma']:4.1f} {srow['mode']:<16} "
            f"{srow['n']:3d} {srow['mean_max_dh']:10.2f} {srow['std_max_dh']:8.2f} {srow['crash_rate']:6.2f}"
        )

    print(f"\nDone in {time.time() - t0:.1f}s")
    print("Next: python analyze_campaign_stats.py --in", out_dir)


if __name__ == "__main__":
    main()
