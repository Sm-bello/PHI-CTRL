<img width="1536" height="1024" alt="arc" src="https://github.com/user-attachments/assets/9ba2d7e1-1a57-42f5-9983-6c5fc508b46b" />
# PHI-CTRL — Reproducible F-16 Integrity-Aware Control Release

This archive contains the PHI-CTRL research implementation, F-16 JSBSim plant path, elevator-effectiveness fault injection, PHI-Twin CNN-BiLSTM health estimator, adaptive/control components, residual policy, supplied telemetry dataset, and experiment scripts.

## One-command reproduction

After installing the dependencies:

```bash
python reproduce.py --smoke
```

For the primary campaign:

```bash
python reproduce.py --full --seeds 20
```

The orchestrator records environment/provenance information and creates a SHA-256 manifest under `results/reproduction/`.

## What the release actually runs

The named integrity modes in `eval_campaign_tier12.py` are explicit:

- `none` — no integrity compensation.
- `oracle` — perfect knowledge of physical remaining elevator effectiveness; reference condition.
- `mmae` — physics-bank estimate from `detector/mmae_bank.py`.
- `twin` — **actual online CNN-BiLSTM inference** from `models/phi_twin_cnn_bilstm.pt` through `detector/phi_twin_cnn_bilstm.py`.
- `fused` — confidence-weighted PHI-Twin + MMAE estimate.

The campaign no longer uses a synthetic lagged gamma estimate under these named modes.

## Key scripts

| Script | Purpose |
|---|---|
| `reproduce.py` | Top-level reproducibility orchestrator |
| `eval_campaign_tier12.py` | Multi-envelope controller/integrity campaign |
| `eval/eval_multiseed.py` | Multi-seed evaluation with deterministic seed schedule |
| `scripts/generate_fault_dataset_f16.py` | F-16 fault telemetry generation |
| `scripts/train_phi_twin_detector.py` | CNN-BiLSTM training with episode-level split and held-out test |
| `analyze_campaign_stats.py` | Bootstrap pairwise campaign effects |
| `phi_ctrl_unified_f16.py` | Single unified controller experiment and baseline gate |

## PHI-Twin training

```bash
python scripts/train_phi_twin_detector.py \
  --data data/phi_ctrl_f16_fault \
  --epochs 25
```

The training script splits **episodes before sliding-window construction** to prevent leakage between overlapping windows. Feature normalization is fitted on training episodes only. The resulting checkpoint stores the split definition and held-out test metrics.

## Primary campaign

The default full matrix is:

- 10,000 ft / 300 kt
- 15,000 ft / 400 kt
- 20,000 ft / 450 kt
- γ = 1.0, 0.8, 0.6, 0.5
- BASELINE, CLASSICAL_KAPPA, TECS_MRAC, HYBRID, FULL_STACK
- configurable number of seeds (20 recommended for the primary result)

Example:

```bash
python eval_campaign_tier12.py --full-matrix --seeds 20 --integrity twin --out results/campaign_twin
```

Noise and randomized onset can be enabled separately:

```bash
python eval_campaign_tier12.py --full-matrix --seeds 20 --integrity twin --noise --onset-random --out results/campaign_twin_robust
```

## Baseline gate

The release preserves the principle that the baseline is checked before augmented cases. A failed baseline should be treated as a reproduction failure rather than hidden by augmented-controller results.

## Reproducibility files

See `REPRODUCIBILITY.md` for the exact workflow and scientific interpretation. `CITATION.cff` is included for citation tooling.

## Limitations

This is research simulation software and is not flight-certified. Numerical results can vary across operating systems, CPU/GPU backends, and library versions; the release therefore defines reproducibility through controlled seeds, provenance, hashes, and numerical/experimental acceptance criteria rather than bit-for-bit identity.
