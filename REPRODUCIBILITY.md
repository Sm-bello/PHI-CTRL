# PHI-CTRL Reproducibility Guide

This release is organized around one entry point: `reproduce.py`.

## Quick checks

```bash
python reproduce.py --smoke
```

This verifies the baseline, loads the included PHI-Twin checkpoint, runs a small campaign using the **actual CNN-BiLSTM online inference path**, and writes provenance/manifest files.

## Full primary campaign

```bash
python reproduce.py --full --seeds 20
```

The full campaign covers the three envelope points, four effectiveness levels, five controller modes, and 20 deterministic seeds, using the PHI-Twin integrity source for augmented cases.

## Train the PHI-Twin from the supplied dataset

```bash
python scripts/train_phi_twin_detector.py --data data/phi_ctrl_f16_fault --epochs 25
```

The training script now splits by **episode before creating overlapping windows**, fits normalization statistics on training episodes only, and reports held-out episode test metrics.

## Individual stages

```bash
python reproduce.py --baseline
python reproduce.py --dataset
python reproduce.py --train-twin
python reproduce.py --campaign --seeds 20
python reproduce.py --analysis
```

## Reproducibility definition

The release controls experiment seeds and records the Git commit, Python/platform information, relevant package versions, command line, and SHA-256 hashes. Results should be compared within numerical tolerances rather than expecting bit-for-bit identity across different operating systems, CPUs, GPUs, or library builds.

## Integrity-source definitions

- `none`: no integrity compensation (`gamma_hat = 1`).
- `oracle`: perfect knowledge of the physical remaining effectiveness; reference/upper-bound condition.
- `mmae`: physics-bank estimate from `ElevEffectivenessBank`.
- `twin`: the included `models/phi_twin_cnn_bilstm.pt` is executed online through `detector/phi_twin_cnn_bilstm.py`.
- `fused`: PHI-Twin estimate blended with the MMAE estimate according to PHI-Twin confidence.

No synthetic lagged gamma estimator is used by these named integrity modes.

## Important scientific boundary

The PHI-Twin is an online learned health-estimation head. The residual policy remains an experimental augmentation. The baseline acceptance gate is evaluated before augmented cases in the unified workflow.
