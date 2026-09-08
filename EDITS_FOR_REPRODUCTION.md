# Reproduction-release edits

## `reproduce.py` — added
- Single entry point for baseline, dataset, PHI-Twin training, campaign, and analysis.
- `--smoke` performs a small wiring/reproduction check.
- `--full --seeds 20` runs the primary campaign.
- Captures Python/platform/package/Git provenance.
- Generates a SHA-256 manifest for release artifacts.

## `scripts/train_phi_twin_detector.py` — revised
- Added deterministic `--seed`.
- Added episode-level train/validation/test split.
- Split happens before overlapping windows are created.
- Normalization statistics are fitted on training data only.
- Added held-out test evaluation.
- Checkpoint now stores split IDs and test metrics.
- Writes `<checkpoint>.test_metrics.json`.

## `eval/eval_multiseed.py` — revised
- Removed Python's process-randomized `hash(mode)` from seed construction.
- Added fixed controller-specific seed offsets for repeatable experiments.

## `eval_campaign_tier12.py` — revised
- Added real `mmae`, `twin`, and `fused` integrity implementations.
- `twin` now executes the supplied CNN-BiLSTM online, window by window.
- `mmae` uses the actual `ElevEffectivenessBank`.
- `fused` combines actual PHI-Twin and MMAE estimates using PHI-Twin confidence.
- `oracle` remains a perfect-information reference condition.
- Removed the synthetic lagged gamma estimator from named integrity modes.
- Added `--twin-checkpoint`.
- Campaign writes `campaign_metadata.json`, including model hashes.

## `README.md` — revised
- Rewritten around the reproducibility workflow.
- Corrected the detector description so the supplied CNN-BiLSTM is represented accurately.
- Documents integrity-source semantics and primary campaign matrix.

## `REPRODUCIBILITY.md` — added
- Exact commands, experiment definitions, and interpretation of reproducibility.

## `CITATION.cff` — added
- Machine-readable citation metadata for the software release.

## `requirements.txt` — revised
- Added JSBSim, PyTorch, Stable-Baselines3, and Gymnasium as explicit dependencies.

## Validation performed before packaging
- Python syntax compilation passed for all modified Python scripts.
- Supplied PHI-Twin checkpoint successfully loaded and executed through the online detector interface.
- Full JSBSim campaign was **not executed in this build environment because JSBSim is not installed here**; the release therefore does not claim a fresh campaign result from this environment.
