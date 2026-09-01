# Artifacts (samples only)

These folders hold **passing-run samples** for orientation:

- `baseline_jsbsim_recovery_PASSING/` — baseline gate plots/logs
- `unified_f16_PASSING/` — unified ablation comparison
- `eval_multiseed/` — summary CSVs

Full result trees and large episode tensors belong on Hugging Face / Zenodo.
Regenerate with:

```bash
python baseline_jsbsim_recovery/run_baseline_recovery.py
python phi_ctrl_unified_f16.py --gamma 0.5
python eval/eval_multiseed.py --seeds 20 --gamma 1.0 0.8 0.5
```
