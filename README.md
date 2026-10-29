# PHI-CTRL: Physics-Hybrid Integrity Flight Control (F-16)

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.22218809"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22218809.svg" alt="DOI"></a>
  <img src="https://img.shields.io/badge/Journal-Aerospace%20Science%20and%20Technology%20(AST)-blue?style=for-the-badge" alt="Elsevier AST"/>
  <img src="https://img.shields.io/badge/Version-v1.3.0-green?style=for-the-badge" alt="v1.3.0"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" alt="MIT"/>
  <img src="https://img.shields.io/badge/Reproducibility-Verified-orange?style=for-the-badge" alt="Reproducibility"/>
</p>

Official implementation and verification suite for:
> **"Physics-Hybrid Integrity Flight Control for Fault-Conditioned Reconfiguration on an F-16 Aircraft: Analytical Stability Margins, Plant Verification Contracts, and Coupled Effector Misattribution"**  
> *Elsevier: Aerospace Science and Technology (AST)*  
> Authors: Mohammed Bello Sani, Haruna Ahmadu Agyo  
> Affiliation: Department of Aerospace Engineering, Air Force Institute of Technology (AFIT), Kaduna, Nigeria  
> Permanent Archive (Zenodo): [https://doi.org/10.5281/zenodo.22218809](https://doi.org/10.5281/zenodo.22218809)

---

## Architecture Overview

**PHI-Ctrl** integrates:
1. **Total Energy Control System (TECS) Baseline:** Outer-loop altitude and airspeed coordination governed by total energy rate and energy distribution channels.
2. **Inner-Loop Pitch-Rate MRAC:** Direct adaptive model reference tracking augmenting the elevator channel.
3. **8-Channel CNN-BiLSTM Sequence Estimator (PHI-Twin):** Multivariate sequence network ($n = 1600$ benchmark: $\text{MAE} = 0.0188 \pm 0.0014$, $R^2 = 0.9874$) extracting effectiveness estimates $\hat{\gamma}_e \in [0.25, 1.0]$ while decoupling aerodynamic from propulsion faults.
4. **Analytical Small-Gain Stability Boundary:** Proved sufficient stability interval $-\frac{\gamma_e}{1+\|T\|_\infty} < \tilde{\gamma}_e < \frac{\gamma_e}{\|T\|_\infty - 1}$ with verified safety floor $\gamma_{\text{floor}} = 0.35$.
5. **Gated PPO Residual Policy:** Additive transient pitch-rate damping triggered only under verified fault conditions and clamped to $|u_{\text{rl}}| \le 2.0^\circ$.
6. **Plant Verification Contracts:** Formal contracts guaranteeing JSBSim F-16 hold-trim containment ($|\dot{h}| = 0.0054\text{ ft/s} \ll 0.20\text{ ft/s}$, $|\Delta h| < 0.035\text{ ft}$) and FBW override enforcement.

---

## Quick Links

| Resource | Description | Location |
|---|---|---|
| **Source Code** | Full Git repository | [github.com/Sm-bello/PHI-CTRL](https://github.com/Sm-bello/PHI-CTRL) |
| **Zenodo Archive** | Citable DOI repository (v1.0.0 – v1.3.0) | [DOI: 10.5281/zenodo.22218809](https://doi.org/10.5281/zenodo.22218809) |
| **Concept DOI** | All versions parent DOI | [DOI: 10.5281/zenodo.22218808](https://doi.org/10.5281/zenodo.22218808) |
| **Frozen Results** | Pre-computed Monte Carlo runs & summaries | `results/` |
| **Figure Generator** | 300-DPI publication figures | `generate_all_ast_figures.py` |

---

## Installation & Setup

```bash
git clone https://github.com/Sm-bello/PHI-CTRL.git
cd PHI-CTRL
pip install -r requirements.txt
```

*Note: JSBSim 6-DOF simulation engine can be installed via conda or pip (`conda install -c conda-forge jsbsim` or `pip install jsbsim`).*

---

## One-Command Complete Reproduction

To execute the verified AST simulation suite:

```bash
# 1. Run the verified multi-stage verification suite (Contract 1, Monte Carlo, Throttle Boundary, Bias Audit)
python run_fast_verified_suite.py

# 2. Generate all 7 publication figures exactly matching manuscript specifications
python generate_all_ast_figures.py
```

Generated outputs will populate `results/` and produce all 7 manuscript figures:
- `arc.png`: System architecture of the 6-mode PHI-Ctrl framework.
- `fig_baseline_verify.png`: Contract 1 hold-trim verification telemetry ($60\text{ s}$).
- `fig_phi_twin_scatter.png`: Parity scatter and signed error distribution vs. Theorem 1 lower bound ($n = 1600$).
- `fig_unified_ablation_50.png`: 8-panel closed-loop ablation trajectories under $50\%$ elevator loss.
- `full_campaign.png`: 20-seed Monte Carlo design point (480 runs) and paired bootstrap confidence intervals.
- `fig_throttle_boundary.png`: Out-of-scope throttle ceiling audit and cross-channel decoupling.
- `fig_bias_sensitivity.png`: Closed-loop detector calibration audit across rate-gyro biases $b_q \in [0.0, 5.0]^\circ\text{/s}$.

---

## Citation

If you use PHI-CTRL in your research, please cite our software release and paper:

```bibtex
@article{SaniAgyo2026PhiCtrl,
  author  = {Mohammed Bello Sani and Haruna Ahmadu Agyo},
  title   = {Physics-Hybrid Integrity Flight Control for Fault-Conditioned Reconfiguration on an F-16 Aircraft: Analytical Stability Margins, Plant Verification Contracts, and Coupled Effector Misattribution},
  journal = {Aerospace Science and Technology},
  year    = {2026},
  note    = {Under Review}
}

@misc{SaniPhiCtrlZenodo,
  author       = {Mohammed Bello Sani and Haruna Ahmadu Agyo},
  title        = {{PHI-CTRL: Physics-Hybrid Integrity Flight Control for Fault-Conditioned Reconfiguration on an F-16 Aircraft}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.3.0},
  doi          = {10.5281/zenodo.22218809},
  url          = {https://doi.org/10.5281/zenodo.22218809}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
