<div align="center">

# PHI-CTRL

### Physics-Hybrid Integrity Control — Fault-Tolerant Flight Control Architecture

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![JSBSim](https://img.shields.io/badge/Plant-JSBSim%20F--16A%206--DOF-0284c7)](https://jsbsim.sourceforge.net/)
[![Domain](https://img.shields.io/badge/Domain-Flight%20Control%20%7C%20PHM%20%7C%20Digital%20Twin-534AB7)](.)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-d97706?logo=huggingface)](https://huggingface.co/datasets/SM-Bello/Physics-Hybrid-Integrity-CTRL-Digital-Twin)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Author](https://img.shields.io/badge/author-S.%20M.%20Bello-0A66C2)](https://smbello.vercel.app)
[![Lab](https://img.shields.io/badge/lab-Penelope%20Inc.%20%7C%20PHI%20Lab-111827)](https://penelope-inc.vercel.app)

**Author:** [Mohammed Bello Sani (S. M. Bello)](https://smbello.vercel.app)  
**Lab:** [Penelope Inc. · PHI Lab](https://penelope-inc.vercel.app)  
**Affiliation:** Air Force Institute of Technology (AFIT) · Beihang M.Sc trajectory

</div>

---

## Abstract

Modern flight control loops remain vulnerable to **actuator effectiveness loss**, sensor bias, and compound faults that fixed-gain and black-box policies handle poorly. PHI-CTRL is a **physics-hybrid integrity** architecture for resilient longitudinal control under elevator-effectiveness faults, demonstrated on the **NASA-oriented JSBSim F-16A** nonlinear 6-DOF plant.

The stack pairs:

1. A **gate-passing baseline** energy-hold / pitch law (honest comparison, zero fault knowledge).
2. A **hybrid compensator** path: inverse-multiplicative gain scaling driven by estimated remaining effectiveness $\gamma$, optional **MRAC**, and an **MMAE-style effectiveness bank**.
3. A **diagnostic observer** (Luenberger-style) for bias monitoring — closed-loop injection is optional and off by default.
4. An optional **residual policy** (PPO) that applies **only** when the bank estimates degradation — not blindly.

Validation follows a verification ladder: baseline recovery gates → unified ablations across $\gamma$ → multi-seed metrics → optional FlightGear live path. This repository is **simulation software** (TRL ~3–5 path). **Hardware-in-the-loop flight and operational certification claims are out of scope.**

---

## Architecture

<img width="1536" height="1024" alt="arc" src="https://github.com/user-attachments/assets/d345e08b-970a-42fe-8db7-29c7622e922d" />


### γ convention (mandatory)

| Symbol | Meaning |
|--------|---------|
| $\gamma = 1.0$ | Healthy elevator |
| $\gamma = 0.8$ | 20% effectiveness **loss** (80% remaining) |
| $\gamma = 0.5$ | 50% effectiveness **loss** |
| $\gamma \in [0, 1]$ | Remaining effectiveness |

**Actuator degradation**

$$\delta_{\text{actual}} = \gamma \cdot \delta_{\text{command}}, \quad \gamma \in [0, 1]$$

**Hybrid fault-conditioned scaling** (concept)

$$\delta = \mathrm{clip}\left(\frac{u_{\mathrm{baseline}}}{\max(\hat\gamma,\,\gamma_{\min})},\,-1,\,1\right)$$

Positive elevator command is **nose-down** on this airframe (verified against the model’s $C_{m_{\delta_e}}$ table). Physical $\gamma$ is applied last at the plant write.

### Honest design notes

- **Observer:** diagnostic-only by default (`OBSERVER_CLOSED_LOOP=False`).
- **Detector path:** GainRatio + MMAE-style bank (not assumed CNN-BiLSTM for the primary gate).
- **Residual RL:** enabled only when fault is active **and** bank estimates degradation.
- **Baseline:** zero fault knowledge — fair comparison case.
- Staged gate: baseline pre-fault must pass before augmented cases are scored.

---

## Verification ladder

| Layer | What runs | Purpose |
|-------|-----------|---------|
| **L0** | `plant/jsbsim_plant_f16.py` | F-16A 6-DOF interface, trim ownership, throttle/elev authority |
| **L1** | `baseline_jsbsim_recovery/` | No-fault / hold-trim / elevator-fault recovery gates |
| **L2** | `phi_ctrl_unified_f16.py` | Unified ablation: baseline · hybrid · TECS/MRAC · full stack |
| **L3** | `eval/eval_multiseed.py` | Multi-seed metrics across $\gamma$ |
| **Optional** | `run_f16_live_flightgear.py` | Live visualization path (FDM null + native socket) |

Sample **PASSING** plots and CSVs live under `artifacts/`. Regenerate full trees locally; large episode datasets are on Hugging Face.

---

## Quick start

### Environment

```bash
git clone https://github.com/Sm-bello/PHI-CTRL.git
cd PHI-CTRL

python -m venv .venv && source .venv/bin/activate   # or: conda activate aerospace
pip install -r requirements.txt
# JSBSim: prefer conda-forge
# conda install -c conda-forge jsbsim
```

### 1) Baseline recovery gates

```bash
cd baseline_jsbsim_recovery
python run_baseline_recovery.py --no-fault
python run_baseline_recovery.py --hold-trim
python run_baseline_recovery.py          # elevator fault case
cd ..
```

### 2) Unified ablation (V2)

```bash
python phi_ctrl_unified_f16.py --gamma 0.5
python phi_ctrl_unified_f16.py --gamma 0.8
```

### 3) Multi-seed metrics

```bash
python eval/eval_multiseed.py --seeds 20 --gamma 1.0 0.8 0.5
```

### 4) Optional residual training / FlightGear

```bash
python scripts/train_residual_f16.py --timesteps 500000 --curriculum

# Terminal A
fgfs --aircraft=f16 --fdm=null --native-fdm=socket,in,60,,5505,udp
# Terminal B
python run_f16_live_flightgear.py
```

See `docs/ICD_HIL_FLIGHTGEAR.md` for the property contract.

---

## Repository layout

```text
PHI-CTRL/
├── README.md
├── LICENSE · CITATION.cff · requirements.txt · .gitignore
├── phi_ctrl_unified_f16.py          # L2 unified orchestrator
├── run_f16_live_flightgear.py       # optional FG live path
├── plant/                           # JSBSim F-16A + longitudinal ODE
├── controller/                      # baseline, adaptive, energy-hold, MRAC
├── detector/                        # GainRatio, MMAE bank, twin models
├── sensor_fusion/                   # diagnostic observer
├── fault_injection/                 # γ / bias injection
├── gym_env/                         # training / eval environments
├── bridge/                          # plant bridge helpers
├── baseline_jsbsim_recovery/        # L1 gate scripts
├── eval/                            # multi-seed & residual grid
├── scripts/                         # dataset, train residual, plots
├── models/                          # final residual zip + optional twin weights
├── data/                            # sample manifest only (full set on HF)
├── artifacts/                       # small PASSING samples
└── docs/                            # TRL roadmap, ICD, narrative text
```

**Not shipped (on purpose):** intermediate RL checkpoints, multi-GB result dumps, console debug logs, private Word progress reports, nested legacy C172 archives. Regenerate with scripts; frozen bulk data → Hugging Face / Zenodo.

---

## Dataset

Benchmarking telemetry and related exports:

**[Physics-Hybrid-Integrity-CTRL-Digital-Twin](https://huggingface.co/datasets/SM-Bello/Physics-Hybrid-Integrity-CTRL-Digital-Twin)** on Hugging Face.

Generate local fault episodes:

```bash
python scripts/generate_fault_dataset_f16.py
```

---

## Documentation

| Document | Content |
|----------|---------|
| [`docs/TRL_ROADMAP_PHI_CTRL.md`](docs/TRL_ROADMAP_PHI_CTRL.md) | TRL 3→5 plan and honest status |
| [`docs/ICD_HIL_FLIGHTGEAR.md`](docs/ICD_HIL_FLIGHTGEAR.md) | FlightGear / HIL property contract |
| [`docs/PHI-CTRL.txt`](docs/PHI-CTRL.txt) | Launch sequence, γ convention |
| [`artifacts/`](artifacts/) | Sample PASSING figures and CSVs |

---

## Limits (read before citing)

- **In scope:** JSBSim F-16A simulation, baseline vs hybrid ablations, multi-seed metrics, optional FG visualization, residual training scripts.
- **Out of scope:** Certified flight software, piloted flight test, formal DO-178C evidence packages, claims that residual RL is always superior to hybrid-only.
- Prefer claim language tied to **logged metrics** in `artifacts/` and regenerated runs, not marketing copy.

---

## Citation

```bibtex
@software{bello2026phictrl,
  title  = {PHI-CTRL: Physics-Hybrid Integrity Control for Fault-Tolerant Flight},
  author = {Bello, Mohammed Sani},
  year   = {2026},
  url    = {https://github.com/Sm-bello/PHI-CTRL},
  note   = {Penelope Inc. / PHI Lab — JSBSim F-16A simulation framework}
}
```

See also `CITATION.cff`. After a Zenodo release, add the DOI here.

---

## Built by

| | |
|---|---|
| **Author** | [Mohammed Bello Sani](https://smbello.vercel.app) — Aerospace Intelligence & Digital Twin Systems |
| **Lab** | [Penelope Inc. · PHI Lab](https://penelope-inc.vercel.app) |
| **Advisors (academic context)** | Prof. Samuel David Iyaghigba (Avionics), Dr. Joel Ajayi (Aircraft Structures) |
| **Institution** | Air Force Institute of Technology (AFIT), Kaduna |

Part of the PHI suite (PHI-Twin, PHI-Chain, PHI-SWARM, PHI-CTRL, and related frameworks).

---

## License

- **Code:** MIT  
- **Datasets / frozen figures on HF or Zenodo:** CC-BY-4.0 recommended  

---

## Status

Public source of truth for the **simulation** framework.  
Tag `v1.0.0` when baseline + unified gates pass on a clean clone; archive frozen metrics on Zenodo and link the DOI here.
