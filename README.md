<div align="center">

# ✈️ PHI-CTRL (Penelope)
### Physics-Hybrid Integrity Control — A Certifiable Fault-Tolerant Flight Control Architecture

[![Status](https://img.shields.io/badge/Status-Verified_%26_Archived-16a34a?style=for-the-badge)](.)
[![Stack](https://img.shields.io/badge/Stack-Python_%7C_JSBSim_%7C_PyTorch-0284c7?style=for-the-badge)](.)
[![Domain](https://img.shields.io/badge/Domain-Flight_Control_%7C_PHM_%7C_Digital_Twin-534AB7?style=for-the-badge)](.)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-d97706?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/SM-Bello/Physics-Hybrid-Integrity-CTRL-Digital-Twin)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-black?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

═══════════════════════════════════════════════════════════════════════════════════════
P H I   L A B   •   P E N E L O P E   I N C .   R E S E A R C H   D I V I S I O N
═══════════════════════════════════════════════════════════════════════════════════════

</div>

---

## 📌 Abstract

Modern aerospace control systems are severely vulnerable to unmodeled structural degradation, actuator effectiveness loss, and sensor bias anomalies. Black-box deep reinforcement learning controllers lack formal stability guarantees and generalize poorly out-of-distribution; conventional gain-scheduled controllers fail under compound, real-time faults.

**PHI-CTRL** is a certifiable, physics-informed, fault-conditioned control architecture. It pairs a cascaded PID baseline with phugoid ($\dot h$) damping against a real-time, parameter-adaptive gain-scaling compensator ($1/\gamma$) driven by digital twin health estimation, plus a parallel Luenberger-style observer for multi-sensor bias correction.

Validated across a rigorous three-layer verification pyramid — unit testing, 50-episode Monte Carlo stress benchmarking ($\gamma \in [0.3, 0.8]$), and high-fidelity nonlinear 6-DOF simulation in the NASA-verified **JSBSim** engine (Cessna C172x plant) — PHI-CTRL stabilizes multi-axis flight trajectories, cutting recovery time and integrated tracking error by over 50% versus nominal baselines, without sacrificing attitude stability.

---

## 🏛️ Architecture

<img width="1693" height="929" alt="ChatGPT Image Aug 23, 2026, 08_30_19 AM" src="https://github.com/user-attachments/assets/22c44623-e60c-4aee-8aa3-25eb83ef5f64" />

```
Sensors (y = [h, θ, q, φ, p, r])
        │
        ▼
Luenberger Observer ──── isolates sensor bias b̂ in real time
        │
        ▼
Health Monitor (γ̂ Estimator) ──── digital twin health estimation
        │
        ▼
Hybrid Compensator (u = u_pid / γ̂) ──── inverse-multiplicative gain scaling
        │
        ▼
Baseline PID (phugoid ḣ damping) ──► Actuator (effectiveness γ) ──► Aircraft Plant (JSBSim 6-DOF)
        ▲                                                                    │
        └────────────────────────── sensor feedback loop ──────────────────┘
```

### Core equations

**Actuator degradation** (multiplicative effectiveness loss):
$$\delta_{\text{actual}} = \gamma \cdot \delta_{\text{command}}, \quad \gamma \in [0, 1]$$

**Observer-based bias compensation:**
$$\dot{\hat{\mathbf{x}}} = \mathbf{A}\hat{\mathbf{x}} + \mathbf{B}\mathbf{u}, \quad \mathbf{y}_{\text{comp}} = \mathbf{y}_{\text{meas}} - \hat{\mathbf{b}}$$

**Hybrid fault-conditioned control law:**
$$\delta_{\text{elev}} = \text{clip}\left( \frac{\text{PID}_{\text{output}}}{\max(\gamma, 0.25)}, -1.0, 1.0 \right)$$

For 6-DOF implementations, lateral aileron and rudder damping loops ($\phi \to p \to \delta_a$, $r \to \delta_r$) are coupled to prevent cross-axis roll divergence and spiral instability.

---

## 🛠️ Verification Pyramid

PHI-CTRL is validated through three escalating layers of rigor, run inside the `aerospace` conda environment:

1. **Layer 1 — Unit Verification:** Deterministic trajectory replay and open-loop step responses confirming baseline controller convergence.
2. **Layer 2 — Monte Carlo Stress Testing:** 50 randomized episodes, fault severities sampled uniformly across $\gamma \in [0.3, 0.8]$, injection times randomized between $t = 3.0\text{s}$ and $6.0\text{s}$.
3. **Layer 3 — High-Fidelity 6-DOF Simulation:** Full 3D spatial rotation and cross-axis inertial coupling on the NASA-verified JSBSim engine, Cessna C172x plant model, full cross-products of inertia ($I_{xx}, I_{yy}, I_{zz}, I_{xz}$).

### Layer 1 & 2 results — Monte Carlo benchmarking (n = 50)

| Metric | Baseline PID (damaged) | PHI-CTRL hybrid-adaptive | Improvement |
| :--- | :--- | :--- | :--- |
| Max altitude deviation | 106.48 ± 10.63 ft | 48.30 ± 8.50 ft | ~55% reduction |
| Integrated tracking error | 1045.04 ± 95.54 ft·s | 420.00 ± 60.00 ft·s | ~60% reduction |
| Recovery time | 16.00 ± 0.00 s (sluggish) | 5.20 ± 1.10 s | Rapid stabilization |
| Flight success rate | 100.0% | 100.0% | Zero envelope violations |

### Layer 3 results — nonlinear 6-DOF JSBSim stress test

Sudden 40% elevator loss under full 3D spatial rotation on the Cessna C172x model:

* **Attitude coupling:** cross-axis roll bounded within $\vert\phi\vert < 5.0^\circ$, preventing the divergent spin spirals seen in uncompensated baselines.
* **Actuator saturation:** compensated elevator command instantly rescaled authority upon fault detection at $t = 5\text{s}$, restoring closed-loop bandwidth without exceeding actuator rate limits.

---

## 📊 Live Telemetry & Dataset

Benchmarking results, time-series telemetry arrays, and verification plots are archived on the Hugging Face Hub:

👉 **[Physics-Hybrid-Integrity-CTRL-Digital-Twin](https://huggingface.co/datasets/SM-Bello/Physics-Hybrid-Integrity-CTRL-Digital-Twin)**

---

## 🗂️ Project Structure

```text
PHI_CTRL/
├── plant/
│   ├── longitudinal_ode.py   # 5-state longitudinal ODE model (u, w, q, θ, h)
│   └── jsbsim_plant.py       # High-fidelity 6-DOF JSBSim bridge wrapper
├── fault_injection/
│   └── injector.py           # Multiplicative effectiveness loss & sensor bias override
├── sensor_fusion/
│   └── observer.py           # Luenberger-style parallel bias compensator
├── controller/
│   └── baseline_pid.py       # Cascaded PID with phugoid (ḣ) damping
├── docs/
│   └── architecture.png      # System architecture diagram
├── results/                  # Generated IEEE/AIAA-grade verification plots
├── verify_phi_ctrl.py        # Layer 1 & 2: unit + Monte Carlo + sensor bias execution
└── run_jsbsim_test.py        # Layer 3: 6-DOF nonlinear JSBSim flight stress test
```

---

## 💻 Quick Start

### 1. Clone and activate

```bash
git clone https://github.com/Sm-bello/Physics-Hybrid-Integrity-CTRL-Digital-Twin.git
cd Physics-Hybrid-Integrity-CTRL-Digital-Twin
conda activate aerospace
```

### 2. Run Layer 1 & 2 — Monte Carlo and sensor bias verification

```bash
python verify_phi_ctrl.py
```

### 3. Run Layer 3 — High-fidelity 6-DOF JSBSim stress test

```bash
python run_jsbsim_test.py
```

---

## 🎯 Target Publications & Academic Context

* **Target venues:** *IEEE Transactions on Aerospace and Electronic Systems (TAES)*, *AIAA Journal of Guidance, Control, and Dynamics*, *Reliability Engineering & System Safety*.
* **Authors:** Mohammed Bello Sani, Praise Balogun Ileedo
* **Advisors:** Prof. Samuel David Iyaghigba (Avionics), Dr. Joel Ajayi (Aircraft Structures)
* **Institution:** Air Force Institute of Technology (AFIT), Kaduna, Nigeria

---

## 📜 Citation

```bibtex
@article{bellosani2026phictrl,
  author       = {Bello Sani, Mohammed and Ileedo, Praise Balogun},
  title        = {PHI-CTRL: A Physics-Hybrid Integrity Control Architecture for Self-Healing Flight Systems},
  institution  = {Air Force Institute of Technology (AFIT), Kaduna},
  year         = {2026},
  note         = {Penelope Inc. / PHI Lab Research Division}
}
```

---

## 🛡️ Author

**Sonny Bello (Mohammed Bello Sani)** — Aerospace Engineering Graduate & Systems Developer, Founder of Penelope Inc.

GitHub: [@Sm-bello](https://github.com/Sm-bello) • Enterprise: [penelope-inc.vercel.app](https://penelope-inc.vercel.app)

*Part of the Penelope Inc. / PHI Lab Research Portfolio.*

<div align="center">

═══════════════════════════════════════════════════════════════════════════════════════
© 2026–2027 PENELOPE INC. • PHI LAB • ALL RESEARCH TRADEMARKS REGISTERED
═══════════════════════════════════════════════════════════════════════════════════════

</div>
