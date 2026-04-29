<div align="center">

# ✈️ PHI-CTRL

### Closed-Loop Digital Twin for Self-Healing Flight Control

[![Status](https://img.shields.io/badge/Status-In_Progress-orange?style=flat-square)](.)
[![Stack](https://img.shields.io/badge/Stack-Python_|_PyTorch_|_MATLAB_|_Stable--Baselines3-blue?style=flat-square)](.)
[![Domain](https://img.shields.io/badge/Domain-Flight_Control_|_PHM-purple?style=flat-square)](.)
[![Author](https://img.shields.io/badge/Author-Sm--bello-black?style=flat-square)](https://github.com/Sm-bello)

</div>

---

## What This Is

**PHI-CTRL** is a closed-loop fault-tolerant flight control framework that extends the PHI-Twin digital twin from *monitoring* to *acting*. When a fault is detected in-flight, the system autonomously reconfigures its control strategy without human intervention — what engineers call **autonomous control reconfiguration**.

This is the natural sequel to PHI-Twin:

```
PHI-Twin (existing):    Sensors → Digital Twin → Fault Classification → 🚨 Alert
PHI-CTRL (this repo):   Sensors → Digital Twin → Fault Classification
                                                          ↓
                                                  Controller Decision
                                                          ↓
                                                  Actuator Command → Aircraft System
                                                          ↑_________________________|
```

The twin stops being a **spectator** and becomes a **co-pilot**.

---

## Self-Healing Scenarios

| Fault Injected | PHI-CTRL Response |
|---|---|
| Partial actuator failure (elevator stuck at 30%) | Redistribute control to remaining surfaces |
| Engine thrust asymmetry (one engine at 60%) | Adaptive trim update + rudder bias compensation |
| Sensor bias (pitot tube icing) | Fall back to twin-estimated airspeed |

---

## Architecture

```
Longitudinal Aircraft Model (Python ODE — pitch/altitude/speed)
        ↓
Fault Injection Module (parameter override mid-episode)
        ↓
PHI-Twin Fault Classifier (CNN-BiLSTM — detects fault type + severity)
        ↓
Adaptive Controller (MRAC → PPO/SAC Reinforcement Learning)
        ↓
Closed-Loop Recovery — altitude/pitch response plotted vs. PID baseline
```

---

## Tech Stack

| Tool | Role |
|---|---|
| `Python` | Aircraft ODE plant model |
| `PyTorch` | CNN-BiLSTM fault detector (reused from PHI-Twin) |
| `Stable-Baselines3` | PPO/SAC RL adaptive controller |
| `MRAC` | Model Reference Adaptive Control (stable baseline) |
| `Custom Gym Env` | Fault-injectable flight simulation environment |
| `Plotly` | Publication-quality recovery comparison figures |

---

## Project Structure

```
PHI-CTRL/
├── plant/              # Aircraft longitudinal ODE model
├── fault_injection/    # Fault parameter override module
├── detector/           # CNN-BiLSTM classifier (PHI-Twin derived)
├── controller/
│   ├── mrac/           # Model Reference Adaptive Control
│   └── rl/             # PPO/SAC reinforcement learning agent
├── gym_env/            # Custom Gym fault-injectable environment
├── results/            # Recovery plots, benchmark tables
├── notebooks/          # Experiments and analysis
└── docs/               # Research notes and references
```

---

## Research Contributions

1. A digital twin architecture whose fault signals are **directly consumable by a controller** — not just dashboard alerts
2. A **fault-conditioned adaptive control policy** (MRAC + RL) validated against PID baseline degradation
3. Quantitative recovery metrics: **time-to-recover**, **altitude deviation**, **control energy cost**

---

## Status

- [ ] Aircraft longitudinal plant model (Python ODE)
- [ ] Fault injection module
- [ ] CNN-BiLSTM fault classifier (adapt from PHI-Twin)
- [ ] MRAC adaptive controller implementation
- [ ] Custom Gym environment with mid-episode fault injection
- [ ] RL agent training (PPO via Stable-Baselines3)
- [ ] Recovery comparison plots (PHI-CTRL vs. baseline PID)
- [ ] AIAA SciTech 2027 abstract

---

## Part of the PHI Suite

| Component | Role |
|---|---|
| **PHI-Twin** | Fault detection (existing) |
| **PHI-CTRL** *(this repo)* | Fault-conditioned adaptive controller |
| **PHI-Suite** | End-to-end: detection → recovery |

---

## Target Publications

- **AIAA SciTech Forum 2027**
- **IEEE Transactions on Aerospace and Electronic Systems (TAES)**
- **MDPI Aerospace**

---

## Author

**Bello** | Aerospace & AI Engineer | [@Sm-bello](https://github.com/Sm-bello)

*Part of the PHI Research Portfolio — physics-informed AI for advanced aerospace systems.*
