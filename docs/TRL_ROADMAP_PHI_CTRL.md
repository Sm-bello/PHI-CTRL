# PHI-CTRL Technology Readiness Roadmap

**System:** Physics-Hybrid Integrity Control (PHI-CTRL)  
**Focus:** Resilient longitudinal control under elevator-effectiveness faults  
**Plant:** JSBSim F-16A Block-32 (6-DOF)  
**Profile:** NASA-oriented TRL maturation (simulation → relevant environment → HIL)

---

## 1. TRL definitions (applied to this work)

| TRL | Meaning for PHI-CTRL | Status |
|-----|----------------------|--------|
| 3 | Analytical / experimental critical function | **Done** — architecture + plant interface + baseline laws |
| 4 | Component validation in laboratory (high-fidelity sim) | **Done** — gate-passing baseline; unified cases A–C; metrics logged |
| 5 | Component/subsystem validation in **relevant environment** | **In progress** — FG live path + ICD; residual needs gated improvement |
| 6 | Prototype demonstration in relevant environment | Future — HIL bench or piloted sim campaign |
| 7+ | Flight / operational | Out of scope for this package |

---

## 2. Evidence already in this repository (TRL 4)

| Artifact | Location | Claim supported |
|----------|----------|-----------------|
| Level trim with gamma-bias correction | `plant/jsbsim_plant_f16.py` | Plant IC is genuinely level (ḣ≈0), not silent climb |
| EnergyHold baseline 60 s hold | `results/baseline_jsbsim_recovery_PASSING/` | Pitch θ ∈ [−0.1°, +0.4°]; survives 20% elev loss |
| Unified orchestrator cases | `results/unified_f16/` | BASELINE / TECS_MRAC / HYBRID metrics, no crash |
| Residual training pipeline | `scripts/train_residual_f16.py`, `gym_env/jsbsim_phi_ctrl_env_f16.py` | PPO residual on real F-16 plant; trim grid + curriculum |
| Safety envelope primitives | rate limits, throttle floor, fbw-override, interlock hooks | Design-assurance style constraints in the loop |

### Quantitative snapshot (unified 60 s runs)

| Case | max \|Δh\| (ft) | min θ (deg) | Crash | Notes |
|------|-----------------|-------------|-------|-------|
| BASELINE | ~579 | −0.1 | No | Gate PASS |
| TECS_MRAC | ~572 | −0.1 | No | Comparable to baseline |
| HYBRID | ~564 | −0.1 | No | Slight improvement vs baseline |
| FULL_STACK | ~2501 | +0.2 | No | Residual active; **larger** excursion — do not claim superiority yet |

---

## 3. Gaps to close for TRL 5

1. **Relevant environment**  
   - Execute and log FlightGear-in-the-loop (or batch FG) against `docs/ICD_HIL_FLIGHTGEAR.md`.  
   - Same property writes as pure JSBSim; metrics still from JSBSim truth.

2. **Integrity → reconfiguration evidence**  
   - MMAE-style detector bank (`detector/mmae_bank.py`) must drive residual enable / gain schedule.  
   - Report detection delay, false alarm rate, and missed detection vs γ ∈ {1.0, 0.8, 0.6}.

3. **Residual policy quality**  
   - Retrain with `--curriculum` over the altitude×speed grid.  
   - Full-stack max\|Δh\| must be ≤ baseline under the same fault before claiming benefit.

4. **Statistical coverage**  
   - `eval/eval_residual_grid.py` across grid × severity; publish tables (mean/std recovery Δh, time-to-recover, control effort).

5. **Design assurance package**  
   - Safety envelope tests (rate limit, pitch interlock, throttle floor) as automated unit/scenario tests.

---

## 4. Exit criteria (suggested)

**TRL 4 complete (this release):**  
- [x] Reproducible baseline PASS on F-16  
- [x] Unified multi-case run without crash  
- [x] Documented plant interface and launch sequence  

**TRL 5 candidate:**  
- [ ] FG or HIL run with ICD compliance log  
- [ ] Detector bank metrics (delay / FAR)  
- [ ] Full-stack ≤ baseline Δh under 20% and 40% elev loss on ≥3 envelope points  
- [ ] Zenodo package with frozen configs + seeds  

---

## 5. Non-claims (integrity of the release)

- Not flight-certified.  
- Not a replacement for production FCS or L1/IFCS flight programs.  
- Full-stack residual is **integrated**, not yet **superior**, on the published checkpoint.  
- C172P material is legacy only (`legacy_c172p/`).

---

## 6. Next program steps (ordered)

1. Run `eval/eval_residual_grid.py` and archive CSV in `results/eval_grid/`.  
2. Curriculum retrain residual; re-run unified FULL_STACK.  
3. Wire MMAE bank into unified residual enable path.  
4. FlightGear live session with ICD checklist.  
5. HIL planning (surface command rate, latency budget) using the same ICD.


## Deferred (explicit)

- CNN-BiLSTM / learned PHI-Twin detector (architecture reserved; not in this release)
- Multi-aircraft statistical campaign beyond F-16 (+ legacy C172P archive only)
