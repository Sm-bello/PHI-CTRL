# JSBSim Classical Baseline Recovery (V10)

**Gate-keeper folder.** Prove a classical baseline survives a 20 % elevator
effectiveness fault **before** MRAC / residual / TECS.

## Plant switch (important)

The upstream JSBSim **c172p** model is marked BETA and consistently fails to
produce a usable airborne trim (gear contact on every IC, sustained descent
after “trim”).  

**V10 therefore defaults to the F-16 model**, which is mature in JSBSim.
The PHI-CTRL architecture itself is unchanged and aircraft-agnostic.

| Layer | Role | Status |
|-------|------|--------|
| 0 Plant | JSBSim F-16 (or c172p legacy) | switched |
| 1 Baseline | Classical alt/pitch + auto-throttle | this folder |
| 2 Resilient | MRAC + observer + detector | next |
| 3 Integration | Unified orchestrator + residual | next |

## Run order

```bash
cd baseline_jsbsim_recovery

# Default = F-16
python run_baseline_recovery.py --hold-trim
python run_baseline_recovery.py --no-fault
python run_baseline_recovery.py                  # 20 % elev fault

# Optional: force legacy C172P (expected to struggle)
python run_baseline_recovery.py --aircraft c172p --hold-trim

# Custom JSBSim root
python run_baseline_recovery.py --jsbsim-root /path/to/JSBSim
```

## Pass criteria

1. Pre-fault |Δh| < 120 ft  
2. Full duration (no bailout)  
3. Min airspeed ≥ 55 % of target  
4. Pitch within ±40°  

Hold-trim must stay near the commanded altitude/speed.  
Only after **OVERALL BASELINE: PASS** proceed to residual / MRAC.
