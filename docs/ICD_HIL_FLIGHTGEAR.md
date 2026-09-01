# Interface Control Document (ICD)  
## PHI-CTRL ↔ JSBSim F-16 ↔ FlightGear / future HIL

**Version:** 1.0 (release package)  
**Plant:** General Dynamics F-16A (JSBSim production model, Hofman)  
**Purpose:** Freeze the software interface so HIL and FlightGear integration do not require re-architecting control laws.

---

## 1. Authority and ownership

| Item | Value | Notes |
|------|-------|-------|
| `fcs/fbw-override` | `1.0` | Required. Without this, native F-16 FCS fights external commands. |
| Gear / flaps | Commanded up / zero | Airborne experiment envelope |
| Integration rate | 120 Hz (`DT = 1/120`) | Matches baseline and gym env |

---

## 2. Command properties (PHI-CTRL → plant)

| Property | Range | Units / meaning | Critical notes |
|----------|-------|-----------------|----------------|
| `fcs/elevator-cmd-norm` | [−1, 1] | Normalized elevator | **Sign: positive = nose down** on this model |
| `fcs/pitch-trim-cmd-norm` | [−1, 1] | Pitch trim | Held near trim solution |
| `fcs/throttle-cmd-norm` | [0, 1] | Throttle command | Written for completeness |
| `fcs/throttle-pos-norm` | [0, 1] | Throttle position | **This is what actually drives thrust** on this F-16 model (cmd alone is insufficient) |
| `fcs/aileron-cmd-norm` | [−1, 1] | Roll | Wing-level law |
| `fcs/rudder-cmd-norm` | [−1, 1] | Yaw | Dutch-roll damping |
| `fcs/speedbrake-cmd-norm` | [0, SB_CAP] | Speedbrake | Cap ≤ 0.20 in baseline (pitch coupling) |

Rate limits (software):

- Elevator command rate ≤ `ELEV_RATE_MAX` (0.35 / s)  
- Throttle rate ≤ `THR_RATE_MAX` (0.40 / s)  
- Throttle floor ≥ `THR_FLOOR` (0.15) unless explicit energy bleed logic allows lower  

---

## 3. Sensing properties (plant → PHI-CTRL)

| Property | Use |
|----------|-----|
| `position/h-sl-ft` | Altitude hold / path |
| `velocities/vc-kts` | Speed / energy |
| `velocities/h-dot-fps` | Path damping |
| `attitude/theta-rad` | Pitch hold |
| `attitude/phi-rad` | Wing level |
| `velocities/q-rad_sec` | Pitch rate damping |
| `velocities/p-rad_sec`, `r-rad_sec` | Roll/yaw damp |
| `aero/alpha-rad` | Monitoring |
| `fcs/elevator-pos-norm` | Surface feedback / diagnostics |
| `propulsion/engine[0]/thrust-lbs` | Thrust verification |

---

## 4. Fault injection interface

| Mechanism | Implementation |
|-----------|----------------|
| Elevator effectiveness γ | Plant-side multiply on elevator command before apply (`elev_plant = elev_cmd * γ`) |
| Nominal | γ = 1.0 |
| Study points | γ ∈ {0.8, 0.6} (20% / 40% loss) |
| Enable time | Configurable (unified / gym: fault after settle) |

Detector bank hypotheses (software): nominal, γ≈0.8, γ≈0.6 — see `detector/mmae_bank.py`.

---

## 5. FlightGear socket (typical)

```text
# FlightGear listens for native FDM
fgfs --aircraft=f16 --fdm=null --native-fdm=socket,in,60,,5505,udp [other opts]

# PHI-CTRL / JSBSim sends state at agreed rate (see run_f16_live_flightgear.py)
```

Rules:

- Metrics and science claims use **JSBSim truth**, not FG rendering.  
- FG is relevant-environment visualization and operator interface toward TRL 5.  
- Do not change property names above without a new ICD revision.

---

## 6. HIL checklist (future)

- [ ] Command latency budget documented (loop ≤ 1/60 s preferred)  
- [ ] Surface rate limits match actuator model  
- [ ] Inhibit: auto-disengage residual if \|θ\| > interlock threshold  
- [ ] Logging: UTC timestamps, γ true, γ estimated, elev_cmd, elev_plant  
- [ ] Repeat baseline PASS scenario on HIL before residual enable  

---

## 7. Revision control

Any change to property names, sign conventions, or rate limits requires:

1. Update this ICD version  
2. Re-run baseline gate  
3. Note in Zenodo version history  
