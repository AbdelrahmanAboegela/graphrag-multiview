# Emergency Response Procedure - Valve Failure

## Document Information

- **Document ID**: ERP-VALVE-001
- **Revision**: 1.5
- **Effective Date**: 2025-01-10
- **Classification**: CRITICAL

## Scope

This procedure covers emergency response for valve failures in oil and gas production facilities, including isolation valves, control valves, and pressure relief valves.

## Types of Valve Failures

### 1. Leaking Valve (External)

**Description**: Fluid leakage from valve body, bonnet, or packing
**Severity**: HIGH (if flammable/toxic fluid)

**Immediate Actions**:

1. Notify control room immediately
2. Evacuate personnel if toxic/flammable release
3. Establish 50-meter exclusion zone
4. Attempt to tighten packing gland (if safe)
5. If leak persists, initiate isolation procedure

**Response Steps**:

- Identify upstream and downstream isolation points
- Verify isolation valves are operational
- Depressurize section per procedure ISO-001
- Apply emergency leak sealant if appropriate
- Monitor gas detection continuously

### 2. Valve Stuck Open/Closed

**Description**: Valve fails to operate, cannot close or open
**Severity**: CRITICAL (for ESD/safety valves)

**Immediate Actions**:

1. Attempt manual operation (handwheel or override)
2. Check air supply (pneumatic valves) - minimum 80 psig required
3. Verify electrical power (motor-operated valves)
4. Check for frozen/plugged condition
5. If controlling critical service, implement backup isolation

**Response Steps**:

- For stuck-open: Use upstream isolation to stop flow
- For stuck-closed: Use bypass line if available
- Check for actuator failure (air leak, solenoid fault)
- Apply penetrating oil to stem (let soak 30 min)
- Use manual override gear if present

### 3. Pressure Relief Valve (PRV) Failure

#### 3A. PRV Leaking (Passing)

**Symptoms**:

- Hissing noise from PRV discharge
- Visible vapor/fluid at vent
- Downstream piping hot/cold

**Response**:

1. Verify system pressure is below set point
2. Check if PRV is blocked open by debris
3. Activate backup PRV if installed
4. Reduce system pressure by 10% below set point
5. Schedule immediate replacement

#### 3B. PRV Fails to Open (Overpressure Event)

**CRITICAL EMERGENCY**

1. **EVACUATE** immediate area
2. Sound alarm - overpressure condition
3. Reduce source pressure immediately:
   - Close inlet isolation
   - Increase downstream flow
   - Use emergency vent if available
4. Monitor pressure continuously
5. Do NOT exceed MAWP (Maximum Allowable Working Pressure)

## Isolation Procedures

### Before Isolation

□ Obtain authorization from Production Supervisor
□ Notify control room of isolation
□ Check P&ID for affected equipment
□ Verify no personnel in hazard zone
□ Ensure PPE is adequate
□ Have spill kit ready if needed

### Isolation Steps

1. Close upstream isolation valve fully
2. Close downstream isolation valve
3. Open vent/drain valves
4. Verify zero pressure using gauge
5. Attach LOTO devices
6. Purge line if necessary (use nitrogen)
7. Post warning signs

### Depressurization Rate

- **Standard**: 50 psi/min maximum
- **Large bore (>6")**: 25 psi/min maximum
- **Cold service (<-20°C)**: 10 psi/min maximum
- **Reason**: Prevent brittle fracture and thermal shock

## Valve Type-Specific Guidance

### Gate Valves

- **Common Failure**: Stem separation, disc wedging
- **Do Not**: Operate under differential pressure
- **Check**: Stem packing leakage, bonnet bolts

### Ball Valves

- **Common Failure**: Seat leakage, actuator failure
- **Do Not**: Throttle (use full open/closed only)
- **Check**: Seat condition, ball rotation limits

### Control Valves

- **Common Failure**: Diaphragm rupture, positioner drift
- **Do Not**: Force manual override beyond stops
- **Check**: Air supply quality, instrument signals

### Check Valves

- **Common Failure**: Disc stuck, slam damage
- **Do Not**: Rely on for primary isolation
- **Check**: Flow direction, backpressure

## Root Cause Analysis

### Investigation Required

- Valve service history
- Operating conditions vs. design limits
- Maintenance records
- Previous failures of same valve
- Process upsets prior to failure

### Common Root Causes

1. **Erosion**: High velocity flow, solids present
2. **Corrosion**: Wrong material selection, chemical attack
3. **Fouling**: Polymer buildup, wax deposition
4. **Freezing**: Water condensation, hydrate formation
5. **Overpressure**: Thermal expansion, blocked discharge
6. **Vibration**: Pulsating flow, water hammer

## Repair vs. Replace Decision Matrix

| Condition | Action | Timeframe |
|-----------|--------|-----------|
| Packing leak | Tighten/replace packing | < 4 hours |
| Minor body leak | Weld repair (if code allows) | < 24 hours |
| Actuator failure | Replace actuator | < 8 hours |
| Seat damage (Class IV) | Replace trim | < 12 hours |
| Body crack | Replace valve | Immediate |
| Safety valve failed | Replace with spare | Immediate |

## Emergency Contacts

- **Control Room**: Ext. 2000 (24/7)
- **Maintenance Supervisor**: Ext. 2100
- **Production Manager**: Ext. 2200
- **Safety Officer**: Ext. 2300
- **Emergency Response Team**: Ext. 9111

## Temporary Repairs (If Approved)

⚠️ **Must be authorized by Site Manager**

### Leak Clamps

- Suitable for: Small pinhole leaks, crack <25mm
- Pressure limit: 50% of MAWP
- Duration: Max 72 hours
- Inspect: Every 4 hours

### Valve Bypassing

- Install temporary bypass line if:
  - Valve is critical to operations
  - Replacement time > 8 hours
  - Bypass can be safely pressurized
- Requires: Engineering approval, pressure test
- Max duration: 30 days

## Post-Incident Actions

1. Complete Incident Report (Form IR-001)
2. Tag failed valve for investigation
3. Update P&ID if valve replaced/modified
4. Review similar valves for preventive replacement
5. Update preventive maintenance schedule if needed
6. Conduct toolbox talk with operations team

## Prevention Measures

- **Predictive Maintenance**:
  - Quarterly valve stroke testing
  - Seat leakage testing (critical valves)
  - Acoustic emission monitoring (high-energy)
  
- **Operator Training**:
  - Proper valve operation techniques
  - Early signs of valve problems
  - Emergency response drills

- **Design Improvements**:
  - Install bypass for critical valves
  - Upgrade to fail-safe actuators
  - Duplex PRV configurations

## Regulatory Reporting

Report to authorities if:

- Release >100 kg of hydrocarbon
- Personnel injury
- Environmental impact
- Near-miss of MAWP exceedance

**Report within**: 24 hours to regulator + internal incident management

---

**Document Owner**: Maintenance Engineering
**Review Frequency**: Annual
**Next Review**: 2026-01-10
