# Centrifugal Pump Preventive Maintenance Guide

## Document Information

- **Document ID**: PM-PUMP-001
- **Revision**: 2.1
- **Date**: 2025-01-15
- **Classification**: INTERNAL

## Overview

This guide provides preventive maintenance procedures for centrifugal pumps in oil and gas facilities to ensure optimal performance and prevent unscheduled downtime.

## Maintenance Schedule

### Daily Inspections

- **Oil Level Check**: Verify oil level in bearing housing using sight glass
- **Temperature Monitoring**: Record bearing temperature (should not exceed 80°C)
- **Vibration Check**: Listen for unusual noise or vibration
- **Seal Inspection**: Check for leakage at mechanical seal

### Weekly Inspections

- **Coupling Alignment**: Check alignment indicator
- **Foundation Bolts**: Inspect for looseness
- **Suction Pressure**: Verify adequate NPSH (Net Positive Suction Head)
- **Discharge Pressure**: Compare against baseline

### Monthly Maintenance

- **Filter Replacement**: Replace strainer/filter elements
- **Lubrication**: Grease bearings per lubrication schedule
- **Seal Water System**: Clean and flush seal water lines
- **Performance Test**: Record flow rate and pressure

### Quarterly Maintenance

- **Vibration Analysis**: Perform detailed vibration spectrum analysis
- **Oil Analysis**: Sample and test bearing oil
- **Seal Inspection**: Detailed inspection of mechanical seal faces
- **Coupling Inspection**: Remove guard and check rubber elements

### Annual Overhaul

- **Bearing Replacement**: Replace all rolling element bearings
- **Seal Replacement**: Install new mechanical seal assembly
- **Impeller Inspection**: Check for erosion, cavitation damage
- **Shaft Runout**: Measure shaft runout (max 0.002 inches)
- **Performance Curve**: Verify pump curve matches manufacturer data

## Common Failure Modes

### Seal Leakage

**Symptoms:**

- Visible fluid leakage at seal area
- Excessive seal flush consumption
- Product contamination

**Root Causes:**

- Worn seal faces
- Improper seal flush pressure
- Excessive vibration
- Chemical attack on seals

**Corrective Action:**

1. Shut down pump immediately if major leak
2. Check seal flush pressure (should be 15-20 psi above suction)
3. Verify seal cooling water flow rate
4. Replace seal if faces are damaged
5. Check for shaft sleeve scoring

### Bearing Failure

**Symptoms:**

- High bearing temperature (>80°C)
- Excessive vibration
- Metallic grinding noise
- Oil contamination (discoloration, metal particles)

**Root Causes:**

- Inadequate lubrication
- Water ingress
- Misalignment
- Excessive axial/radial loads

**Corrective Action:**

1. Stop pump if temperature exceeds 90°C
2. Check oil level and condition
3. Sample oil for analysis
4. Verify alignment is within 0.002 inches
5. Replace bearings if damaged

### Low Flow / High Vibration

**Symptoms:**

- Reduced flow rate
- High vibration at vane pass frequency
- Cavitation noise

**Root Causes:**

- Impeller wear
- Cavitation due to low NPSH
- Recirculation
- Blockage in suction

**Corrective Action:**

1. Check suction pressure - must exceed NPSHr + 3 ft
2. Inspect suction strainer for blockage
3. Verify impeller clearances (should be 0.015-0.030 inches)
4. Check for air ingress in suction line
5. Replace impeller if clearances excessive

## Safety Precautions

⚠️ **WARNING**: Pumps handle hazardous fluids under pressure

- **Lockout/Tagout**: Always apply LOTO before maintenance
- **Depressurize**: Vent and drain pump before opening
- **PPE Required**: Safety glasses, gloves, chemical suit if required
- **Hot Surfaces**: Allow pump to cool before touching
- **Confined Space**: Follow permit procedures for tank entry

## Spare Parts Inventory

### Critical Spares (Keep in Stock)

- Mechanical seal assembly (2 units)
- Bearing set (inboard/outboard) (2 sets)
- O-ring kit
- Coupling spider (2 units)
- Impeller (1 unit)

### Recommended Spares

- Shaft sleeve
- Wear ring set
- Casing gasket set
- Bearing cover
- Seal flush pot

## Performance Baseline

| Parameter | Normal Range | Alert Threshold |
|-----------|--------------|-----------------|
| Flow Rate | 500-550 GPM | < 480 GPM |
| Discharge Pressure | 145-155 psig | < 140 psig |
| Motor Current | 42-48 A | > 52 A |
| Bearing Temp | 60-75°C | > 80°C |
| Vibration | 0.1-0.3 in/s | > 0.4 in/s |

## Troubleshooting Quick Reference

| Problem | Likely Cause | Check First |
|---------|--------------|-------------|
| No flow | Pump not primed | Suction valve open? |
| Low flow | Impeller worn | Discharge pressure |
| High vibration | Misalignment | Coupling alignment |
| Seal leaking | Seal faces worn | Seal flush pressure |
| High bearing temp | Low oil level | Oil sight glass |
| Cavitation noise | Low suction pressure | NPSH available |

## Document References

- API Standard 610: Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries
- Manufacturer Manual: Flowserve Model 3196
- Site P&ID: Drawing P-1001-A
