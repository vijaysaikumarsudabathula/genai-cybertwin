# Cyber Risk Engine

The Cyber Risk Score is a **derived analytical indicator**, not ground truth
and not the same thing as the model's raw class probability. Both are always
shown separately in the dashboard.

## Formula
```
risk_raw = 0.6 * deviation_probability + 0.4 * anomaly_component
Cyber Risk Score = 100 * risk_raw   (0-100 scale)
```
- **deviation_probability** = 1 - P(predicted class is the majority pattern, class 0).
  We do not assume class 0 means "benign" (no data dictionary was provided) --
  this is simply "how much probability the model places away from the
  majority/baseline pattern in this 3-class problem."
- **anomaly_component** = weighted, min-max normalized combination of
  Threat_Level_Index, Malware_Score, Anomaly_Tendency, Device_Response_Time, Login_Attempts, where weights are each feature's own
  SHAP importance (data-driven, not arbitrary), renormalized to sum to 1.
  Min/max bounds are fit on the training split only.

## Weights used (from SHAP importance, actual values)
{
  "Threat_Level_Index": 0.14496793110208447,
  "Malware_Score": 0.08930258905620439,
  "Anomaly_Tendency": 0.012744459043119223,
  "Device_Response_Time": 0.6682167779731255,
  "Login_Attempts": 0.0847682428254663
}

## Categories (configurable)
| Category | Score range |
|---|---|
| LOW | 0 - 25 |
| MEDIUM | 25 - 50 |
| HIGH | 50 - 75 |
| CRITICAL | 75+ |

## Actual category distribution on the held-out test split
{
  "LOW": 806,
  "HIGH": 438,
  "CRITICAL": 359
}
