"""
risk_engine.py
Cyber Risk Score: a transparent, DERIVED analytical indicator. It is NOT
ground truth and is kept visually/structurally separate from the raw model
probability everywhere it is displayed.

IMPORTANT INTERPRETATION NOTE (documented honestly, not assumed):
No data dictionary was provided with this Kaggle competition, so we do not
know whether class 0 represents "benign traffic" or simply "Attack Type 0".
The risk engine therefore does NOT assume class 0 = safe. Instead it uses
`deviation_probability = 1 - P(class 0)`, i.e. "how much probability mass the
model places away from the majority/baseline pattern" -- a description that
holds regardless of what class 0 actually represents.

Formula:
    risk_raw = alpha * deviation_probability + (1 - alpha) * anomaly_component
    Cyber Risk Score = 100 * risk_raw

    anomaly_component = weighted sum of min-max normalized
        [Threat_Level_Index, Malware_Score, Anomaly_Tendency,
         Device_Response_Time, Login_Attempts]
    weights = each feature's SHAP importance (from explainability.py),
              renormalized to sum to 1 across just these five features.
    alpha = 0.6 (documented, configurable below)

Categories (configurable):
    LOW      : score <  25
    MEDIUM   : 25 <= score < 50
    HIGH     : 50 <= score < 75
    CRITICAL : score >= 75
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

ALPHA = 0.6
ANOMALY_FEATURES = ["Threat_Level_Index", "Malware_Score", "Anomaly_Tendency",
                     "Device_Response_Time", "Login_Attempts"]
THRESHOLDS = {"LOW": 0, "MEDIUM": 25, "HIGH": 50, "CRITICAL": 75}


def get_category(score, thresholds=THRESHOLDS):
    if score >= thresholds["CRITICAL"]:
        return "CRITICAL"
    if score >= thresholds["HIGH"]:
        return "HIGH"
    if score >= thresholds["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def load_anomaly_weights():
    shap_imp = pd.read_csv(RESULTS_DIR / "shap_importance.csv").set_index("feature")["shap_importance"]
    sub = shap_imp.loc[ANOMALY_FEATURES]
    weights = (sub / sub.sum()).to_dict()
    return weights


def compute_risk(X_raw: pd.DataFrame, y_proba: np.ndarray, train_ref: pd.DataFrame, weights=None):
    """X_raw: dataframe with the ANOMALY_FEATURES columns (post feature-engineering,
    pre-scaling). train_ref: the training split, used ONLY to fetch min/max for
    normalization (fit-on-train, no leakage). y_proba: (n,3) model probabilities."""
    if weights is None:
        weights = load_anomaly_weights()

    deviation_probability = 1 - y_proba[:, 0]

    anomaly_component = np.zeros(len(X_raw))
    for feat, w in weights.items():
        lo, hi = train_ref[feat].min(), train_ref[feat].max()
        norm = (X_raw[feat].values - lo) / (hi - lo + 1e-9)
        norm = np.clip(norm, 0, 1)
        anomaly_component += w * norm

    risk_raw = ALPHA * deviation_probability + (1 - ALPHA) * anomaly_component
    risk_score = np.clip(100 * risk_raw, 0, 100)
    categories = np.array([get_category(s) for s in risk_score])
    return risk_score, categories, deviation_probability


def run_risk_engine():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    X_test, y_test = splits["X_test"], splits["y_test"]
    X_train = splits["X_train"]

    y_proba = model.predict_proba(X_test)
    weights = load_anomaly_weights()
    risk_score, categories, dev_prob = compute_risk(X_test, y_proba, X_train, weights)

    df = pd.DataFrame({
        "true_label": y_test.values,
        "predicted_label": model.predict(X_test),
        "model_probability_predicted_class": y_proba.max(axis=1),
        "deviation_probability": dev_prob,
        "cyber_risk_score": risk_score,
        "risk_category": categories,
    })
    df.to_csv(RESULTS_DIR / "risk_scores.csv", index=False)

    summary = {
        "alpha": ALPHA,
        "anomaly_feature_weights": weights,
        "thresholds": THRESHOLDS,
        "category_counts": pd.Series(categories).value_counts().to_dict(),
        "mean_risk_by_predicted_class": df.groupby("predicted_label")["cyber_risk_score"].mean().to_dict(),
    }
    with open(RESULTS_DIR / "risk_engine_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # write documentation
    doc = f"""# Cyber Risk Engine

The Cyber Risk Score is a **derived analytical indicator**, not ground truth
and not the same thing as the model's raw class probability. Both are always
shown separately in the dashboard.

## Formula
```
risk_raw = {ALPHA} * deviation_probability + {1-ALPHA:.1f} * anomaly_component
Cyber Risk Score = 100 * risk_raw   (0-100 scale)
```
- **deviation_probability** = 1 - P(predicted class is the majority pattern, class 0).
  We do not assume class 0 means "benign" (no data dictionary was provided) --
  this is simply "how much probability the model places away from the
  majority/baseline pattern in this 3-class problem."
- **anomaly_component** = weighted, min-max normalized combination of
  {', '.join(ANOMALY_FEATURES)}, where weights are each feature's own
  SHAP importance (data-driven, not arbitrary), renormalized to sum to 1.
  Min/max bounds are fit on the training split only.

## Weights used (from SHAP importance, actual values)
{json.dumps(weights, indent=2)}

## Categories (configurable)
| Category | Score range |
|---|---|
| LOW | 0 - {THRESHOLDS['MEDIUM']} |
| MEDIUM | {THRESHOLDS['MEDIUM']} - {THRESHOLDS['HIGH']} |
| HIGH | {THRESHOLDS['HIGH']} - {THRESHOLDS['CRITICAL']} |
| CRITICAL | {THRESHOLDS['CRITICAL']}+ |

## Actual category distribution on the held-out test split
{json.dumps(summary['category_counts'], indent=2)}
"""
    (RESULTS_DIR / "risk_engine.md").write_text(doc)
    print(json.dumps(summary, indent=2, default=str))
    return df, summary


if __name__ == "__main__":
    run_risk_engine()
