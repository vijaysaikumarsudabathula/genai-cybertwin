"""
genai.py
Builds the strict structured-evidence JSON object (spec section 38) for a
given test-split event, and defines the system prompt rules the GenAI layer
must follow. The GenAI layer NEVER performs the prediction -- it only
narrates evidence that was already computed by the ML/SHAP/risk/counterfactual
modules above.

In this deployment, "the LLM" is Claude operating directly on this structured
JSON (no external API key configured) -- see README.md for how to swap in an
OpenAI-compatible call later without changing this evidence contract.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

GENAI_SYSTEM_PROMPT = """You are a cybersecurity analysis assistant.
You must only reason from the structured evidence supplied by the application.
Never invent: numerical values, attack indicators, IP addresses, CVEs, malware
names, evidence, model predictions, or intervention outcomes.
Clearly distinguish: Observed evidence / ML prediction / counterfactual
simulation / AI-generated interpretation.
Use cautious language ("indicates", "simulated", "may", "model-based").
If a field is missing, contradictory, or absent in the evidence, say so
explicitly (e.g. "No counterfactual data was available for this event") --
never fill the gap with a plausible-sounding invented value.
Output format: Threat Summary / Evidence / Model Interpretation /
Counterfactual Findings / Hypothetical Scenario / Defensive Considerations /
Limitations.
"""


def build_structured_evidence(event_idx: int):
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    raw_splits = joblib.load(MODELS_DIR / "raw_splits.joblib")
    X_test, y_test = splits["X_test"], splits["y_test"]
    X_test_raw = raw_splits["X_test_raw"]

    row = X_test.iloc[[event_idx]]
    raw_row = X_test_raw.iloc[event_idx]
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    confidence = float(proba.max())

    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(row))
    contribs = sv[0, :, pred] if sv.ndim == 3 else sv[0]
    top_features = sorted(zip(X_test.columns, row.iloc[0].values.tolist(), contribs.tolist()),
                            key=lambda x: -abs(x[2]))[:6]

    risk_path = RESULTS_DIR / "risk_scores.csv"
    risk_score, risk_category = None, None
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        if event_idx < len(risk_df):
            risk_score = float(risk_df.iloc[event_idx]["cyber_risk_score"])
            risk_category = risk_df.iloc[event_idx]["risk_category"]

    cf_path = RESULTS_DIR / "counterfactual_results.csv"
    counterfactuals = []
    if cf_path.exists():
        cf_df = pd.read_csv(cf_path)
        matches = cf_df[cf_df.event_idx == event_idx]
        for _, r in matches.iterrows():
            counterfactuals.append({
                "method": r["method"], "intervention": r["intervention"],
                "original_probability": float(r["original_probability"]),
                "new_probability": float(r["new_probability"]),
                "probability_change": float(r["probability_change"]),
                "prediction_changed": bool(r["new_class"] != r["original_class"]),
            })

    limitations = [
        "The counterfactual results are model-based simulations, not observed real-world "
        "intervention outcomes.",
        "Class semantics (what 'Attack Type 0/1/2' concretely represent) are not documented "
        "by the competition; interpretation is limited to the model's statistical pattern.",
        "This event is from the held-out test split, so its true label is known here for "
        "research validation -- in a live deployment it would not be.",
    ]

    evidence = {
        "event_id": int(X_test_raw.iloc[event_idx]["ID"]) if "ID" in X_test_raw.columns else event_idx,
        "timestamp": str(X_test_raw.iloc[event_idx]["Timestamp"]) if "Timestamp" in X_test_raw.columns else None,
        "true_label": int(y_test.values[event_idx]),
        "prediction": pred,
        "probabilities": {str(k): float(p) for k, p in enumerate(proba)},
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "top_features": [{"feature": f, "value": v} for f, v, _ in top_features],
        "shap_contributions": [{"feature": f, "value": v, "contribution": c} for f, v, c in top_features],
        "counterfactuals": counterfactuals if counterfactuals else "N/A - no counterfactual generated for this event",
        "limitations": limitations,
    }
    return evidence


if __name__ == "__main__":
    ev = build_structured_evidence(0)
    print(json.dumps(ev, indent=2, default=str))
