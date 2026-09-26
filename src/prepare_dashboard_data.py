"""
prepare_dashboard_data.py
Consolidates every results/ file produced by the pipeline into one JSON
payload the HTML dashboard embeds directly (no server, no fetch). Every
number here traces back to a file already produced by validation.py,
train.py, evaluate.py, explainability.py, risk_engine.py, counterfactual.py,
ablation.py, error_analysis.py, or generate_genai_narratives.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "data" / "reports"
DASHBOARD_DIR = ROOT / "dashboard"

ACTIONABLE_FEATURES = ["Suspicious_Port_Activity", "Traffic_Volume_Variation",
                        "Login_Attempts", "Malware_Score", "User_Behavior_Score",
                        "Packet_Retry_Rate", "Geo_Dispersion"]
INTERVENTION_LABELS = {
    "Suspicious_Port_Activity": "Restrict suspicious port activity",
    "Traffic_Volume_Variation": "Apply traffic shaping / rate limiting",
    "Login_Attempts": "Enforce MFA / account lockout policy",
    "Malware_Score": "Run malware remediation (AV/EDR)",
    "User_Behavior_Score": "Tighten access control / user re-training",
    "Packet_Retry_Rate": "Apply network QoS / stability controls",
    "Geo_Dispersion": "Apply geo-fencing / VPN access restriction",
}
EDA_FEATURES = ["Device_Response_Time", "Threat_Level_Index", "Malware_Score",
                "Login_Attempts", "User_Behavior_Score", "Packet_Retry_Rate"]


def r(x, nd=4):
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


def downsample(arr, n=40):
    arr = np.asarray(arr)
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


def build():
    # ---- dataset report ----
    dataset_report = json.load(open(REPORTS_DIR / "dataset_report.json"))

    # ---- model comparison ----
    model_comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv").round(4).to_dict("records")

    # ---- evaluation ----
    ev = json.load(open(RESULTS_DIR / "evaluation_full.json"))
    roc = {k: {"fpr": downsample(v["fpr"]), "tpr": downsample(v["tpr"]), "auc": r(v["auc"])}
           for k, v in ev["roc"].items()}
    pr = {k: {"precision": downsample(v["precision"]), "recall": downsample(v["recall"]),
              "average_precision": r(v["average_precision"])} for k, v in ev["pr"].items()}
    calibration = ev["calibration"]
    confidence_summary = ev["confidence_summary"]
    confusion_matrix = ev["confusion_matrix"]
    confusion_matrix_norm = ev["confusion_matrix_normalized"]
    classification_report = ev["classification_report"]

    # ---- confidence histogram ----
    conf_df = pd.read_csv(RESULTS_DIR / "confidence_analysis.csv")
    hist, edges = np.histogram(conf_df["confidence"], bins=20, range=(0.33, 1.0))
    confidence_hist = {"counts": hist.tolist(), "edges": edges.round(3).tolist()}

    # ---- feature importance comparison ----
    shap_importance = pd.read_csv(RESULTS_DIR / "shap_importance.csv").round(5).to_dict("records")

    # ---- risk engine ----
    risk_summary = json.load(open(RESULTS_DIR / "risk_engine_summary.json"))
    risk_df = pd.read_csv(RESULTS_DIR / "risk_scores.csv")

    # ---- ablation ----
    ablation = pd.read_csv(RESULTS_DIR / "ablation_results.csv").round(4).to_dict("records")
    ablation_note = (RESULTS_DIR / "ablation_note.txt").read_text()

    # ---- error analysis ----
    error_summary = json.load(open(RESULTS_DIR / "error_analysis_summary.json"))

    # ---- counterfactual ----
    cf_metrics = json.load(open(RESULTS_DIR / "counterfactual_metrics.json"))
    cf_results = pd.read_csv(RESULTS_DIR / "counterfactual_results.csv")
    intervention_ranking = pd.read_csv(RESULTS_DIR / "intervention_ranking.csv").round(4).to_dict("records")

    cf_by_idx = {}
    for _, row in cf_results.iterrows():
        idx = int(row["event_idx"])
        cf_by_idx.setdefault(str(idx), []).append({
            "method": row["method"], "intervention": row["intervention"],
            "original_probability": r(row["original_probability"]),
            "new_probability": r(row["new_probability"]),
            "probability_change": r(row["probability_change"]),
            "new_class": int(row["new_class"]), "original_class": int(row["original_class"]),
        })

    # ---- genai narratives ----
    genai_narratives = json.load(open(RESULTS_DIR / "genai_narratives.json"))
    grounding_tests = pd.read_csv(RESULTS_DIR / "genai_grounding_tests.csv").to_dict("records")

    # ---- per-event data for Attack Explorer / Investigation (ALL test-split events) ----
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    raw_splits = joblib.load(MODELS_DIR / "raw_splits.joblib")
    X_test, y_test = splits["X_test"], splits["y_test"]
    X_test_raw = raw_splits["X_test_raw"]
    X_train_full = splits["X_trainval"]
    feature_names = list(X_test.columns)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    shap_values = np.load(MODELS_DIR / "shap_values_sample.npy")  # (n, n_features, n_classes)

    featured_ids = set(int(k) for k in genai_narratives.keys())

    # ---- actionable feature bounds (for What-If slider ranges) ----
    actionable_bounds = {f: [r(X_train_full[f].quantile(0.01), 3), r(X_train_full[f].quantile(0.99), 3)]
                          for f in ACTIONABLE_FEATURES}

    # ---- risk engine normalization bounds (must match risk_engine.py's train_ref = splits['X_train']) ----
    X_train_for_risk = splits["X_train"]
    risk_anomaly_features = list(risk_summary["anomaly_feature_weights"].keys())
    risk_normalization_bounds = {f: [r(X_train_for_risk[f].min(), 4), r(X_train_for_risk[f].max(), 4)]
                                  for f in risk_anomaly_features}

    events = []
    for i in range(len(X_test)):
        pred = int(y_pred[i])
        proba = y_proba[i]
        contribs = shap_values[i, :, pred] if shap_values.ndim == 3 else None
        top_shap = []
        if contribs is not None:
            order = np.argsort(-np.abs(contribs))[:5]
            for j in order:
                top_shap.append({"feature": feature_names[j],
                                  "value": r(X_test.iloc[i, j], 3),
                                  "contribution": r(contribs[j], 3)})
        risk_row = risk_df.iloc[i]
        events.append({
            "idx": i,
            "event_id": int(X_test_raw.iloc[i]["ID"]),
            "timestamp": str(X_test_raw.iloc[i]["Timestamp"]),
            "true_label": int(y_test.values[i]),
            "predicted_label": pred,
            "confidence": r(proba.max(), 4),
            "probabilities": [r(p, 4) for p in proba],
            "risk_score": r(risk_row["cyber_risk_score"], 2),
            "risk_category": risk_row["risk_category"],
            "top_shap": top_shap,
            "has_counterfactual": str(i) in cf_by_idx,
            "is_featured": i in featured_ids,
            "correct": bool(pred == int(y_test.values[i])),
            "feature_vector": [r(v, 6) for v in X_test.iloc[i].values.tolist()],
        })

    # ---- EDA: correlation matrix (raw numeric features, training data) ----
    corr_features = ["Suspicious_Port_Activity", "Traffic_Volume_Variation", "Packet_Length_Anomaly",
                      "Payload_Entropy", "Packet_Retry_Rate", "Malware_Score", "Threat_Level_Index",
                      "Anomaly_Tendency", "User_Behavior_Score", "Login_Attempts", "Geo_Dispersion",
                      "Device_Response_Time", "Session_Duration"]
    corr_matrix = X_train_full[corr_features].corr().round(3)
    eda_correlation = {"features": corr_features, "matrix": corr_matrix.values.tolist()}

    # ---- EDA: per-class quartiles for key features (for box plots) ----
    y_train_full = splits["y_trainval"]
    train_with_y = X_train_full.copy()
    train_with_y["_y"] = y_train_full.values
    eda_boxplots = {}
    for feat in EDA_FEATURES:
        per_class = {}
        for cls in [0, 1, 2]:
            vals = train_with_y.loc[train_with_y["_y"] == cls, feat]
            per_class[str(cls)] = {
                "min": r(vals.min(), 2), "q1": r(vals.quantile(0.25), 2),
                "median": r(vals.median(), 2), "q3": r(vals.quantile(0.75), 2),
                "max": r(vals.max(), 2),
            }
        eda_boxplots[feat] = per_class

    # ---- EDA: hour-of-day x attack-class heatmap ----
    hour_class_counts = train_with_y.groupby(["hour", "_y"]).size().unstack(fill_value=0)
    hour_class_counts = hour_class_counts.reindex(columns=[0, 1, 2], fill_value=0).reindex(range(24), fill_value=0)
    eda_hour_heatmap = {"hours": list(range(24)), "matrix": hour_class_counts.values.T.tolist()}

    payload = {
        "meta": {
            "final_model": "XGBoost",
            "generated_note": "All figures computed from the actual KAUST train.csv/test.csv by this project's pipeline.",
        },
        "dataset_report": dataset_report,
        "model_comparison": model_comparison,
        "roc": roc, "pr": pr, "calibration": calibration,
        "confidence_summary": confidence_summary, "confidence_hist": confidence_hist,
        "confusion_matrix": confusion_matrix, "confusion_matrix_norm": confusion_matrix_norm,
        "classification_report": classification_report,
        "shap_importance": shap_importance,
        "risk_summary": risk_summary,
        "ablation": ablation, "ablation_note": ablation_note,
        "error_summary": error_summary,
        "counterfactual_metrics": cf_metrics,
        "counterfactual_by_idx": cf_by_idx,
        "intervention_ranking": intervention_ranking,
        "genai_narratives": genai_narratives,
        "grounding_tests": grounding_tests,
        "events": events,
        "actionable_features": ACTIONABLE_FEATURES,
        "intervention_labels": INTERVENTION_LABELS,
        "actionable_bounds": actionable_bounds,
        "risk_normalization_bounds": risk_normalization_bounds,
        "feature_names": feature_names,
        "eda_correlation": eda_correlation,
        "eda_boxplots": eda_boxplots,
        "eda_hour_heatmap": eda_hour_heatmap,
    }

    out_path = DASHBOARD_DIR / "dashboard_data.json"

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    with open(out_path, "w") as f:
        json.dump(payload, f, default=_default)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB), {len(events)} events")
    return payload


if __name__ == "__main__":
    build()
