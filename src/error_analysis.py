"""
error_analysis.py
Analyzes actual misclassifications of the final model on the held-out test
split: most common confusion pairs, high/low confidence errors, and SHAP
evidence for representative errors.
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


def run_error_analysis():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    X_test, y_test = splits["X_test"], splits["y_test"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_test_arr = y_test.values

    errors_mask = y_pred != y_test_arr
    errors_df = pd.DataFrame({
        "row_index": np.where(errors_mask)[0],
        "true_label": y_test_arr[errors_mask],
        "predicted_label": y_pred[errors_mask],
        "confidence": y_proba[errors_mask].max(axis=1),
    })
    errors_df["confusion_pair"] = errors_df["true_label"].astype(str) + "_to_" + errors_df["predicted_label"].astype(str)

    confusion_counts = errors_df["confusion_pair"].value_counts().to_dict()
    most_common_pair = errors_df["confusion_pair"].value_counts().idxmax() if len(errors_df) else None

    high_conf_errors = errors_df[errors_df.confidence > 0.8].sort_values("confidence", ascending=False)
    low_conf_errors = errors_df[errors_df.confidence < 0.6].sort_values("confidence")

    errors_df.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)

    # SHAP for the single highest-confidence error (most "surprising" mistake)
    shap_for_top_error = None
    if len(high_conf_errors) > 0:
        idx = int(high_conf_errors.iloc[0]["row_index"])
        explainer = shap.TreeExplainer(model)
        row = X_test.iloc[[idx]]
        sv = np.asarray(explainer.shap_values(row))
        pred_class = int(y_pred[idx])
        contribs = sv[0, :, pred_class] if sv.ndim == 3 else sv[0]
        top = sorted(zip(X_test.columns, row.iloc[0].values.tolist(), contribs.tolist()),
                      key=lambda x: -abs(x[2]))[:6]
        shap_for_top_error = {
            "row_index": idx, "true_label": int(y_test_arr[idx]), "predicted_label": pred_class,
            "confidence": float(y_proba[idx].max()),
            "top_contributing_features": [{"feature": f, "value": v, "shap_contribution": c}
                                            for f, v, c in top],
        }

    summary = {
        "total_test_events": int(len(y_test)),
        "total_errors": int(errors_mask.sum()),
        "error_rate": float(errors_mask.mean()),
        "confusion_pair_counts": confusion_counts,
        "most_common_confusion": most_common_pair,
        "n_high_confidence_errors_gt_0.8": int(len(high_conf_errors)),
        "n_low_confidence_errors_lt_0.6": int(len(low_conf_errors)),
        "representative_high_confidence_error_shap": shap_for_top_error,
    }
    with open(RESULTS_DIR / "error_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "representative_high_confidence_error_shap"},
                      indent=2))
    return summary


if __name__ == "__main__":
    run_error_analysis()
