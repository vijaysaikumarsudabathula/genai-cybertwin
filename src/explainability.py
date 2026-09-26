"""
explainability.py
SHAP-based explainability for the final XGBoost model: global importance,
per-class importance, and individual-event explanations. Also computes
model-native and permutation importance for a three-way comparison table,
per spec section 25.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def run_explainability(sample_size=1603):
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    X_test, y_test = splits["X_test"], splits["y_test"]
    feature_names = list(X_test.columns)

    X_sample = X_test.iloc[:sample_size]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)  # shape: (n_samples, n_features, n_classes) for newer shap
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[0] != X_sample.shape[0]:
        # older shap returns list-per-class -> stack; ensure (n_samples, n_features, n_classes)
        shap_values = np.transpose(shap_values, (1, 2, 0))
    elif shap_values.ndim == 3 and shap_values.shape[-1] != 3:
        pass  # already (n_samples, n_features, n_classes)

    n_classes = shap_values.shape[-1] if shap_values.ndim == 3 else 1

    # ---- global importance (mean |SHAP| across all classes) ----
    if shap_values.ndim == 3:
        global_importance = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        global_importance = np.mean(np.abs(shap_values), axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": feature_names, "shap_importance": global_importance
    }).sort_values("shap_importance", ascending=False)

    # ---- per-class importance ----
    class_importance = {}
    if shap_values.ndim == 3:
        for k in range(n_classes):
            imp = np.mean(np.abs(shap_values[:, :, k]), axis=0)
            class_importance[k] = dict(sorted(zip(feature_names, imp.tolist()),
                                                key=lambda x: -x[1]))

    # ---- model-native importance ----
    model_importance = dict(zip(feature_names, model.feature_importances_.tolist()))

    # ---- permutation importance (on test split, macro F1 scoring) ----
    perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42,
                                   scoring="f1_macro", n_jobs=-1)
    perm_importance = dict(zip(feature_names, perm.importances_mean.tolist()))

    comparison = pd.DataFrame({
        "feature": feature_names,
        "model_importance": [model_importance[f] for f in feature_names],
        "shap_importance": [dict(zip(feature_names, global_importance))[f] for f in feature_names],
        "permutation_importance": [perm_importance[f] for f in feature_names],
    }).sort_values("shap_importance", ascending=False)
    comparison.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)

    # ---- individual event explanations for a handful of representative events ----
    # pick: highest confidence correct prediction per class, plus the largest
    # misclassifications (for error analysis / GenAI narrative later)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    representative_idx = []
    for cls in [0, 1, 2]:
        mask = (y_test.values == cls) & (y_pred == cls)
        if mask.any():
            idx = np.where(mask)[0]
            best = idx[np.argmax(y_proba[idx, cls])]
            representative_idx.append(int(best))
    mis_mask = y_pred != y_test.values
    if mis_mask.any():
        mis_idx = np.where(mis_mask)[0]
        worst = mis_idx[np.argmax(y_proba[mis_idx].max(axis=1))]
        representative_idx.append(int(worst))

    individual_explanations = []
    for idx in representative_idx:
        row = X_sample.iloc[idx] if idx < len(X_sample) else X_test.iloc[idx]
        pred_class = int(y_pred[idx])
        if shap_values.ndim == 3 and idx < shap_values.shape[0]:
            contribs = shap_values[idx, :, pred_class]
        else:
            contribs = np.zeros(len(feature_names))
        top = sorted(zip(feature_names, row.values.tolist(), contribs.tolist()),
                      key=lambda x: -abs(x[2]))[:10]
        individual_explanations.append({
            "row_index": idx,
            "true_label": int(y_test.values[idx]),
            "predicted_label": pred_class,
            "confidence": float(y_proba[idx].max()),
            "class_probabilities": y_proba[idx].tolist(),
            "top_features": [{"feature": f, "value": v, "shap_contribution": c,
                               "direction": "increases risk" if c > 0 else "decreases risk"}
                              for f, v, c in top],
        })

    with open(RESULTS_DIR / "individual_explanations.json", "w") as f:
        json.dump(individual_explanations, f, indent=2)

    global_shap_summary = {
        "top_10_global": shap_importance_df.head(10).to_dict("records"),
        "per_class_top_5": {k: dict(list(v.items())[:5]) for k, v in class_importance.items()},
    }
    with open(RESULTS_DIR / "shap_global_summary.json", "w") as f:
        json.dump(global_shap_summary, f, indent=2, default=str)

    # save raw shap values + sample for dashboard chart building
    np.save(MODELS_DIR / "shap_values_sample.npy", shap_values)
    X_sample.to_csv(MODELS_DIR / "shap_sample_features.csv", index=False)

    print("Global SHAP importance (top 5):")
    print(shap_importance_df.head())
    return shap_importance_df, individual_explanations


if __name__ == "__main__":
    run_explainability()
