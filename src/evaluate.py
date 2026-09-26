"""
evaluate.py
Deep evaluation of the FINAL selected model on the held-out test split:
confusion matrix, per-class metrics, ROC/PR (one-vs-rest), calibration,
confidence/entropy analysis. All numbers computed from actual predictions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                              precision_recall_curve, average_precision_score, brier_score_loss)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CLASS_NAMES = {0: "Attack Type 0", 1: "Attack Type 1", 2: "Attack Type 2"}


def load_final():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    return model, splits


def run_evaluation():
    model, splits = load_final()
    X_test, y_test = splits["X_test"], splits["y_test"]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    n_classes = 3

    # ---- confusion matrices ----
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm, index=[f"actual_{i}" for i in range(3)],
                 columns=[f"pred_{i}" for i in range(3)]).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    # ---- classification report ----
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULTS_DIR / "classification_report.csv")

    # ---- one-vs-rest ROC / PR ----
    roc_data, pr_data = {}, {}
    y_test_arr = np.asarray(y_test)
    for k in range(n_classes):
        y_bin = (y_test_arr == k).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, k])
        roc_data[k] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, k])
        pr_data[k] = {"precision": prec.tolist(), "recall": rec.tolist(),
                       "average_precision": float(average_precision_score(y_bin, y_proba[:, k]))}

    # ---- calibration (per-class one-vs-rest reliability, 10 bins) ----
    calibration = {}
    for k in range(n_classes):
        y_bin = (y_test_arr == k).astype(int)
        probs = y_proba[:, k]
        bins = np.linspace(0, 1, 11)
        bin_ids = np.digitize(probs, bins) - 1
        bin_ids = np.clip(bin_ids, 0, 9)
        mean_pred, mean_actual, counts = [], [], []
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() > 0:
                mean_pred.append(float(probs[mask].mean()))
                mean_actual.append(float(y_bin[mask].mean()))
                counts.append(int(mask.sum()))
        brier = brier_score_loss(y_bin, probs)
        # expected calibration error
        ece = float(np.sum([c * abs(p - a) for p, a, c in zip(mean_pred, mean_actual, counts)]) / len(probs))
        calibration[k] = {"mean_predicted": mean_pred, "mean_actual": mean_actual,
                           "bin_counts": counts, "brier_score": float(brier), "ece": ece}

    # ---- confidence / entropy analysis ----
    max_proba = y_proba.max(axis=1)
    pred_entropy = entropy(y_proba.T, base=2)
    correct = (y_pred == y_test_arr)
    confidence_df = pd.DataFrame({
        "true_label": y_test_arr, "predicted_label": y_pred,
        "confidence": max_proba, "entropy": pred_entropy, "correct": correct,
    })
    confidence_df.to_csv(RESULTS_DIR / "confidence_analysis.csv", index=False)

    confidence_summary = {
        "mean_confidence_overall": float(max_proba.mean()),
        "mean_confidence_correct": float(max_proba[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(max_proba[~correct].mean()) if (~correct).any() else None,
        "pct_high_confidence_gt_0.9": float((max_proba > 0.9).mean() * 100),
        "pct_low_confidence_lt_0.6": float((max_proba < 0.6).mean() * 100),
    }

    out = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "classification_report": report,
        "roc": roc_data,
        "pr": pr_data,
        "calibration": calibration,
        "confidence_summary": confidence_summary,
    }
    with open(RESULTS_DIR / "evaluation_full.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Confusion matrix:\n", cm)
    print("\nMacro F1 (test):", report["macro avg"]["f1-score"])
    print("Confidence summary:", json.dumps(confidence_summary, indent=2))
    return out


if __name__ == "__main__":
    run_evaluation()
