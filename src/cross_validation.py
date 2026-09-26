"""
cross_validation.py
5-fold stratified cross-validation of the final model architecture (XGBoost)
on the full labeled dataset (train+val+test recombined, since CV needs its
own splits). This is run for the RESEARCH PAPER's evaluation protocol
section, to report mean +/- std across folds rather than a single split.
Preprocessing (winsorization) is refit inside each fold to avoid leakage.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_raw, TARGET_COL
from preprocessing import Winsorizer
from feature_engineering import TemporalFeatures
from train import FEATURE_COLUMNS, apply_preprocessor, RANDOM_STATE

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def run_cv(n_splits=5):
    train_raw, _ = load_raw()
    X = train_raw.drop(columns=[TARGET_COL])
    y = train_raw[TARGET_COL]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr_raw, X_te_raw = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_tr, y_te = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

        temporal = TemporalFeatures()
        winsor = Winsorizer()
        X_tr = apply_preprocessor(X_tr_raw, temporal, winsor, fit=True)
        X_te = apply_preprocessor(X_te_raw, temporal, winsor, fit=False)

        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   subsample=0.9, colsample_bytree=0.9,
                                   objective="multi:softprob", num_class=3,
                                   eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)

        m = {
            "fold": fold_idx,
            "accuracy": accuracy_score(y_te, y_pred),
            "macro_precision": precision_score(y_te, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_te, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
            "roc_auc_ovr_macro": roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro"),
        }
        fold_metrics.append(m)
        print(f"Fold {fold_idx}: macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

    df = pd.DataFrame(fold_metrics)
    summary = {
        "n_splits": n_splits,
        "per_fold": fold_metrics,
        "mean": {c: float(df[c].mean()) for c in df.columns if c != "fold"},
        "std": {c: float(df[c].std()) for c in df.columns if c != "fold"},
    }
    df.to_csv(RESULTS_DIR / "cross_validation_folds.csv", index=False)
    with open(RESULTS_DIR / "cross_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nMean +/- std across folds:")
    for k in summary["mean"]:
        print(f"  {k}: {summary['mean'][k]:.4f} +/- {summary['std'][k]:.4f}")
    return summary


if __name__ == "__main__":
    run_cv()
