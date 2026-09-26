"""
train.py
Trains baseline + candidate models on the KAUST dataset with a strict,
documented, leakage-free methodology:

  1. train.csv (the only labeled file) is split 70/15/15 into
     train / validation / test, stratified on Attack Type, random_state=42.
  2. TemporalFeatures + Winsorizer are fit on the TRAIN split only, then
     applied to validation/test (and later, the unlabeled Kaggle test.csv).
  3. Each candidate model is trained on train, scored on validation to pick
     the winner by Macro F1, then the winning model type is retrained on
     train+validation and given one final, only-looked-at-once evaluation on
     the held-out test split. That test-split score is what's reported as
     the model's real performance.

kaggle test.csv is never used for model selection (it has no labels) --
it is only used at the end to produce predictions for completeness.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, log_loss, brier_score_loss)
import xgboost as xgb

from data_loader import load_raw, TARGET_COL, ID_COL, TIMESTAMP_COL
from preprocessing import Winsorizer
from feature_engineering import TemporalFeatures

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Suspicious_Port_Activity", "Traffic_Volume_Variation", "Packet_Length_Anomaly",
    "Payload_Entropy", "Packet_Retry_Rate",
    "Malware_Score", "Threat_Level_Index", "Anomaly_Tendency",
    "User_Behavior_Score", "Login_Attempts", "Geo_Dispersion",
    "Device_Response_Time", "Session_Duration",
    "hour", "minute", "day", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
]


def build_preprocessor():
    """Shared preprocessing pipeline: temporal features + winsorization.
    Fit ONCE on the training split, reused everywhere else."""
    temporal = TemporalFeatures()
    winsor = Winsorizer()
    return temporal, winsor


def apply_preprocessor(df, temporal, winsor, fit=False):
    df = temporal.transform(df)
    if fit:
        winsor.fit(df)
    df = winsor.transform(df)
    return df[FEATURE_COLUMNS]


def split_data():
    train_raw, kaggle_test_raw = load_raw()
    X = train_raw.drop(columns=[TARGET_COL])
    y = train_raw[TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    return (X_train.reset_index(drop=True), y_train.reset_index(drop=True),
            X_val.reset_index(drop=True), y_val.reset_index(drop=True),
            X_test.reset_index(drop=True), y_test.reset_index(drop=True),
            kaggle_test_raw)


def multiclass_metrics(y_true, y_pred, y_proba, n_classes=3):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        metrics["roc_auc_ovr_macro"] = None
    try:
        metrics["log_loss"] = log_loss(y_true, y_proba, labels=list(range(n_classes)))
    except Exception:
        metrics["log_loss"] = None
    # multiclass Brier score = mean over classes of one-vs-rest brier
    try:
        y_true_oh = np.eye(n_classes)[y_true]
        briers = [brier_score_loss(y_true_oh[:, k], y_proba[:, k]) for k in range(n_classes)]
        metrics["brier_score"] = float(np.mean(briers))
    except Exception:
        metrics["brier_score"] = None
    return metrics


MODEL_FACTORY = {
    "MajorityBaseline": lambda: DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE,
                                                     class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=12,
                                                     random_state=RANDOM_STATE, n_jobs=-1,
                                                     class_weight="balanced"),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                          subsample=0.9, colsample_bytree=0.9,
                                          objective="multi:softprob", num_class=3,
                                          eval_metric="mlogloss", random_state=RANDOM_STATE,
                                          n_jobs=-1),
}


def fit_and_score(name, factory, X_train, y_train, X_eval, y_eval, needs_scaling=False, scaler=None):
    model = factory()
    Xt, Xe = X_train, X_eval
    if needs_scaling:
        if scaler is None:
            scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_train)
        Xe = scaler.transform(X_eval)
    t0 = time.time()
    model.fit(Xt, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(Xe)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xe)
    else:
        y_proba = np.eye(3)[y_pred]
    infer_time = (time.time() - t0) / max(len(Xe), 1)

    m = multiclass_metrics(y_eval, y_pred, y_proba)
    m["training_time_sec"] = train_time
    m["inference_time_sec_per_row"] = infer_time
    return model, scaler, m


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, kaggle_test_raw = split_data()

    temporal, winsor = build_preprocessor()
    Xtr = apply_preprocessor(X_train, temporal, winsor, fit=True)
    Xva = apply_preprocessor(X_val, temporal, winsor, fit=False)
    Xte = apply_preprocessor(X_test, temporal, winsor, fit=False)

    print(f"Split sizes -> train: {len(Xtr)}, val: {len(Xva)}, test: {len(Xte)}")

    # ---- Phase 1: compare all candidates on validation split ----
    val_results = {}
    fitted = {}
    for name, factory in MODEL_FACTORY.items():
        needs_scaling = name in ("LogisticRegression",)
        model, scaler, m = fit_and_score(name, factory, Xtr, y_train, Xva, y_val, needs_scaling)
        val_results[name] = m
        fitted[name] = (model, scaler, needs_scaling)
        print(f"[val] {name}: macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

    winner_name = max(val_results, key=lambda k: val_results[k]["macro_f1"])
    print(f"\nWinner on validation (Macro F1): {winner_name}")

    # ---- Phase 2: retrain winner on train+val, final untouched test evaluation ----
    X_trainval = pd.concat([Xtr, Xva], axis=0).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    final_results = {}
    final_models = {}
    for name, factory in MODEL_FACTORY.items():
        needs_scaling = name in ("LogisticRegression",)
        model, scaler, m = fit_and_score(name, factory, X_trainval, y_trainval, Xte, y_test, needs_scaling)
        final_results[name] = m
        final_models[name] = (model, scaler, needs_scaling)
        print(f"[test] {name}: macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

    comparison_df = pd.DataFrame(final_results).T
    comparison_df.index.name = "model"
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv")

    final_model, final_scaler, final_needs_scaling = final_models[winner_name]
    joblib.dump(final_model, MODELS_DIR / "final_model.joblib")
    joblib.dump({"temporal": temporal, "winsor": winsor, "feature_columns": FEATURE_COLUMNS,
                 "scaler": final_scaler, "needs_scaling": final_needs_scaling},
                MODELS_DIR / "preprocessor.joblib")

    with open(RESULTS_DIR / "model_selection.json", "w") as f:
        json.dump({
            "validation_results": val_results,
            "final_test_results": final_results,
            "selected_model": winner_name,
            "selection_reason": (
                f"{winner_name} achieved the highest Macro F1 on the held-out validation split "
                f"({val_results[winner_name]['macro_f1']:.4f}), which is the primary metric because "
                "classes are imbalanced (~50/25/25). Selection was not based on raw accuracy alone."
            ),
        }, f, indent=2, default=str)

    # persist the exact splits (feature-engineered) for downstream phases
    splits = {
        "X_train": Xtr, "y_train": y_train, "X_val": Xva, "y_val": y_val,
        "X_test": Xte, "y_test": y_test, "X_trainval": X_trainval, "y_trainval": y_trainval,
    }
    joblib.dump(splits, MODELS_DIR / "splits.joblib")

    # raw (pre-feature-engineering) versions too, needed for counterfactual/investigation display
    raw_splits = {"X_train_raw": X_train, "X_val_raw": X_val, "X_test_raw": X_test}
    joblib.dump(raw_splits, MODELS_DIR / "raw_splits.joblib")

    print(f"\nSaved final model ({winner_name}) and preprocessor.")
    return winner_name, comparison_df


if __name__ == "__main__":
    main()
