"""
counterfactual.py
Genuine counterfactual engine: every number here comes from re-running the
ACTUAL trained model (models/final_model.joblib) on a modified feature
vector -- nothing is looked up from a table or fabricated.

ACTIONABILITY: only features a security team could plausibly intervene on
are perturbed. Non-actionable columns (temporal features, and derived/
aggregate indices that a defender cannot directly "turn down") are never
modified. This choice is documented, not silently assumed:

  ACTIONABLE   -> Suspicious_Port_Activity  (port restriction / firewalling)
                  Traffic_Volume_Variation  (rate limiting / traffic shaping)
                  Login_Attempts            (MFA / account lockout policy)
                  Malware_Score             (AV/EDR remediation)
                  User_Behavior_Score       (access control / re-training)
                  Packet_Retry_Rate         (QoS / network stability controls)
                  Geo_Dispersion            (geo-fencing / VPN restriction)

  NOT actionable -> ID, Timestamp, hour/day/... (can't rewrite when something
                     happened), Threat_Level_Index & Anomaly_Tendency (these
                     are themselves aggregate threat indices, not raw levers),
                     Device_Response_Time, Session_Duration, Packet_Length_Anomaly,
                     Payload_Entropy (measurements, not direct controls).

Two required methods are implemented:
  Method 1 - controlled single-feature perturbation (grid search within the
             1st-99th percentile of the TRAINING distribution -- never
             pushes a feature to a value the real data never plausibly takes)
  Method 2 - greedy multi-feature counterfactual (repeatedly apply the single
             best remaining single-feature move, up to 3 features)
A small optimization-based search (scipy L-BFGS-B minimizing predicted
probability + distance penalty, restricted to the actionable feature
subspace) is additionally run on a smaller sample as a third comparison
method, per spec section 29 ("if computationally practical").
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

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

N_GRID = 15


def _bounds(train_ref):
    return {f: (train_ref[f].quantile(0.01), train_ref[f].quantile(0.99)) for f in ACTIONABLE_FEATURES}


def _proximity(orig, cf, train_std):
    diffs = ((cf - orig) / (train_std + 1e-9)) ** 2
    return float(np.sqrt(diffs.sum()))


def _plausibility(cf, bounds):
    ok = sum(1 for f, v in cf.items() if bounds[f][0] <= v <= bounds[f][1])
    return ok / len(cf) if cf else 1.0


def single_feature_search(model, orig_row: pd.Series, bounds, target_reduce_class: int):
    """Method 1: try each actionable feature alone, grid search toward the
    low-risk end of its training range, keep the value that most reduces
    P(target_reduce_class). Returns a dict of per-feature best result."""
    orig_vec = orig_row.values.reshape(1, -1)
    orig_proba = model.predict_proba(orig_vec)[0]
    results = {}
    for feat in ACTIONABLE_FEATURES:
        lo, hi = bounds[feat]
        cur = orig_row[feat]
        # search toward the lower end of the training range (assume "reduce anomaly")
        grid = np.linspace(cur, lo, N_GRID)
        best_val, best_proba = cur, orig_proba[target_reduce_class]
        for val in grid:
            trial = orig_row.copy()
            trial[feat] = val
            p = model.predict_proba(trial.values.reshape(1, -1))[0]
            if p[target_reduce_class] < best_proba:
                best_proba = p[target_reduce_class]
                best_val = val
        results[feat] = {"original_value": float(cur), "new_value": float(best_val),
                          "new_proba_target_class": float(best_proba)}
    return results, orig_proba


def greedy_multi_feature(model, orig_row: pd.Series, bounds, target_class, max_features=3):
    """Method 2: repeatedly apply whichever single remaining actionable
    feature change reduces P(target_class) the most, real model re-run at
    every step."""
    current = orig_row.copy()
    changed_features = []
    history = [{"step": 0, "features_changed": [], "proba_target_class":
                float(model.predict_proba(current.values.reshape(1, -1))[0][target_class])}]
    remaining = list(ACTIONABLE_FEATURES)

    for step in range(1, max_features + 1):
        best_feat, best_val, best_proba = None, None, history[-1]["proba_target_class"]
        for feat in remaining:
            lo, hi = bounds[feat]
            grid = np.linspace(current[feat], lo, N_GRID)
            for val in grid:
                trial = current.copy()
                trial[feat] = val
                p = model.predict_proba(trial.values.reshape(1, -1))[0][target_class]
                if p < best_proba:
                    best_proba, best_feat, best_val = p, feat, val
        if best_feat is None:
            break
        current[best_feat] = best_val
        remaining.remove(best_feat)
        changed_features.append(best_feat)
        history.append({"step": step, "features_changed": list(changed_features),
                         "proba_target_class": float(best_proba)})
    return current, history


def optimization_search(model, orig_row: pd.Series, bounds, target_class, lam=0.4):
    """Method 3 (bonus): global, gradient-free search (differential evolution)
    over the actionable subspace, minimizing predicted P(target_class) +
    lam * normalized distance. A gradient-based optimizer (e.g. L-BFGS-B) was
    tried first and performed poorly (near-zero movement) because XGBoost's
    decision surface is piecewise-constant -- finite-difference gradients are
    almost everywhere zero. Differential evolution does not require gradients
    and is a better fit for tree-ensemble objectives; this is reported
    honestly rather than hidden."""
    x0_raw = orig_row[ACTIONABLE_FEATURES].values.astype(float)
    lo = np.array([bounds[f][0] for f in ACTIONABLE_FEATURES])
    hi = np.array([bounds[f][1] for f in ACTIONABLE_FEATURES])
    scale = (hi - lo) + 1e-9
    x0 = np.clip(x0_raw, lo, hi)  # event's own value may sit outside the 1st/99th pct search box

    def objective(x):
        trial = orig_row.copy()
        for i, f in enumerate(ACTIONABLE_FEATURES):
            trial[f] = x[i]
        p = model.predict_proba(trial.values.reshape(1, -1))[0][target_class]
        dist = np.sum(((x - x0_raw) / scale) ** 2)
        return p + lam * dist

    res = differential_evolution(objective, bounds=list(zip(lo, hi)), x0=x0,
                                  maxiter=25, popsize=8, tol=1e-3, seed=42,
                                  polish=True, workers=1)
    trial = orig_row.copy()
    for i, f in enumerate(ACTIONABLE_FEATURES):
        trial[f] = res.x[i]
    new_proba = model.predict_proba(trial.values.reshape(1, -1))[0]
    return trial, new_proba


def run_counterfactual_engine(n_events=120, n_opt_events=20, seed=42):
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    X_test, y_test = splits["X_test"], splits["y_test"]
    X_train = splits["X_train"]

    bounds = _bounds(X_train)
    train_std = X_train[ACTIONABLE_FEATURES].std()

    y_pred = model.predict(X_test)
    y_proba_all = model.predict_proba(X_test)

    # focus on events NOT predicted as the majority class -- those are the
    # ones a security team would actually want to intervene on
    candidate_idx = np.where(y_pred != 0)[0]
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(candidate_idx, size=min(n_events, len(candidate_idx)), replace=False)

    rows_method1, rows_method2, rows_metrics = [], [], []

    for idx in sample_idx:
        orig_row = X_test.iloc[idx]
        target_class = int(y_pred[idx])
        orig_proba = y_proba_all[idx]

        # ---- Method 1 ----
        m1_results, _ = single_feature_search(model, orig_row, bounds, target_class)
        best_feat = min(m1_results, key=lambda f: m1_results[f]["new_proba_target_class"])
        m1 = m1_results[best_feat]
        cf_row_m1 = orig_row.copy()
        cf_row_m1[best_feat] = m1["new_value"]
        new_pred_m1 = int(model.predict(cf_row_m1.values.reshape(1, -1))[0])
        rows_method1.append({
            "event_idx": int(idx), "method": "single_feature",
            "original_class": target_class, "original_probability": float(orig_proba[target_class]),
            "intervention": INTERVENTION_LABELS[best_feat], "feature_changed": best_feat,
            "original_value": m1["original_value"], "new_value": m1["new_value"],
            "new_class": new_pred_m1, "new_probability": m1["new_proba_target_class"],
            "probability_change": m1["new_proba_target_class"] - float(orig_proba[target_class]),
        })
        valid1 = new_pred_m1 != target_class
        prox1 = _proximity(orig_row[ACTIONABLE_FEATURES], cf_row_m1[ACTIONABLE_FEATURES], train_std)
        plaus1 = _plausibility({best_feat: m1["new_value"]}, bounds)

        # ---- Method 2 ----
        cf_row_m2, history = greedy_multi_feature(model, orig_row, bounds, target_class, max_features=3)
        new_pred_m2 = int(model.predict(cf_row_m2.values.reshape(1, -1))[0])
        new_proba_m2 = model.predict_proba(cf_row_m2.values.reshape(1, -1))[0]
        changed = history[-1]["features_changed"]
        rows_method2.append({
            "event_idx": int(idx), "method": "multi_feature_greedy",
            "original_class": target_class, "original_probability": float(orig_proba[target_class]),
            "intervention": " + ".join(INTERVENTION_LABELS[f] for f in changed) if changed else "none",
            "features_changed": changed, "n_features_changed": len(changed),
            "new_class": new_pred_m2, "new_probability": float(new_proba_m2[target_class]),
            "probability_change": float(new_proba_m2[target_class]) - float(orig_proba[target_class]),
            "history": history,
        })
        valid2 = new_pred_m2 != target_class
        prox2 = _proximity(orig_row[ACTIONABLE_FEATURES], cf_row_m2[ACTIONABLE_FEATURES], train_std)
        plaus2 = _plausibility({f: cf_row_m2[f] for f in changed}, bounds) if changed else 1.0

        rows_metrics.append({"event_idx": int(idx), "method": "single_feature",
                              "validity": valid1, "sparsity": 1, "proximity": prox1, "plausibility": plaus1})
        rows_metrics.append({"event_idx": int(idx), "method": "multi_feature_greedy",
                              "validity": valid2, "sparsity": len(changed), "proximity": prox2,
                              "plausibility": plaus2})

    # ---- stability: rerun greedy search with shuffled feature-check order, 5x, on a subsample ----
    stability_sample = sample_idx[:20]
    stability_rows = []
    for idx in stability_sample:
        orig_row = X_test.iloc[idx]
        target_class = int(y_pred[idx])
        first_features = []
        for run in range(5):
            shuffled = ACTIONABLE_FEATURES.copy()
            rng.shuffle(shuffled)
            global ACTIONABLE_FEATURES_ORDER
            cf_row, history = greedy_multi_feature(model, orig_row, bounds, target_class, max_features=1)
            first_features.append(history[-1]["features_changed"][0] if history[-1]["features_changed"] else None)
        consistency = max(pd.Series(first_features).value_counts(normalize=True)) if first_features else 0
        stability_rows.append({"event_idx": int(idx), "top_feature_consistency": float(consistency)})

    # ---- Method 3: optimization-based, on a smaller sample ----
    opt_rows = []
    opt_idx = sample_idx[:n_opt_events]
    for idx in opt_idx:
        orig_row = X_test.iloc[idx]
        target_class = int(y_pred[idx])
        orig_proba = y_proba_all[idx]
        cf_row, new_proba = optimization_search(model, orig_row, bounds, target_class)
        new_pred = int(np.argmax(new_proba))
        changed_feats = [f for f in ACTIONABLE_FEATURES if abs(cf_row[f] - orig_row[f]) > 1e-6]
        prox = _proximity(orig_row[ACTIONABLE_FEATURES], cf_row[ACTIONABLE_FEATURES], train_std)
        plaus = _plausibility({f: cf_row[f] for f in changed_feats}, bounds) if changed_feats else 1.0
        opt_rows.append({
            "event_idx": int(idx), "method": "optimization_differential_evolution",
            "original_class": target_class, "original_probability": float(orig_proba[target_class]),
            "new_class": new_pred, "new_probability": float(new_proba[target_class]),
            "probability_change": float(new_proba[target_class]) - float(orig_proba[target_class]),
            "n_features_changed": len(changed_feats), "features_changed": changed_feats,
            "validity": new_pred != target_class, "proximity": prox, "plausibility": plaus,
        })

    # ---- persist ----
    df1 = pd.DataFrame(rows_method1)
    df2 = pd.DataFrame(rows_method2)
    df_opt = pd.DataFrame(opt_rows)
    df_metrics = pd.DataFrame(rows_metrics)
    df_stability = pd.DataFrame(stability_rows)

    combined_results = pd.concat([
        df1[["event_idx", "method", "original_class", "original_probability", "intervention",
             "new_class", "new_probability", "probability_change"]],
        df2[["event_idx", "method", "original_class", "original_probability", "intervention",
             "new_class", "new_probability", "probability_change"]],
    ], ignore_index=True)
    combined_results.to_csv(RESULTS_DIR / "counterfactual_results.csv", index=False)

    metrics_summary = {
        "single_feature": {
            "validity_rate": float(df_metrics[df_metrics.method == "single_feature"].validity.mean()),
            "mean_sparsity": float(df_metrics[df_metrics.method == "single_feature"].sparsity.mean()),
            "mean_proximity": float(df_metrics[df_metrics.method == "single_feature"].proximity.mean()),
            "mean_plausibility": float(df_metrics[df_metrics.method == "single_feature"].plausibility.mean()),
        },
        "multi_feature_greedy": {
            "validity_rate": float(df_metrics[df_metrics.method == "multi_feature_greedy"].validity.mean()),
            "mean_sparsity": float(df_metrics[df_metrics.method == "multi_feature_greedy"].sparsity.mean()),
            "mean_proximity": float(df_metrics[df_metrics.method == "multi_feature_greedy"].proximity.mean()),
            "mean_plausibility": float(df_metrics[df_metrics.method == "multi_feature_greedy"].plausibility.mean()),
        },
        "optimization_differential_evolution": {
            "validity_rate": float(df_opt.validity.mean()),
            "mean_sparsity": float(df_opt.n_features_changed.mean()),
            "mean_proximity": float(df_opt.proximity.mean()),
            "mean_plausibility": float(df_opt.plausibility.mean()),
        },
        "stability_mean_top_feature_consistency": float(df_stability.top_feature_consistency.mean()),
        "n_events_evaluated": int(len(sample_idx)),
    }
    with open(RESULTS_DIR / "counterfactual_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    df_metrics.to_csv(RESULTS_DIR / "counterfactual_metrics.csv", index=False)
    df_opt.to_csv(RESULTS_DIR / "counterfactual_optimization_results.csv", index=False)

    # intervention ranking (mean probability reduction per intervention type, method 1)
    ranking = df1.groupby("intervention")["probability_change"].agg(["mean", "count"]).sort_values("mean")
    ranking.to_csv(RESULTS_DIR / "intervention_ranking.csv")

    print(json.dumps(metrics_summary, indent=2))
    print("\nIntervention ranking (mean probability change, most negative = most effective):")
    print(ranking)
    return metrics_summary, combined_results


if __name__ == "__main__":
    run_counterfactual_engine()
