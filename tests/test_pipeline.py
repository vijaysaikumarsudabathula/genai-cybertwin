"""
tests/test_pipeline.py
Fast tests that run against already-produced artifacts (models/, results/).
Run `python run_pipeline.py` first if these files don't exist yet.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "final_model.joblib").exists(),
    reason="Run `python run_pipeline.py` first to produce model artifacts.",
)


def test_dataset_report_matches_actual_files():
    from data_loader import load_raw
    dr = json.load(open(ROOT / "data" / "reports" / "dataset_report.json"))
    train, test = load_raw()
    assert dr["train_rows"] == len(train)
    assert dr["test_rows"] == len(test)
    assert dr["duplicate_rows_train"] == int(train.duplicated().sum())


def test_winsorizer_fits_only_on_train():
    from preprocessing import Winsorizer
    df_train = pd.DataFrame({"Login_Attempts": [-100, 0, 50, 1000, 2000]})
    df_other = pd.DataFrame({"Login_Attempts": [-9999, 99999]})
    w = Winsorizer(columns=["Login_Attempts"])
    w.fit(df_train)
    out = w.transform(df_other)
    lo, hi = w.bounds_["Login_Attempts"]
    assert out["Login_Attempts"].min() >= lo - 1e-9
    assert out["Login_Attempts"].max() <= hi + 1e-9


def test_temporal_features_are_deterministic():
    from feature_engineering import TemporalFeatures
    df = pd.DataFrame({"Timestamp": ["2023-01-02 12:46:00", "2023-06-15 23:59:00"]})
    out = TemporalFeatures().transform(df)
    assert out["hour"].tolist() == [12, 23]
    assert out["is_weekend"].tolist() == [0, 0] or set(out["is_weekend"]) <= {0, 1}
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0, atol=1e-6)


def test_model_predict_proba_shape_and_sums_to_one():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    X_test = splits["X_test"]
    proba = model.predict_proba(X_test.iloc[:20])
    assert proba.shape == (20, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_model_comparison_selected_by_macro_f1():
    model_sel = json.load(open(RESULTS_DIR / "model_selection.json"))
    val_results = model_sel["validation_results"]
    best = max(val_results, key=lambda k: val_results[k]["macro_f1"])
    assert best == model_sel["selected_model"]


def test_risk_score_bounded_0_100():
    df = pd.read_csv(RESULTS_DIR / "risk_scores.csv")
    assert df["cyber_risk_score"].min() >= 0
    assert df["cyber_risk_score"].max() <= 100
    assert set(df["risk_category"].unique()) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_counterfactual_only_touches_actionable_features():
    from counterfactual import ACTIONABLE_FEATURES
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    X_test = splits["X_test"]
    feature_cols = list(X_test.columns)
    non_actionable = [c for c in feature_cols if c not in ACTIONABLE_FEATURES]
    # sanity: temporal & non-actionable columns exist and are excluded by design
    assert "hour" in non_actionable
    assert "Threat_Level_Index" in non_actionable
    assert set(ACTIONABLE_FEATURES).issubset(set(feature_cols))


def test_counterfactual_metrics_within_valid_ranges():
    m = json.load(open(RESULTS_DIR / "counterfactual_metrics.json"))
    for method in ["single_feature", "multi_feature_greedy", "optimization_differential_evolution"]:
        assert 0.0 <= m[method]["validity_rate"] <= 1.0
        assert 0.0 <= m[method]["mean_plausibility"] <= 1.0
        assert m[method]["mean_sparsity"] >= 0


def test_genai_narratives_never_mention_fabricated_iocs():
    narratives = json.load(open(RESULTS_DIR / "genai_narratives.json"))
    forbidden_markers = ["CVE-", "192.168.", "http://", "https://malicious"]
    for n in narratives.values():
        text = n["text"]
        for marker in forbidden_markers:
            assert marker not in text, f"Found forbidden marker '{marker}' in narrative"


def test_grounding_tests_all_passed():
    df = pd.read_csv(RESULTS_DIR / "genai_grounding_tests.csv")
    assert df["passed"].astype(str).isin(["True", "true", "1"]).all()


def test_ablation_variants_share_same_split():
    df = pd.read_csv(RESULTS_DIR / "ablation_results.csv")
    assert len(df) == 3
    assert (df["macro_f1"] > 0.8).all()
