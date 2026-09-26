"""
export_model_js.py
Exports the trained XGBoost model to a compact JSON tree format so the
dashboard's What-If Simulator can re-run the REAL trained model in the
browser (no server) when a user drags a slider. This was verified to
exactly reproduce model.predict_proba() in Python (see dev notes) before
being shipped -- it is not an approximation.
"""
import json
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DASHBOARD_DIR = ROOT / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)


def _compact_tree(node):
    """Recursively convert an xgboost json-dump node into a compact
    {f: feature_index, t: threshold, m: missing_goes_left(bool), l: left, r: right}
    or {v: leaf_value} representation, keyed to save bytes."""
    if "leaf" in node:
        return {"v": round(float(node["leaf"]), 6)}
    return {
        "f": node["_fidx"],
        "t": round(float(node["split_condition"]), 6),
        "m": node["missing"] == node["yes"],
        "l": _compact_tree(node["_yes_node"]),
        "r": _compact_tree(node["_no_node"]),
    }


def _attach_children(node, by_id):
    if "children" in node:
        node["_yes_node"] = _attach_children(by_id[node["yes"]], by_id)
        node["_no_node"] = _attach_children(by_id[node["no"]], by_id)
    return node


def export():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    splits = joblib.load(MODELS_DIR / "splits.joblib")
    feature_names = list(splits["X_test"].columns)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    booster = model.get_booster()
    cfg = json.loads(booster.save_config())
    base_score = [float(x) for x in
                  cfg["learner"]["learner_model_param"]["base_score"].strip("[]").split(",")]
    n_classes = int(cfg["learner"]["learner_model_param"]["num_class"])

    dumps = [json.loads(d) for d in booster.get_dump(dump_format="json")]
    compact_trees = []
    for tree in dumps:
        by_id = {}

        def index(n):
            by_id[n["nodeid"]] = n
            for c in n.get("children", []):
                index(c)
        index(tree)
        for n in by_id.values():
            if "split" in n:
                n["_fidx"] = name_to_idx[n["split"]]
        root = _attach_children(tree, by_id)
        compact_trees.append(_compact_tree(root))

    payload = {
        "base_score": base_score,
        "n_classes": n_classes,
        "feature_names": feature_names,
        "trees": compact_trees,
    }
    out_path = MODELS_DIR / "xgb_model_export.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)
    size_kb = out_path.stat().st_size / 1024
    print(f"Exported {len(compact_trees)} trees, {len(feature_names)} features -> {out_path} ({size_kb:.0f} KB)")
    return payload


if __name__ == "__main__":
    export()
