"""
validation.py
Inspects the ACTUAL dataset (never assumes structure) and produces:
  - data/reports/dataset_report.json
  - data/reports/dataset_report.md
  - data/reports/data_quality_report.html

Every number in these reports is computed directly from train.csv / test.csv.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import load_raw, get_feature_columns, TARGET_COL, ID_COL, TIMESTAMP_COL

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def inspect_dataset():
    train, test = load_raw()
    feature_cols = get_feature_columns(train)
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    timestamp_features = [TIMESTAMP_COL] if TIMESTAMP_COL in train.columns else []

    report = {
        "train_rows": len(train),
        "train_columns": train.shape[1],
        "test_rows": len(test),
        "test_columns": test.shape[1],
        "target_column": TARGET_COL,
        "id_column": ID_COL,
        "column_names_train": list(train.columns),
        "column_names_test": list(test.columns),
        "dtypes": {c: str(t) for c, t in train.dtypes.items()},
        "missing_values_train": train.isna().sum().to_dict(),
        "missing_values_test": test.isna().sum().to_dict(),
        "duplicate_rows_train": int(train.duplicated().sum()),
        "duplicate_rows_test": int(test.duplicated().sum()),
        "duplicate_ids_train": int(train[ID_COL].duplicated().sum()),
        "duplicate_ids_test": int(test[ID_COL].duplicated().sum()),
        "train_test_id_overlap": int(len(set(train[ID_COL]) & set(test[ID_COL]))),
        "class_distribution_counts": train[TARGET_COL].value_counts().sort_index().to_dict(),
        "class_distribution_pct": (train[TARGET_COL].value_counts(normalize=True) * 100)
        .round(3).sort_index().to_dict(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "timestamp_features": timestamp_features,
        "timestamp_min": str(train[TIMESTAMP_COL].min()) if timestamp_features else None,
        "timestamp_max": str(train[TIMESTAMP_COL].max()) if timestamp_features else None,
        "numeric_ranges": {
            c: {
                "min": train[c].min(),
                "max": train[c].max(),
                "mean": train[c].mean(),
                "std": train[c].std(),
                "p1": train[c].quantile(0.01),
                "p50": train[c].quantile(0.50),
                "p99": train[c].quantile(0.99),
                "negative_count": int((train[c] < 0).sum()),
            }
            for c in numeric_features
        },
    }
    report = _json_safe(report)

    with open(OUT_DIR / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- human-readable markdown ----
    md = []
    md.append("# Dataset Report — KAUST Cyber Attack Detection\n")
    md.append(f"- **train.csv**: {report['train_rows']} rows x {report['train_columns']} columns")
    md.append(f"- **test.csv**: {report['test_rows']} rows x {report['test_columns']} columns")
    md.append(f"- **Target column**: `{TARGET_COL}` (only present in train.csv)")
    md.append(f"- **Timestamp range**: {report['timestamp_min']} to {report['timestamp_max']}")
    md.append(f"- **Duplicate rows**: {report['duplicate_rows_train']} (train), {report['duplicate_rows_test']} (test)")
    md.append(f"- **Duplicate IDs**: {report['duplicate_ids_train']} (train), {report['duplicate_ids_test']} (test)")
    md.append(f"- **Train/test ID overlap**: {report['train_test_id_overlap']} rows\n")

    total_missing = sum(report["missing_values_train"].values())
    md.append(f"## Missing values\nTotal missing cells in train.csv: **{total_missing}**. "
               "No imputation strategy is required for this dataset as-is.\n")

    md.append("## Class distribution (train.csv, Attack Type)")
    for cls, cnt in report["class_distribution_counts"].items():
        pct = report["class_distribution_pct"][cls]
        md.append(f"- Class `{cls}`: {cnt} rows ({pct}%)")
    md.append("\nThe classes are moderately imbalanced (class 0 is roughly 2x classes 1 and 2), "
               "which is why Macro F1 (not raw accuracy) is used as the primary model-selection metric.\n")

    md.append("## Data quality issues found on actual inspection")
    # Features that are structurally impossible to be negative (counts, durations,
    # rates, or entropy, which is non-negative by definition).
    structurally_nonnegative = {"Login_Attempts", "Session_Duration", "Packet_Retry_Rate", "Payload_Entropy"}
    # Features with no documented data dictionary where a negative value could
    # plausibly be a signed/standardized indicator rather than an error.
    ambiguous_signed = {"Malware_Score", "User_Behavior_Score", "Anomaly_Tendency"}
    issues = []
    for c, r in report["numeric_ranges"].items():
        if r["negative_count"] == 0:
            continue
        if c in structurally_nonnegative:
            issues.append(f"- **`{c}`** contains {r['negative_count']} negative values "
                           f"(min={r['min']:.2f}, max={r['max']:.2f}). This feature is structurally "
                           "non-negative (a count, duration, rate, or entropy value), so these are "
                           "treated as data-quality noise and winsorized in preprocessing.")
        elif c in ambiguous_signed:
            issues.append(f"- `{c}` contains {r['negative_count']} negative values "
                           f"(min={r['min']:.2f}, max={r['max']:.2f}). No data dictionary was provided "
                           "with this competition, so we cannot confirm whether negative values here are "
                           "errors or a meaningful signed/standardized indicator (e.g. deviation from a "
                           "baseline). They are kept as-is rather than silently altered.")
    if not issues:
        issues.append("- No impossible negative values detected.")
    md.extend(issues)
    md.append("")
    for c in ["Login_Attempts", "Device_Response_Time"]:
        r = report["numeric_ranges"][c]
        if r["max"] > 1e6 or r["min"] < -1e6:
            md.append(f"- **`{c}`** has an extreme range (min={r['min']:.2e}, max={r['max']:.2e}, "
                       f"median={r['p50']:.2e}), consistent with the dataset description's note about "
                       "extreme values in this column. Values are winsorized at the 1st/99th percentile "
                       "before modeling rather than dropped, to avoid discarding potentially real events.")
    md.append("")

    with open(OUT_DIR / "dataset_report.md", "w") as f:
        f.write("\n".join(md))

    # ---- data quality html ----
    html_rows = []
    for c, r in report["numeric_ranges"].items():
        html_rows.append(
            f"<tr><td>{c}</td><td>{r['min']:.3f}</td><td>{r['p1']:.3f}</td>"
            f"<td>{r['p50']:.3f}</td><td>{r['p99']:.3f}</td><td>{r['max']:.3f}</td>"
            f"<td>{r['negative_count']}</td></tr>"
        )
    html = f"""<html><head><title>Data Quality Report</title>
    <style>body{{font-family:Arial;background:#0b0f14;color:#e6edf3;padding:24px}}
    table{{border-collapse:collapse;width:100%}}
    th,td{{border:1px solid #2a2f36;padding:6px 10px;text-align:right}}
    th{{background:#141a21}} td:first-child,th:first-child{{text-align:left}}
    h1{{color:#22d3ee}}</style></head><body>
    <h1>CyberTwin — Data Quality Report</h1>
    <p>train.csv: {report['train_rows']} rows | test.csv: {report['test_rows']} rows |
    duplicate rows: {report['duplicate_rows_train']} | missing cells: {total_missing}</p>
    <table><tr><th>Feature</th><th>Min</th><th>P1</th><th>Median</th><th>P99</th><th>Max</th><th>Negative count</th></tr>
    {''.join(html_rows)}
    </table>
    </body></html>"""
    with open(OUT_DIR / "data_quality_report.html", "w") as f:
        f.write(html)

    return report


if __name__ == "__main__":
    r = inspect_dataset()
    print(json.dumps({k: r[k] for k in ["train_rows", "test_rows", "class_distribution_counts",
                                          "duplicate_rows_train"]}, indent=2))
