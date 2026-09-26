"""
make_report_figures.py
Generates static PNG charts (matplotlib) for the PDF report, using the same
underlying real result files as the interactive dashboard.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "text.color": "#111111", "xtick.color": "#333333", "ytick.color": "#333333",
    "font.size": 10, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
})
PALETTE = ["#0e7c86", "#4c5fd5", "#c0567a", "#e08e3e", "#5aa469"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison():
    df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["accuracy"], width=w, label="Accuracy", color=PALETTE[0])
    ax.bar(x + w/2, df["macro_f1"], width=w, label="Macro F1", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Model comparison (held-out test split)")
    save(fig, "model_comparison.png")


def fig_confusion_matrix():
    ev = json.load(open(RESULTS_DIR / "evaluation_full.json"))
    cm = np.array(ev["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=11)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Pred 0", "Pred 1", "Pred 2"])
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Actual 0", "Actual 1", "Actual 2"])
    ax.set_title("Confusion matrix (XGBoost, test split)")
    save(fig, "confusion_matrix.png")


def fig_roc():
    ev = json.load(open(RESULTS_DIR / "evaluation_full.json"))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for i, (k, v) in enumerate(ev["roc"].items()):
        ax.plot(v["fpr"], v["tpr"], label=f"Type {k} (AUC {v['auc']:.3f})", color=PALETTE[i])
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)"); ax.legend(fontsize=8)
    save(fig, "roc_curves.png")


def fig_shap_importance():
    df = pd.read_csv(RESULTS_DIR / "shap_importance.csv").sort_values("shap_importance").tail(12)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["feature"], df["shap_importance"], color=PALETTE[0])
    ax.set_title("Global SHAP importance (top 12 features)")
    save(fig, "shap_importance.png")


def fig_risk_distribution():
    summary = json.load(open(RESULTS_DIR / "risk_engine_summary.json"))
    counts = summary["category_counts"]
    cats = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    vals = [counts.get(c, 0) for c in cats]
    colors = ["#5aa469", "#e0b23e", "#e08e3e", "#c0433f"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(cats, vals, color=colors)
    ax.set_title("Cyber Risk category distribution (test split)")
    save(fig, "risk_distribution.png")


def fig_ablation():
    df = pd.read_csv(RESULTS_DIR / "ablation_results.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(df["variant"], df["macro_f1"], color=PALETTE[1])
    ax.set_ylim(0.85, 0.88)
    ax.set_title("Ablation study — Macro F1")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, "ablation.png")


def fig_intervention_ranking():
    df = pd.read_csv(RESULTS_DIR / "intervention_ranking.csv")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = [PALETTE[0] if v < 0 else PALETTE[3] for v in df["mean"]]
    ax.barh(df["intervention"], df["mean"], color=colors)
    ax.set_title("Intervention ranking — mean probability change")
    ax.set_xlabel("mean Δ probability (negative = risk reduced)")
    save(fig, "intervention_ranking.png")


def fig_counterfactual_methods():
    m = json.load(open(RESULTS_DIR / "counterfactual_metrics.json"))
    methods = ["single_feature", "multi_feature_greedy", "optimization_differential_evolution"]
    labels = ["Method 1\n(single feature)", "Method 2\n(multi-feature greedy)", "Method 3\n(diff. evolution)"]
    validity = [m[k]["validity_rate"] for k in methods]
    sparsity = [m[k]["mean_sparsity"] for k in methods]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    axes[0].bar(labels, validity, color=PALETTE[0]); axes[0].set_title("Validity rate"); axes[0].set_ylim(0, 1)
    axes[1].bar(labels, sparsity, color=PALETTE[1]); axes[1].set_title("Mean sparsity (features changed)")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=10, ha="right", fontsize=8)
    save(fig, "counterfactual_methods.png")


def fig_class_distribution():
    dr = json.load(open(ROOT / "data" / "reports" / "dataset_report.json"))
    counts = dr["class_distribution_counts"]
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.pie(counts.values(), labels=[f"Type {k}\n({v})" for k, v in counts.items()],
           colors=PALETTE[:3], autopct="%1.1f%%", startangle=90)
    ax.set_title("Attack Type distribution (train.csv)")
    save(fig, "class_distribution.png")


if __name__ == "__main__":
    fig_model_comparison()
    fig_confusion_matrix()
    fig_roc()
    fig_shap_importance()
    fig_risk_distribution()
    fig_ablation()
    fig_intervention_ranking()
    fig_counterfactual_methods()
    fig_class_distribution()
    print("Figures written to", FIG_DIR)
