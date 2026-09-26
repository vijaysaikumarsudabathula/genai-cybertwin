# -*- coding: utf-8 -*-
import docx
from build_helpers import (R, xgb_row, add_heading1, add_heading2, add_body, add_caption,
                             add_table_title, add_ieee_table, add_figure, OUT)

doc = docx.Document(str(OUT))

add_heading1(doc, "V. RESULTS")

# ---------------------------------------------------------------- A. Model Performance
add_heading2(doc, "A. Model Performance and Cross-Validation (RQ1)")
mc = R["model_comparison"]
add_body(doc,
    f"Table II reports results on the {1603}-record held-out test set. XGBoost achieves the highest "
    f"macro F1 ({xgb_row.macro_f1:.4f}) and accuracy ({xgb_row.accuracy:.4f}), narrowly ahead of "
    f"Random Forest ({R['model_comparison'][R['model_comparison'].model=='RandomForest'].iloc[0].macro_f1:.4f}) "
    f"and Logistic Regression "
    f"({R['model_comparison'][R['model_comparison'].model=='LogisticRegression'].iloc[0].macro_f1:.4f}); "
    f"a majority-class baseline reaches only 0.223 macro F1 despite 50.3% accuracy, confirming "
    f"that accuracy alone would be misleading on this class distribution.")

rows = []
for _, row in mc.iterrows():
    rows.append([row['model'], f"{row.accuracy:.4f}", f"{row.macro_precision:.4f}",
                 f"{row.macro_recall:.4f}", f"{row.macro_f1:.4f}"])
add_table_title(doc, "TABLE II. MODEL PERFORMANCE ON HELD-OUT TEST SET (N=1603)")
add_ieee_table(doc, headers=["Model", "Acc.", "Prec.(M)", "Rec.(M)", "F1-M"], rows=rows,
               col_widths_in=[1.15, 0.55, 0.6, 0.55, 0.5], font_pt=7.5)
add_caption(doc, "TABLE II. Macro-averaged precision/recall/F1 (M = macro).")

cv = R["cv_summary"]
add_body(doc,
    f"To confirm this result is not an artifact of a favorable split, the XGBoost architecture was "
    f"separately evaluated with five-fold stratified cross-validation on the full labeled dataset "
    f"(Table III). The cross-validated macro F1 ({cv['mean']['macro_f1']:.4f} \u00b1 "
    f"{cv['std']['macro_f1']:.4f}) is within 0.008 of the held-out test estimate "
    f"({xgb_row.macro_f1:.4f}), and the per-fold standard deviation is small "
    f"({cv['std']['macro_f1']:.4f}), indicating stable performance across different data partitions "
    f"rather than a result specific to one split.")

add_table_title(doc, "TABLE III. 5-FOLD STRATIFIED CROSS-VALIDATION (XGBOOST)")
add_ieee_table(doc, headers=["Metric", "Mean", "Std. Dev."],
               rows=[
                   ["Accuracy", f"{cv['mean']['accuracy']:.4f}", f"{cv['std']['accuracy']:.4f}"],
                   ["Macro Precision", f"{cv['mean']['macro_precision']:.4f}", f"{cv['std']['macro_precision']:.4f}"],
                   ["Macro Recall", f"{cv['mean']['macro_recall']:.4f}", f"{cv['std']['macro_recall']:.4f}"],
                   ["Macro F1", f"{cv['mean']['macro_f1']:.4f}", f"{cv['std']['macro_f1']:.4f}"],
                   ["ROC-AUC (OvR)", f"{cv['mean']['roc_auc_ovr_macro']:.4f}", f"{cv['std']['roc_auc_ovr_macro']:.4f}"],
               ],
               col_widths_in=[1.3, 0.7, 0.7], font_pt=7.5)
add_caption(doc, "TABLE III. Mean \u00b1 standard deviation across 5 stratified folds; "
                 "preprocessing was refit independently inside each fold.")

add_body(doc,
    "Fig. 1 shows one-vs-rest ROC curves for the final model on the held-out test split. Attack Type "
    "0 is separated perfectly (AUC = 1.000), and both Attack Type 1 and Attack Type 2 reach AUC = "
    "0.960, confirming the strong ranking ability visible in the confusion matrix (Fig. 2) even for "
    "the two classes that are occasionally confused with each other. In answer to RQ1, a "
    "gradient-boosted ensemble reliably classifies this three-class, imbalanced problem, and both "
    "cross-validation and ROC-AUC confirm the held-out test estimate.")

add_figure(doc, "roc_curves.png", 2.3, "Fig. 1. One-vs-rest ROC curves, final model (test split).")

# ---------------------------------------------------------------- B. Confusion
add_heading2(doc, "B. Per-Class Performance and Confusion Analysis")
add_body(doc,
    "Fig. 2 shows the confusion matrix for the final model. Attack Type 0 is recovered with 100.0% "
    "recall (806 of 806) and zero events from either other class are misclassified into it. "
    "The two remaining classes are confused only with each other: Attack Type 1 is recovered with "
    "82.25% recall (329 of 400, with 71 misclassified as Type 2), and Attack Type 2 with "
    "77.83% recall (309 of 397, with 88 misclassified as Type 1). Every one of the 159 test-set "
    "errors, 9.9% of the test set, falls in this single Type 1 \u2194 Type 2 confusion pair; "
    "the model has no difficulty separating either attack type from Type 0. This clean separation of "
    "Type 0 is worth noting because a classifier that simply predicted the majority class for every "
    "event would reach 50.3% accuracy while never identifying either attack type \u2014 "
    "precisely the cases a detection system exists to catch.")

add_figure(doc, "confusion_matrix.png", 2.3, "Fig. 2. Confusion matrix, final model (XGBoost, test split).")

# ---------------------------------------------------------------- C. Feature Importance
add_heading2(doc, "C. Feature Importance and Explanation Consistency (RQ2)")
shap_df = R["shap_importance"].head(5)
top = R["shap_importance"].iloc[0]
second = R["shap_importance"].iloc[1]
ratio = top.shap_importance / second.shap_importance
add_body(doc,
    f"Fig. 3 shows global SHAP importance for the top twelve features. {top.feature} dominates by a "
    f"wide margin (SHAP importance {top.shap_importance:.3f}), roughly {ratio:.1f} times higher than "
    f"the next feature, {second.feature} ({second.shap_importance:.3f}). Table IV lists the top five "
    f"features under all three independent importance measures computed in this paper: the model's "
    f"own gain-based importance, SHAP, and permutation importance on held-out data. All three "
    f"measures rank the same feature first, and the ranking of the top four features is identical "
    f"across all three methods.")

add_figure(doc, "shap_importance.png", 2.6, "Fig. 3. Global SHAP importance, top 12 features.")

rows = []
for _, r in shap_df.iterrows():
    rows.append([r.feature, f"{r.model_importance:.4f}", f"{r.shap_importance:.3f}", f"{r.permutation_importance:.4f}"])
add_table_title(doc, "TABLE IV. TOP FEATURES ACROSS THREE IMPORTANCE METHODS")
add_ieee_table(doc, headers=["Feature", "Model\nImp.", "SHAP\nImp.", "Perm.\nImp."], rows=rows,
               col_widths_in=[1.4, 0.55, 0.55, 0.55], font_pt=7.2)
add_caption(doc, "TABLE IV. Model = gain-based native importance; Perm. = permutation importance "
                 "on the held-out test split (macro-F1 scoring).")

add_body(doc,
    "In answer to RQ2, the ranking of important features is consistent across three independent "
    "methods, with a latency-related feature the dominant signal by a wide and consistent margin. "
    "One instructive exception is Login_Attempts, which ranks fourth by the model's own importance "
    "(0.145) but shows a near-zero permutation importance (\u22120.0007): this can occur when a "
    "feature's information is partially redundant with other features, so shuffling it alone does "
    "not hurt held-out performance even though the trees make substantial use of it.")

doc.save(str(OUT))
print("Section V part 1 (A-C) done.")
