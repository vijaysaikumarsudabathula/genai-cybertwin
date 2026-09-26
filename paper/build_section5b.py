# -*- coding: utf-8 -*-
import docx
from build_helpers import (R, add_heading2, add_body, add_caption, add_figure, OUT)

doc = docx.Document(str(OUT))

rs = R["risk_summary"]
cats = rs["category_counts"]
total_test = sum(cats.values())

# ---------------------------------------------------------------- D. Risk Scoring
add_heading2(doc, "D. Cyber Risk Severity Distribution (RQ3)")
add_body(doc,
    f"On the test split of {total_test} events the severity breakdown is: LOW at "
    f"{100*cats.get('LOW',0)/total_test:.1f}% ({cats.get('LOW',0)} events), HIGH at "
    f"{100*cats.get('HIGH',0)/total_test:.1f}% ({cats.get('HIGH',0)} events), and CRITICAL at "
    f"{100*cats.get('CRITICAL',0)/total_test:.1f}% ({cats.get('CRITICAL',0)} events). Notably, "
    f"no test event falls in the MEDIUM band, 30\u201350: mean risk by predicted class is "
    f"{rs['mean_risk_by_predicted_class']['0']:.1f} for Type 0, "
    f"{rs['mean_risk_by_predicted_class']['1']:.1f} for Type 1, and "
    f"{rs['mean_risk_by_predicted_class']['2']:.1f} for Type 2 \u2014 because the classifier is rarely "
    f"ambiguous about which class an event belongs to, the derived risk score clusters at the "
    f"extremes rather than filling the middle of the range. The risk score's weights are not "
    f"arbitrary: each of the five contributing features is weighted by its own SHAP importance, "
    f"renormalized to sum to one, so Device_Response_Time \u2014 the dominant classification signal "
    f"\u2014 also dominates the risk formula at "
    f"{rs['anomaly_feature_weights']['Device_Response_Time']:.3f} of the combined weight. In answer "
    f"to RQ3, the composite score produces a severity breakdown that is empirically distinct from a "
    f"uniform spread across bands, giving operators a data-driven way to prioritize the highest and "
    f"lowest-risk events separately from the model's raw probability, which is always displayed "
    f"alongside it rather than in its place.")

# ---------------------------------------------------------------- E. Counterfactual
add_heading2(doc, "E. Counterfactual Method Comparison (RQ4)")
cf = R["cf_metrics"]
add_body(doc,
    f"Fig. 4 compares the three counterfactual search methods on {cf['n_events_evaluated']} sampled "
    f"events not predicted as the majority class. Single-feature perturbation, Method 1, is the "
    f"sparsest by construction, exactly one feature changed with mean proximity "
    f"{cf['single_feature']['mean_proximity']:.2f}, but has the lowest validity at "
    f"{cf['single_feature']['validity_rate']:.1%}. The greedy multi-feature search, Method 2, "
    f"Algorithm 1, roughly doubles validity to {cf['multi_feature_greedy']['validity_rate']:.1%} at "
    f"the cost of changing three features on average, proximity "
    f"{cf['multi_feature_greedy']['mean_proximity']:.2f}. The global differential-evolution search, "
    f"Method 3, reaches the highest validity by far at "
    f"{cf['optimization_differential_evolution']['validity_rate']:.1%} but changes "
    f"{cf['optimization_differential_evolution']['mean_sparsity']:.2f} of the seven actionable "
    f"features on average, proximity {cf['optimization_differential_evolution']['mean_proximity']:.2f} "
    f"\u2014 a clear validity/sparsity trade-off documented quantitatively rather than asserted. All "
    f"three methods keep essentially all generated counterfactuals within the training "
    f"distribution's plausible range, mean plausibility 100% for all three, and a stability "
    f"check, five reruns of the greedy search on 20 events with shuffled feature-check order, shows "
    f"perfect top-feature agreement at {cf['stability_mean_top_feature_consistency']:.0%}.")

add_figure(doc, "counterfactual_methods.png", 3.0,
           "Fig. 4. Validity rate and sparsity across the three counterfactual methods (N=120).")

add_body(doc,
    "A methodological finding worth reporting directly: a gradient-based optimizer, L-BFGS-B, was "
    "tried first for Method 3 and achieved 0% validity with near-zero movement in every case, "
    "because finite-difference gradients are almost everywhere zero on a tree ensemble's "
    "piecewise-constant decision surface. Switching to differential evolution, which requires no "
    "gradient, immediately recovered the 90% validity rate reported above; this negative "
    "result is included because it generalizes beyond this dataset. In answer to RQ4, counterfactual "
    "simulation reliably identifies interventions that reduce model-estimated threat probability, "
    f"with malware remediation showing the largest mean effect, mean \u0394 probability "
    f"{R['intervention_ranking'].iloc[0]['mean']:.3f}, Fig. 5, among the six "
    "intervention types tested, and the three search strategies trade off validity against sparsity "
    "in a way that should inform which method a deployment chooses.")

add_figure(doc, "intervention_ranking.png", 2.8,
           "Fig. 5. Mean probability change by intervention type (Method 1, N=120).")

# ---------------------------------------------------------------- F. Ablation
add_heading2(doc, "F. Ablation Study")
ab = R["ablation"]
add_body(doc,
    f"Fig. 6 compares Macro F1 across feature-set variants. Adding temporal features to the raw "
    f"feature set changes Macro F1 by only "
    f"{ab.iloc[1].macro_f1 - ab.iloc[0].macro_f1:+.4f}, a negligible effect. {R['ablation_note']}")

add_figure(doc, "ablation.png", 2.3, "Fig. 6. Macro F1 across feature-set variants.")

# ---------------------------------------------------------------- G. Error analysis
add_heading2(doc, "G. Error Analysis")
es = R["error_summary"]
add_body(doc,
    f"Of {es['total_test_events']} test events, {es['total_errors']} at {es['error_rate']:.1%} were "
    f"misclassified, all in the Type 1 \u2194 Type 2 pair identified in Section V.B. "
    f"{es['n_high_confidence_errors_gt_0.8']} of these errors were made with over 80% "
    f"confidence and {es['n_low_confidence_errors_lt_0.6']} with under 60% confidence; the "
    f"latter group is exactly what the confidence-based routing described in Section III.B is "
    f"designed to surface for manual review rather than automatic action.")

doc.save(str(OUT))
print("Section V part 2 (D-G) done.")
