# -*- coding: utf-8 -*-
import docx
from build_helpers import (R, xgb_row, add_heading1, add_heading2, add_body, set_run, OUT)

doc = docx.Document(str(OUT))

add_heading1(doc, "I. INTRODUCTION")

add_body(doc,
    "A classifier that labels a security event \u201cAttack Type 2\u201d does not, by itself, tell an "
    "analyst *why* the event was flagged, *which* signals drove that decision, or *what* a defender "
    "could change to reduce the estimated risk. Consider an analyst reviewing an event assigned to "
    "the highest-risk category with 99% confidence: a bare label supports exactly one action, "
    "escalate or not, and gives no guidance on which of several mitigations \u2014 restricting a port, "
    "forcing re-authentication, running malware remediation \u2014 would most reduce the estimated "
    "risk. Closing this gap is the motivation for GenAI-CyberTwin.")

add_body(doc,
    "This paper argues a practical cyber-attack classifier should have three properties beyond raw "
    "accuracy: explain individual decisions using more than one independent method, since a single "
    "explanation technique can be an artifact of its own assumptions; simulate hypothetical "
    "interventions and report honestly when a method fails rather than showing only successes; and "
    "support natural-language narration without letting it invent evidence, a known failure mode of "
    "language models [11]. Few published systems combine cross-validated accuracy, multi-method "
    "explainability, transparent risk scoring, and grounded narration in one execution-verified "
    "pipeline on a real, if sparsely documented, public dataset \u2014 the KAUST Cyber Attack "
    "Detection dataset used here provides no published data dictionary, a constraint this paper works "
    "honestly within rather than papers over.")

add_heading2(doc, "A. Research Questions")
add_body(doc,
    "RQ1: Can a gradient-boosted ensemble classify cyber-attack events into three realistically "
    "imbalanced categories with high macro F1, confirmed by five-fold cross-validation? "
    "RQ2: Which features most strongly influence decisions, and is that ranking consistent across "
    "three independent importance methods? "
    "RQ3: Does a transparent, SHAP-weighted composite risk score produce a severity breakdown "
    "meaningfully distinct from the model's raw probability? "
    "RQ4: Can counterfactual simulation identify interventions that measurably reduce estimated "
    "threat probability, and how do a single-feature, greedy multi-feature, and global gradient-free "
    "search compare on validity, sparsity, and proximity? "
    "RQ5: Can a generative-AI narration layer restricted to structured evidence avoid fabricating "
    "evidence, confidence, or intervention outcomes?", indent=False)

add_heading2(doc, "B. Contributions")
add_body(doc,
    "This work describes a fourteen-step, execution-verified pipeline spanning data validation, "
    "leakage-free preprocessing, temporal feature engineering, ensemble classification, dual-method "
    "explainability, transparent risk scoring, three-method counterfactual simulation, ablation, "
    "error analysis, and grounded generative-AI narration. It reports a macro F1 of "
    f"{xgb_row.macro_f1:.4f} and accuracy of {xgb_row.accuracy:.4f} on a held-out test split, "
    f"cross-checked against five-fold cross-validation (macro F1 "
    f"{R['cv_summary']['mean']['macro_f1']:.4f} \u00b1 {R['cv_summary']['std']['macro_f1']:.4f}) so "
    "the result is not a single-split artifact. It formalizes a greedy multi-feature counterfactual "
    "search as an explicit algorithm (Section III.E) and compares it quantitatively against "
    "single-feature perturbation and a global differential-evolution search, reporting a "
    "validity/sparsity trade-off rather than only the best-looking method. It defines a composite "
    "cyber-risk score from five SHAP-weighted factors kept numerically and visually separate from the "
    "model's raw probability, and documents a negative result \u2014 a gradient-based optimizer "
    "failing outright against a tree ensemble's piecewise-constant decision surface \u2014 as a "
    "finding rather than an omission. Section II reviews related work; Section III describes the "
    "architecture and search algorithm; Section IV the dataset and protocol; Section V the results; "
    "Section VI discusses implications and limitations; Section VII concludes.")

add_heading2(doc, "C. Scope and Assumptions")
add_body(doc,
    "GenAI-CyberTwin assumes a pre-aggregated, per-event feature vector rather than raw packet "
    "captures or logs, so feature engineering is limited to temporal features derivable from a "
    "timestamp (Section III.A). No data dictionary accompanies the dataset's three attack classes, "
    "so this paper never asserts what \u201cAttack Type 0, 1, or 2\u201d concretely represent; "
    "interpretation is limited to the statistical pattern the models learn. The counterfactual engine "
    "(Section III.C) produces model-based simulations of the classifier's output under a hypothetical "
    "feature change; it does not claim a real-world control would produce the same numeric change, "
    "since the dataset contains no observed intervention outcomes to validate against. The "
    "generative-AI narration layer (Section III.D) only reads structured evidence the pipeline has "
    "already computed and never issues a classification, risk score, or counterfactual result of its "
    "own.")

doc.save(str(OUT))
print("Section I done.")
