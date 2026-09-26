# -*- coding: utf-8 -*-
import docx
from build_helpers import (R, xgb_row, add_heading1, add_heading2, add_body, OUT)

doc = docx.Document(str(OUT))

add_heading1(doc, "VI. DISCUSSION")

add_heading2(doc, "A. Key Findings and Practical Implications")
add_body(doc,
    "A single latency-related feature dominates this classifier under every importance method "
    "tested: an analyst reviewing why an event was flagged should look at Device_Response_Time "
    "first, and the risk score, whose weights come from the same SHAP values, inherits that "
    "dependence. Events whose risk is driven by something else \u2014 the resistant counterfactual "
    "case in Section V.E, where none of the seven actionable levers moved the model's assessment "
    "\u2014 are exactly the cases current interventions have the least leverage over. Comparing the "
    "three counterfactual methods side by side, rather than reporting only the best-performing one, "
    "surfaces a genuine trade-off: a team wanting a single, easily-explained lever should prefer the "
    "sparser searches, while one wanting the highest chance of actually changing the model's "
    "assessment should accept a broader, harder-to-explain combined intervention.")
add_body(doc,
    "Practically, the classifier's speed and the client-side-verified model export (Section III.D) "
    "suggest tree ensembles of this kind can support real-time triage without a dedicated inference "
    "server, useful for bandwidth-constrained or air-gapped environments. The deliberate separation "
    "between the model's raw probability and the derived risk score (Sections III.B, V.D) matters in "
    "practice: the what-if simulator shows that changing a feature outside the risk formula's five "
    "inputs can shift the predicted class without moving the risk score, and vice versa, so analysts "
    "using either number should understand they answer different questions. The GenAI grounding "
    "results (Section V.H) suggest that restricting a language model to structured, pre-computed "
    "evidence with an explicit \u201cN/A\u201d instruction is a workable pattern for adding "
    "natural-language explanation without the model becoming a second, less auditable source of "
    "truth about the event.")

add_heading2(doc, "B. Limitations and Future Work")
add_body(doc,
    "Class semantics for Attack Type 0, 1, and 2 are undocumented, so every interpretation here is "
    "statistical, not a confirmed real-world taxonomy. The counterfactual engine is a model-based "
    "simulation with no observed real-world intervention data to validate against. The composite "
    "risk score's SHAP-derived weights (\u03b1 = 0.6, five features) were not independently tuned and "
    "should be revisited per deployment, and class weighting rather than SMOTE-style oversampling "
    "[16], [17] was a conservative but untested choice given undocumented feature semantics. "
    "Hyperparameters used standard defaults rather than exhaustive search, so the reported macro F1 "
    "is a solid baseline, not an upper bound. The GenAI grounding evaluation is a manual, "
    "self-assessed check against six constructed scenarios, not an automated benchmark. Finally, this "
    "study does not model an adversary who actively probes the classifier or the counterfactual "
    "engine, a documented risk for tree-based security classifiers [7]; formal robustness evaluation, "
    "an automated grounding benchmark, a live LLM-backed narration API, an expanded actionable "
    "feature set covering Device_Response_Time-dominated events, and investigating the specific "
    "driver of the Type 1/Type 2 confusion are left for future work.")

add_heading1(doc, "VII. CONCLUSION")
add_body(doc,
    "This paper described GenAI-CyberTwin, a fourteen-step pipeline combining cross-validated "
    "ensemble classification, three-method explainability, transparent composite risk scoring, "
    "quantitatively-compared counterfactual security-intervention simulation, and grounded "
    "generative-AI narration, evaluated end-to-end on the KAUST Cyber Attack Detection dataset.")
add_body(doc,
    f"Across five research questions, a gradient-boosted ensemble reaches macro F1 "
    f"{xgb_row.macro_f1:.4f} on a realistically imbalanced three-class problem, confirmed by "
    f"five-fold cross-validation rather than a single split; three independent explanation methods "
    f"agree on which features drive these decisions; a SHAP-weighted composite risk score produces a "
    f"severity breakdown empirically distinct from a uniform distribution; counterfactual simulation "
    f"identifies effective interventions while exposing a measured validity/sparsity trade-off across "
    f"three strategies, including a gradient-based negative result reported rather than hidden; and a "
    f"generative-AI narration layer restricted to structured evidence passes every constructed "
    f"anti-hallucination check applied to it. These findings suggest that combining cross-validated "
    f"accuracy, multi-method explainability, honestly-reported counterfactual simulation, and "
    f"grounded natural-language narration in one open, execution-verified pipeline is a workable, "
    f"reproducible approach to making cyber-attack classification more useful to the analysts who "
    f"act on it \u2014 with the limitations in Section VI.B marking where further validation is "
    f"needed before high-stakes deployment.")

doc.save(str(OUT))
print("Sections VI-VII done.")
