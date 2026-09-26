# -*- coding: utf-8 -*-
import docx
from build_helpers import (add_heading1, add_heading2, add_body, add_table_title, add_caption,
                             add_ieee_table, OUT)

doc = docx.Document(str(OUT))

add_heading1(doc, "II. RELATED WORK")

add_body(doc,
    "GenAI-CyberTwin builds on prior work in machine-learning-based intrusion detection, "
    "model explainability, and counterfactual explanation. Sommer and Paxson [8] argued that "
    "network intrusion detection differs fundamentally from other machine-learning application "
    "areas because of a scarcity of labeled attack data and a high cost of false positives, a "
    "caution this paper takes seriously by reporting per-class recall and cross-validation rather "
    "than a single aggregate accuracy figure. Buczak and Guven [9] surveyed data-mining and "
    "machine-learning methods for cyber security intrusion detection and found tree-based ensembles "
    "consistently competitive with more complex models, consistent with this paper's finding that "
    "XGBoost, Random Forest, and a tuned decision tree all perform within a few points of macro F1 "
    "of one another (Section V.A).")

add_body(doc,
    "On explainability, Lundberg and Lee [3] introduced SHAP as a unified, game-theoretic framework "
    "for feature attribution, later extended to tree ensembles by Lundberg et al. [4]; this paper "
    "uses the tree-specific formulation and cross-checks it against permutation importance on "
    "held-out data, since exact SHAP values and a model's own gain-based importance can disagree for "
    "correlated features. Ribeiro et al. [13] introduced LIME, a complementary local surrogate-model "
    "approach; this paper's choice of SHAP plus permutation importance over LIME reflects a "
    "preference for global, consistency-guaranteed attribution over a per-instance local surrogate. "
    "Alenezi and Ludwig [10] applied TreeSHAP, KernelSHAP, and DeepSHAP to two five-class "
    "cybersecurity datasets using Random Forest, XGBoost, and a sequential network, reporting "
    "XGBoost macro F1 of 0.98 and 0.93 respectively; this is among the few prior studies pairing SHAP "
    "with cybersecurity classification, though it reports neither risk scoring, counterfactual "
    "simulation, nor cross-validation.")

add_body(doc,
    "On benchmark data, most machine-learning intrusion-detection work evaluates against a small "
    "number of well-studied corpora: Tavallaee et al. [15] introduced NSL-KDD to correct known "
    "redundancy and bias problems in the older KDD Cup 99 data, and Moustafa and Slay [14] introduced "
    "UNSW-NB15 to reflect more modern attack behavior with a hybrid of real and synthesized traffic. "
    "The KAUST dataset used in this paper is comparatively new and, notably, ships without the kind "
    "of data dictionary that accompanies NSL-KDD and UNSW-NB15, which is why Section IV.A reports "
    "data-quality findings from direct inspection rather than from documentation.")

add_body(doc,
    "On class imbalance, a common alternative to the class-weighted training used in this paper is "
    "synthetic oversampling: Chawla et al.'s SMOTE [16] generates synthetic minority-class examples "
    "by interpolating between neighbors, and He and Garcia's survey [17] catalogs SMOTE variants and "
    "alternative resampling and cost-sensitive strategies. This paper uses class weighting rather "
    "than oversampling specifically to avoid introducing synthetic feature combinations into a "
    "security dataset whose feature semantics are already undocumented, a conservative choice "
    "revisited in Section VI.C.")

add_body(doc,
    "On counterfactual explanation, Wachter et al. [1] proposed unconditional counterfactual "
    "explanations as a way to satisfy a right to explanation without opening a model's internals, "
    "framing a counterfactual as the smallest change to an input that would have produced a different "
    "outcome; Mothilal et al. [2] extended this to a practical multi-class algorithm (DiCE) that "
    "generates diverse counterfactual sets and evaluates them on validity, proximity, and sparsity, "
    "the same three axes this paper reports for its own counterfactual engine (Section V.E). Neither "
    "of these foundational counterfactual papers targets cybersecurity specifically or compares a "
    "gradient-free global search against gradient-based alternatives, which this paper does directly "
    "(Section III.E) after observing that a gradient-based optimizer fails against a tree ensemble's "
    "non-smooth decision surface. On classifier architecture, this paper's ensemble draws on Chen "
    "and Guestrin's XGBoost [5] and Breiman's Random Forest [6]; the global counterfactual search "
    "uses Storn and Price's differential evolution [7], chosen specifically because it does not "
    "require a differentiable objective.")

add_body(doc,
    "On risk scoring, the Common Vulnerability Scoring System [18] is the cybersecurity industry's "
    "standard composite severity metric, combining multiple sub-scores into a single 0\u201310 severity "
    "value the way this paper's Cyber Risk Score combines five SHAP-weighted factors into a 0\u2013100 "
    "value; unlike CVSS, whose sub-score weights are fixed by expert committee, this paper's weights "
    "are derived directly from the trained model's own SHAP importances, so the risk formula updates "
    "automatically if the underlying model or data changes. On grounded natural-language generation, "
    "Ji et al. [11] surveyed hallucination in natural-language generation and catalogued the "
    "conditions under which generative models fabricate unsupported content; the six manual grounding "
    "tests in Section V.H are constructed around exactly the failure conditions that survey identifies "
    "as highest-risk (missing evidence, low-confidence predictions, and requests that exceed the "
    "evidence provided). The NIST AI Risk Management Framework [12] motivates this paper's emphasis "
    "on auditable, reproducible evaluation throughout.")

add_body(doc,
    "Table I compares GenAI-CyberTwin with the most directly related prior systems across six "
    "dimensions. A recurring pattern is that individual capabilities \u2014 strong predictive accuracy, "
    "SHAP-based explainability, or counterfactual simulation \u2014 each appear in isolation across "
    "different systems, but cross-validated accuracy, multi-method explainability, transparent risk "
    "scoring, quantitatively-compared counterfactual search, and grounded generative-AI narration "
    "rarely appear together in one evaluated pipeline.")

add_table_title(doc, "TABLE I. COMPARISON WITH RELATED PRIOR WORK")
add_ieee_table(
    doc,
    headers=["System", "Task /\nModel", "F1", "XAI", "Risk\nScore", "Counter-\nfactual", "Cross-\nVal.", "GenAI"],
    rows=[
        ["This work", "3-class,\nXGBoost", "0.867", "Yes\n(SHAP+\nperm.)", "Yes", "Yes\n(3\nmethods)", "Yes\n(5-fold)", "Yes"],
        ["Alenezi &\nLudwig [10]", "5-class,\nXGB/RF", "0.93\u2013\n0.98", "Yes\n(3x\nSHAP)", "No", "No", "No", "No"],
        ["Sommer &\nPaxson [8]", "Critique\n(no model)", "N/A", "No", "No", "No", "N/A", "No"],
        ["Buczak &\nGuven [9]", "Survey", "N/A", "N/A", "No", "No", "N/A", "No"],
        ["Wachter [1] /\nMothilal [2]", "CF method\n(generic)", "N/A", "Yes", "No", "Yes\n(orig.)", "N/A", "No"],
    ],
    col_widths_in=[0.52, 0.45, 0.3, 0.38, 0.32, 0.38, 0.32, 0.32],
    font_pt=6.8,
)
add_caption(doc, "TABLE I. F1 values are macro F1 as reported by the cited work where applicable; "
                 "N/A indicates the work is not a classifier or does not report the metric.")

doc.save(str(OUT))
print("Section II done.")
