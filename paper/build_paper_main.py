# -*- coding: utf-8 -*-
"""
build_paper_main.py
Uses build_helpers.py to replace ExARG's title/authors/body with the full
GenAI-CyberTwin paper, inside the SAME docx template (fonts, 2-column
section, table borders all inherited from ExARG.docx).
"""
import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

from build_helpers import (R, xgb_row, rf_row, add_heading1, add_heading2, add_body,
                             add_caption, add_table_title, add_ieee_table, add_figure,
                             add_algorithm, add_reference, set_run, add_runs, SRC, OUT)

doc = docx.Document(str(SRC))

# ---------------------------------------------------------------- 1. TITLE
title_p = doc.paragraphs[0]
for r in list(title_p.runs):
    r.text = ""
title_p.runs[0].text = "GenAI-CyberTwin: A Lightweight Counterfactual Framework for Explainable Cyber-Threat Analysis and Security Intervention Simulation"
set_run(title_p.runs[0], bold=True, size=Pt(15))

# ---------------------------------------------------------------- 2. AUTHORS
author_table = doc.tables[0]
merged = author_table.cell(0, 0).merge(author_table.cell(1, 1))
for p in list(merged.paragraphs):
    p._p.getparent().remove(p._p)
p1 = merged.add_paragraph()
p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
r1 = p1.add_run("[Author Name(s) \u2014 to be completed]")
set_run(r1, bold=False, size=Pt(10))
p2 = merged.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("Department / Affiliation, City, Country")
set_run(r2, italic=True, size=Pt(9))
p3 = merged.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run("email@example.com")
set_run(r3, size=Pt(9))

# ---------------------------------------------------------------- 3. ABSTRACT / KEYWORDS
for idx in (3, 4, 5):
    p = doc.paragraphs[idx]
    for r in list(p.runs):
        r.text = ""

abs_p = doc.paragraphs[3]
r = abs_p.add_run("Abstract\u2014")
set_run(r, bold=True, size=Pt(10))
abstract_text = (
    "Security operations teams increasingly rely on machine-learned classifiers to triage cyber-attack "
    "telemetry, but a bare class label does not explain why an event was flagged or what intervention "
    "would reduce its risk fastest, limiting analyst trust. Existing intrusion-detection systems "
    "rarely combine cross-validated accuracy, multi-method explainability, transparent risk scoring, "
    "counterfactual analysis, and grounded narration within one pipeline. This paper presents "
    "GenAI-CyberTwin, a fourteen-step pipeline integrating gradient-boosted ensemble classification, "
    "dual-method explainability, a transparent composite cyber-risk score, a three-method "
    "counterfactual security-intervention engine, and a grounded generative-AI narration layer. The "
    "framework is evaluated on the KAUST Cyber Attack Detection dataset with a stratified 70/15/15 "
    "split and five-fold cross-validation; SHAP and permutation importance cross-check feature "
    "attributions, and three counterfactual search strategies, single-feature perturbation, greedy "
    "multi-feature search, and differential-evolution global search, are compared on validity, "
    f"sparsity, and proximity. The selected XGBoost model achieves macro F1 = {xgb_row.macro_f1:.4f} "
    f"and accuracy = {xgb_row.accuracy:.4f} on a held-out test split, confirmed by cross-validation at "
    f"{R['cv_summary']['mean']['macro_f1']:.4f} \u00b1 {R['cv_summary']['std']['macro_f1']:.4f}. Three "
    "independent importance methods agree that one latency-related feature dominates the model by a "
    "factor of 4.6. The global search reaches 90.0% validity versus 42.5% for single-feature "
    "perturbation, at the cost of sparsity, while a gradient-based optimizer tried first failed "
    "outright at 0% validity against the classifier's piecewise-constant decision surface. The "
    "narration layer, grounded strictly in structured evidence, passed all six constructed "
    "anti-hallucination checks. These results show that predictive performance, explanation "
    "consistency, and intervention validity can be jointly measured without overstating any single "
    "component, and that search strategy should be chosen based on whether sparsity or validity "
    "matters more. This work contributes a reproducible, cross-validated pipeline, a formalized "
    "counterfactual search algorithm, a transparent SHAP-weighted risk score, and an "
    "evidence-grounded narration layer."
)
add_runs(abs_p, abstract_text)
abs_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
abs_p.paragraph_format.space_after = Pt(6)

cont_p = doc.paragraphs[4]
cont_p.text = ""

kw_p = doc.paragraphs[5]
rkw = kw_p.add_run("Keywords\u2014")
set_run(rkw, bold=True, italic=True, size=Pt(10))
rkw2 = kw_p.add_run(
    "Explainable AI, Cybersecurity, Intrusion Detection, SHAP, Counterfactual Explanation, "
    "Composite Risk Scoring, Generative AI, Ensemble Classification, Cross-Validation, Differential Evolution"
)
set_run(rkw2, italic=True, size=Pt(10))
kw_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

doc.save(str(OUT))
print("Phase 1 (title/authors/abstract/keywords) done ->", OUT)
