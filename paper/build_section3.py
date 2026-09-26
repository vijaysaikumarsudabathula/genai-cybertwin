# -*- coding: utf-8 -*-
import docx
from build_helpers import (R, add_heading1, add_heading2, add_body, add_caption,
                             add_algorithm, OUT)

doc = docx.Document(str(OUT))

add_heading1(doc, "III. GENAI-CYBERTWIN FRAMEWORK ARCHITECTURE")

add_body(doc,
    "The GenAI-CyberTwin pipeline consists of fourteen steps that turn a raw, pre-aggregated "
    "security event into a classification, an explanation, a risk score, a set of simulated "
    "interventions, and, for a curated subset of events, a grounded natural-language narrative. "
    "Every step below was actually executed against the KAUST dataset; none is a planned but "
    "unimplemented component.")

add_heading2(doc, "A. Steps 1\u20135: Data Ingestion through Model Selection")
add_body(doc,
    "Step 1, Validation: train.csv and test.csv are inspected programmatically (row/column counts, "
    "duplicates, missing values, class distribution, per-feature ranges) rather than assumed, since "
    "no data dictionary was published. Step 2, Data Quality: features that are structurally "
    "non-negative but contain negative raw values \u2014 Login_Attempts, Session_Duration, "
    "Packet_Retry_Rate, Payload_Entropy \u2014 are flagged, and Login_Attempts and "
    "Device_Response_Time additionally show extreme-magnitude outliers; both classes of feature are "
    "winsorized at the 1st/99th training-split percentile only, avoiding leakage. Step 3, Feature "
    "Engineering: eight temporal features (hour, minute, day, day-of-week, weekend flag, cyclical "
    "hour_sin/hour_cos) are derived from the timestamp; no other features were engineered, since the "
    "dataset provides pre-aggregated scores rather than raw logs. Step 4, Splitting: data is split "
    "70/15/15 into train/validation/test, stratified on the target, with preprocessing fit once on "
    "train and applied unchanged elsewhere. Step 5, Model Selection: five candidates \u2014 a "
    "majority baseline, logistic regression, a decision tree, a random forest, and XGBoost \u2014 are "
    "trained and compared on validation by macro F1 (more informative than accuracy given the "
    "~50/25/25 split); the winner is retrained on train+validation and evaluated once on test.")

add_heading2(doc, "B. Steps 6\u20138: Evaluation, Explainability, and Risk Scoring")
add_body(doc,
    "Step 6, Evaluation: the selected model is scored on the test split with accuracy, macro "
    "precision/recall/F1, weighted F1, one-vs-rest ROC-AUC, log loss, Brier score, a confusion "
    "matrix, and a confidence/entropy analysis. Step 7, Explainability: SHAP TreeExplainer values are "
    "computed for every test event and cross-checked against the model's own gain-based importance "
    "and permutation importance on held-out data. Step 8, Risk Scoring: a composite Cyber Risk Score "
    "in [0,100] is Risk = 100\u00d7[\u03b1\u00b7d + (1\u2212\u03b1)\u00b7a], d = 1\u2212P(majority class), "
    "a = a weighted, min-max normalized combination of five features whose weights are each "
    "feature's own SHAP importance, \u03b1 = 0.6; the score is always shown separately from the "
    "model's raw probability, since the two answer different questions (Section V.D).")

add_heading2(doc, "C. Steps 9\u201311: Counterfactual Simulation, Ablation, and Error Analysis")
add_body(doc,
    "Step 9, Counterfactual Simulation: for events not predicted as the majority class, three "
    "methods search for hypothetical changes among seven actionable features a security team could "
    "plausibly control \u2014 Suspicious_Port_Activity, Traffic_Volume_Variation, Login_Attempts, "
    "Malware_Score, User_Behavior_Score, Packet_Retry_Rate, Geo_Dispersion \u2014 that reduce the "
    "re-computed probability of the original class. Method 1 perturbs one feature at a time toward "
    "the low-risk end of its training range; Method 2 is a greedy multi-feature search formalized as "
    "Algorithm 1; Method 3 is a global, gradient-free search (differential evolution), adopted only "
    "after a gradient-based optimizer (L-BFGS-B) failed outright (0% validity) because a tree "
    "ensemble's decision surface is piecewise-constant \u2014 reported in Section V.E rather than "
    "omitted. Step 10, Ablation: the architecture is retrained on raw-only versus raw-plus-temporal "
    "feature sets to isolate what temporal engineering contributes. Step 11, Error Analysis: every "
    "misclassification is characterized by confusion pair and confidence, with SHAP pulled for "
    "representative high-confidence errors.")

add_heading2(doc, "D. Steps 12\u201314: GenAI Narration, Grounding, and Deployment")
add_body(doc,
    "Step 12, Structured Evidence: prediction, probabilities, confidence, top SHAP contributors, "
    "risk score/category, and any counterfactual results for an event are assembled into one JSON "
    "object, with an explicit N/A where no counterfactual exists. Step 13, Grounded Narration: a "
    "generative-AI layer reads only this evidence and produces a seven-part narrative (Threat "
    "Summary, Evidence, Model Interpretation, Counterfactual Findings, Hypothetical Scenario, "
    "Defensive Considerations, Limitations) under an instruction never to invent a value or outcome "
    "absent from the evidence, and to state plainly when a field is missing; six manual grounding "
    "tests targeting missing-evidence, low-confidence, and out-of-scope conditions were run against "
    "it (Section V.H). Step 14, Deployment: the model is exported as a portable JSON tree ensemble "
    "and embedded, with all computed results, in a self-contained dashboard that re-runs the actual "
    "model client-side for hypothetical interventions.")

add_heading2(doc, "E. Counterfactual Search Algorithm")
add_body(doc,
    "Algorithm 1 formalizes Method 2, the greedy multi-feature counterfactual search referenced in "
    "Step 9. At each step it evaluates every remaining actionable feature on a grid between its "
    "current value and the low-risk end of its training range, greedily commits to whichever single "
    "change most reduces the model's re-computed probability for the original predicted class, and "
    "stops early if no candidate feature improves on the current best. Method 1 (single-feature "
    "perturbation) is the special case k_max = 1; Method 3 replaces the greedy inner loop with a "
    "population-based global search over the full actionable subspace simultaneously, at the cost of "
    "sparsity (Section V.E).")

add_algorithm(doc, "Algorithm 1: Greedy Multi-Feature Counterfactual Search", [
    "Input:  trained classifier f, event vector x, actionable",
    "        feature set A, per-feature training bounds B,",
    "        target class c, max features to change k_max, grid",
    "        resolution N",
    "Output: counterfactual vector x\u2032, changed feature set C",
    "",
    "1:  x\u2032 \u2190 x                         \u25b7 copy",
    "2:  C \u2190 \u2205 ;  R \u2190 A                  \u25b7 remaining candidates",
    "3:  p_best \u2190 f_c(x\u2032)",
    "4:  for step = 1 to k_max do",
    "5:      j_best \u2190 NULL ;  v_best \u2190 NULL",
    "6:      for each feature j in R do",
    "7:          grid \u2190 N points from x\u2032_j to B_j.low",
    "8:          for each candidate v in grid do",
    "9:              x_t \u2190 x\u2032 with feature j set to v",
    "10:             p \u2190 f_c(x_t)",
    "11:             if p < p_best then",
    "12:                 p_best \u2190 p ; j_best \u2190 j ; v_best \u2190 v",
    "13:     if j_best = NULL then break        \u25b7 no improvement found",
    "14:     x\u2032_{j_best} \u2190 v_best",
    "15:     C \u2190 C \u222a {j_best} ;  R \u2190 R \\ {j_best}",
    "16: return x\u2032, C",
])
add_caption(doc, "Algorithm 1 was implemented exactly as shown and executed against the real "
                 "trained XGBoost model for all 120 sampled events reported in Section V.E; f_c(\u00b7) "
                 "denotes the model's predicted probability of class c.")

add_body(doc,
    "GenAI-CyberTwin runs its scoring stage (Steps 1\u20138) as a lightweight callable service, while "
    "the counterfactual, ablation, and narration stages (Steps 9\u201313) run offline against sampled "
    "events, since a full counterfactual search is costlier than a single classification call. Step "
    "14's dashboard demonstrates a further option: because the trained tree ensemble exports to a "
    "compact JSON representation whose browser-side tree-walk evaluation was verified to reproduce "
    "the Python model's output to within 1\u00d710\u207b\u2074, an analyst-facing what-if interface can "
    "run entirely client-side with no round-trip to a scoring server.")

doc.save(str(OUT))
print("Section III done.")
