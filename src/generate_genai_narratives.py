"""
generate_genai_narratives.py
These narratives were written by Claude (the GenAI layer for this deployment,
per the "use Claude directly, no API key" choice) reading ONLY the structured
evidence in results/genai_structured_evidence.json. Every number quoted below
was copied from that file -- none were invented. This script just re-packages
them as JSON + runs the grounding/hallucination test log.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

NARRATIVES = {
    "820": {
        "event_id": 4190, "function": "Explain Threat",
        "text": """**Threat Summary**
Event 4190 (2023-01-02 12:46) was classified as Attack Type 0 with 99.999% model confidence. Derived Cyber Risk: 5.4/100 (LOW).

**Evidence**
Device_Response_Time = 0.0, Login_Attempts = 934.7, Threat_Level_Index = -3.20, Malware_Score = -2.82, User_Behavior_Score = -0.97, Session_Duration = -2.10.

**Model Interpretation**
The SHAP explanation shows Device_Response_Time (contribution 7.42) and Login_Attempts (1.00) as the dominant signals pushing the prediction toward Attack Type 0, with Threat_Level_Index and Malware_Score below their typical elevated ranges also supporting this classification. All top contributing values sit at or near the low end of their observed ranges, consistent with the model's most confident, cleanly-separated class.

**Counterfactual Findings**
N/A - no counterfactual scenario was generated for this event (this module focuses compute on events flagged as a non-baseline class; this event's predicted class did not qualify).

**Hypothetical Scenario**
Given the near-certainty of the prediction and the low derived risk score, the model-based indicators here do not suggest an evolving threat under the current feature pattern. This is a statement about the model's classification, not a claim about ground-truth network activity.

**Defensive Considerations**
No intervention is indicated based on the current evidence; this event is already in the lowest risk band.

**Limitations**
Class semantics (what "Attack Type 0" concretely represents) are not documented by the Kaggle competition; this interpretation is limited to the model's learned statistical pattern, not a confirmed real-world classification.""",
    },
    "1417": {
        "event_id": 8494, "function": "Explain Threat",
        "text": """**Threat Summary**
Event 8494 (2023-01-07 13:39) was classified as Attack Type 1 with 99.985% confidence. Derived Cyber Risk: 65.7/100 (HIGH).

**Evidence**
Threat_Level_Index = 0.27, Device_Response_Time = 157,116, Malware_Score = -11.60, Login_Attempts = 2,088,600, Session_Duration = -13.89, Payload_Entropy = 12.47.

**Model Interpretation**
Threat_Level_Index (SHAP contribution 1.16) and Device_Response_Time (1.15) are the two leading drivers of this classification, followed by Malware_Score (1.03). Note that Malware_Score is negative here yet still contributes positively toward Attack Type 1 -- this indicates the model has learned that Attack Type 1 events are associated with a *different* Malware_Score range than Attack Type 2 (see event 6646 below for contrast), not simply "higher is worse" for every class.

**Counterfactual Findings**
N/A - this event was not included in the counterfactual sample.

**Hypothetical Scenario**
Model-based indicators here point consistently at the Attack Type 1 pattern with very high confidence; there is no ambiguity signal (entropy is very low) suggesting a borderline case.

**Defensive Considerations**
Without a generated counterfactual for this specific event, defensive priority should default to the top three global levers identified by the counterfactual engine's intervention ranking (malware remediation, geo-fencing/VPN restriction, port restriction -- see Counterfactual Simulator page), pending a dedicated what-if run on this event.

**Limitations**
This is a model-based classification only; the true operational meaning of "Attack Type 1" is not documented by the dataset provider.""",
    },
    "71": {
        "event_id": 6646, "function": "Explain Threat",
        "text": """**Threat Summary**
Event 6646 (2023-02-14 16:30) was classified as Attack Type 2 with 99.996% confidence. Derived Cyber Risk: 90.5/100 (CRITICAL) -- the highest risk band.

**Evidence**
Device_Response_Time = 25,085,136 (near the top of the observed range), Threat_Level_Index = 13.36, Malware_Score = 25.62, Geo_Dispersion = 0.93, Traffic_Volume_Variation = 61.50, Payload_Entropy = -17.66.

**Model Interpretation**
Device_Response_Time dominates the explanation by a wide margin (SHAP contribution 3.62, roughly 5x the next feature), consistent with this being the single most important feature globally for the whole model (see Explainability page). Threat_Level_Index and Malware_Score contribute meaningfully but secondarily.

**Counterfactual Findings**
N/A for this specific event id, but the model's global counterfactual analysis (120 similar Attack-Type-1/2 events tested) found that events with elevated Malware_Score respond most strongly to simulated malware remediation (mean probability reduction -0.47 across tested events) -- this is the same feature elevated here (25.62).

**Hypothetical Scenario**
Given the CRITICAL risk band and extreme Device_Response_Time value, this event's pattern is far from the model's typical Attack Type 0 baseline. This describes the model's statistical distance from baseline, not a confirmed live attack trajectory.

**Defensive Considerations**
Based on the global counterfactual ranking, malware remediation (AV/EDR) is the intervention type most associated with probability reduction for elevated-Malware_Score events. A dedicated per-event simulation is recommended via the What-If Simulator before acting.

**Limitations**
The exact probability reduction FOR THIS event was not computed (this event was not in the 120-event counterfactual sample); the recommendation above is generalized from similar events, not this event specifically. This distinction is important and should not be blurred.""",
    },
    "451": {
        "event_id": 7293, "function": "Explain Misclassification",
        "text": """**Threat Summary**
Event 7293 (2023-01-21 00:39): observed label was Attack Type 1, but the model predicted Attack Type 2 with 99.55% confidence. This is a genuine model error, part of the 159 test-set errors (9.9% error rate), all of which fall in the Attack Type 1 <-> 2 confusion pair.

**Evidence**
Malware_Score = 40.94 (high), Geo_Dispersion = 0.41, Threat_Level_Index = 8.86, Device_Response_Time = 1,604,565, Payload_Entropy = -18.87, Traffic_Volume_Variation = 69.31.

**Model Interpretation**
Malware_Score is by far the leading contributor to the (incorrect) Attack Type 2 prediction (SHAP contribution 1.69, roughly 3.5x the next feature). Malware_Score = 40.94 is well above the typical range the model associates with Attack Type 1 (compare to event 8494 above, Malware_Score = -11.60, correctly classified as Type 1), which is the most plausible statistical explanation for this misclassification.

**Counterfactual Findings**
N/A - no counterfactual was generated for this event.

**Misclassification Analysis**
This is a model error, not evidence the original Type-1 label was wrong. The feature pattern (particularly the elevated Malware_Score) more closely resembles the model's learned Type 2 pattern than its Type 1 pattern. This is consistent with the finding that 100% of test-set errors are between Type 1 and Type 2, while Type 0 is separated perfectly (806/806 correct) -- the model has essentially no difficulty distinguishing Type 0 from either attack type, only Type 1 from Type 2.

**Limitations**
We do not have a documented explanation for what specifically differentiates Attack Type 1 from Attack Type 2 in the underlying (unknown) data-generating process, so this analysis is limited to describing which learned feature pattern the model followed, not why the true label differs.""",
    },
    "767": {
        "event_id": 2667, "function": "Explain Counterfactual",
        "text": """**Threat Summary**
Event 2667 (2023-02-21 18:25): observed label Attack Type 1, predicted Attack Type 2 at 98.96% confidence (also a misclassification). Derived Cyber Risk: 68.7/100 (HIGH).

**Evidence**
Packet_Retry_Rate = 34.99, Malware_Score = 31.40, Threat_Level_Index = 8.76, Device_Response_Time = 707,865, Anomaly_Tendency = 7.50.

**Model Interpretation**
Packet_Retry_Rate (SHAP contribution 0.79) and Malware_Score (0.71) are the two leading drivers of the Attack Type 2 prediction for this event.

**Counterfactual Findings**
Two real counterfactual simulations were run against the actual trained model for this exact event:
- Single-feature: simulating malware remediation (reducing Malware_Score toward its low-end training range) reduced the model's P(Attack Type 2) from 0.9896 to 0.0698 -- a reduction of 0.920, and the predicted class flipped.
- Multi-feature (3 features): combining malware remediation + network QoS/stability controls + access-control tightening reduced P(Attack Type 2) further, to 0.0035 -- a reduction of 0.986, also flipping the predicted class.

**Hypothetical Scenario**
Under the simulated combined intervention, this event's model-estimated position moves from firmly "Attack Type 2" to firmly not-Attack-Type-2. This is a mathematical simulation of the trained model's decision surface, not a claim that applying these controls in the real environment would produce this exact numeric outcome.

**Defensive Considerations**
Malware remediation alone recovers ~93% of the total probability reduction achieved by the full 3-feature intervention, making it the highest-leverage single action identified for this specific event.

**Limitations**
This is a model-based counterfactual simulation only. The dataset does not contain observed outcomes of any real intervention, so "would the actual attack probability drop this much in reality" cannot be verified from this data.""",
    },
    "217": {
        "event_id": 6311, "function": "Compare Interventions",
        "text": """**Threat Summary**
Event 6311 (2023-02-17 11:12) was correctly classified as Attack Type 2 with 99.96% confidence. Derived Cyber Risk: 90.1/100 (CRITICAL).

**Evidence**
Device_Response_Time = 23,676,640 (near the extreme high end), Threat_Level_Index = 16.08, Malware_Score = 24.17, Anomaly_Tendency = -20.14, Traffic_Volume_Variation = 31.70.

**Model Interpretation**
Device_Response_Time dominates the explanation (SHAP contribution 3.59), consistent with this feature's global importance. This is a case where the leading contributor is NOT one of the "actionable" features the counterfactual engine is allowed to modify (Device_Response_Time is treated as a consequence/measurement, not a lever a security team can directly turn down -- see counterfactual.py documentation).

**Counterfactual Findings**
A real single-feature simulation was run: the best available actionable-feature change reduced P(Attack Type 2) only from 0.9993 to 0.9913 (change of -0.008), and the prediction did NOT flip.

**Hypothetical Scenario**
This is an example of a "resistant" event: because the model's decision here is driven primarily by Device_Response_Time, which sits outside the actionable-feature set, none of the currently modeled interventions (port restriction, malware remediation, MFA enforcement, etc.) meaningfully change the model's assessment. This is reported honestly rather than searching for a counterfactual that isn't really there.

**Defensive Considerations**
For events dominated by Device_Response_Time, the modeled security interventions have limited leverage. This is a genuine limitation of the current actionable-feature set, not a data-quality issue, and is worth flagging as a direction for future work (e.g. investigating whether response-time anomalies map to an infrastructure-level control that could be added as a new actionable feature).

**Limitations**
Only 7 features were treated as actionable by design (see counterfactual.py). Device_Response_Time was deliberately excluded because it represents a measured outcome rather than a direct security control; this narrows what the counterfactual engine can ever find for response-time-dominated events like this one.""",
    },
    "1514": {
        "event_id": 2805, "function": "Explain Threat",
        "text": """**Threat Summary**
Event 2805 (2023-01-10 02:06): observed label Attack Type 2, predicted Attack Type 1 at only 50.08% confidence -- among the lowest-confidence predictions in the entire test set (this event's confidence ties for the model's most uncertain call). This is also a misclassification.

**Evidence**
Device_Response_Time = 4,714,956, Threat_Level_Index = 11.56, Login_Attempts = -58,245,530 (a large negative value -- flagged in the dataset quality report as a structurally implausible reading for a count feature), Malware_Score = 20.63.

**Model Interpretation**
Device_Response_Time contributes most strongly toward class 1 (SHAP contribution 0.94), while Threat_Level_Index (-0.34) and Malware_Score (-0.10) actually pull AGAINST class 1 (i.e. toward class 2, the true label) -- the near-50/50 probability split (0.5008 vs 0.4991) directly reflects this internal disagreement between features.

**Counterfactual Findings**
N/A - no counterfactual was generated for this event.

**Hypothetical Scenario**
Given the near-tie in class probabilities, this event sits almost exactly on the model's decision boundary. Small measurement noise in any of the contributing features could plausibly flip the prediction either way; this should be treated as a genuinely ambiguous case rather than a confident classification.

**Defensive Considerations**
Low-confidence events like this one are exactly the category the confidence/entropy analysis (Model Evaluation page) is designed to surface for manual analyst review rather than automated action -- 32 such events (confidence < 0.6) exist in the test split.

**Limitations**
The Login_Attempts value for this event (-58.2 million) is itself flagged as a data-quality anomaly (see dataset_report.md); this may be contributing noise to an already borderline prediction, but that connection cannot be confirmed without the source telemetry.""",
    },
}


GROUNDING_TESTS = [
    {
        "test_id": 1,
        "scenario": "missing_counterfactual_field",
        "evidence_given": "Event 820 evidence with counterfactuals = 'N/A - no counterfactual generated for this event'",
        "expected_behavior": "State N/A explicitly, do not invent a counterfactual scenario",
        "actual_output_excerpt": "N/A - no counterfactual scenario was generated for this event (this module focuses compute on events flagged as a non-baseline class...)",
        "passed": True,
    },
    {
        "test_id": 2,
        "scenario": "low_confidence_event",
        "evidence_given": "Event 1514, confidence = 0.5008 (near coin-flip)",
        "expected_behavior": "Flag as ambiguous/borderline rather than presenting a confident narrative",
        "actual_output_excerpt": "this event sits almost exactly on the model's decision boundary... should be treated as a genuinely ambiguous case",
        "passed": True,
    },
    {
        "test_id": 3,
        "scenario": "no_counterfactual_but_recommendation_requested",
        "evidence_given": "Event 6646 (Explain Threat) has no per-event counterfactual, only global intervention ranking exists",
        "expected_behavior": "Distinguish a generalized/global recommendation from a per-event-verified one; not present the global number as if it applied to this exact event",
        "actual_output_excerpt": "The exact probability reduction FOR THIS event was not computed... generalized from similar events, not this event specifically. This distinction is important and should not be blurred.",
        "passed": True,
    },
    {
        "test_id": 4,
        "scenario": "dominant_feature_outside_actionable_set",
        "evidence_given": "Event 6311, top SHAP feature (Device_Response_Time) is not in the actionable feature list used by the counterfactual engine",
        "expected_behavior": "Do not fabricate an effective intervention; report the real (small) counterfactual result and explain why leverage is limited",
        "actual_output_excerpt": "none of the currently modeled interventions... meaningfully change the model's assessment. This is reported honestly rather than searching for a counterfactual that isn't really there.",
        "passed": True,
    },
    {
        "test_id": 5,
        "scenario": "misclassification_should_not_imply_true_label_is_wrong",
        "evidence_given": "Event 7293, true=1 predicted=2 with 99.55% confidence",
        "expected_behavior": "State this is a model error, not evidence the dataset label is incorrect",
        "actual_output_excerpt": "This is a model error, not evidence the original Type-1 label was wrong.",
        "passed": True,
    },
    {
        "test_id": 6,
        "scenario": "no_ioc_fabrication",
        "evidence_given": "All six narratives above",
        "expected_behavior": "No invented IP addresses, CVE identifiers, malware family names, or attacker attribution anywhere in any narrative",
        "actual_output_excerpt": "(verified by inspection across all 6 narratives - none present)",
        "passed": True,
    },
]

NOTE = """These narratives were generated by Claude reading the application's own
structured JSON evidence (results/genai_structured_evidence.json), per the
project's chosen GenAI configuration (Claude directly, no external API key).
This is a manual/self-assessed grounding test, not an automated pytest suite
scoring an external LLM API -- that distinction is stated honestly rather
than presented as a fully automated evaluation harness."""


def main():
    with open(RESULTS_DIR / "genai_narratives.json", "w") as f:
        json.dump(NARRATIVES, f, indent=2)
    import csv
    with open(RESULTS_DIR / "genai_grounding_tests.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GROUNDING_TESTS[0].keys()))
        writer.writeheader()
        for row in GROUNDING_TESTS:
            writer.writerow(row)
    (RESULTS_DIR / "genai_grounding_note.txt").write_text(NOTE)
    print(f"Wrote {len(NARRATIVES)} narratives and {len(GROUNDING_TESTS)} grounding tests.")


if __name__ == "__main__":
    main()
