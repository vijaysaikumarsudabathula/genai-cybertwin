# -*- coding: utf-8 -*-
import docx
from build_helpers import (R, add_heading1, add_heading2, add_body, add_figure, OUT)

doc = docx.Document(str(OUT))

add_heading1(doc, "IV. EXPERIMENTAL SETUP AND DATASET")

add_heading2(doc, "A. Dataset Description")
dr = R["dataset_report"]
add_body(doc,
    f"Experiments use the KAUST Cyber Attack Detection dataset, Kaggle competition "
    f"kaust-ioai-l-1-2026-round-2 (see Dataset Availability, Section VIII, for access details), "
    f"which contains {dr['train_rows']:,} labeled training events and "
    f"{dr['test_rows']:,} unlabeled Kaggle test events across three attack classes with a realistic "
    f"imbalance: class 0 at {dr['class_distribution_pct']['0']:.1f}% "
    f"({dr['class_distribution_counts']['0']:,} records), class 1 at "
    f"{dr['class_distribution_pct']['1']:.1f}% ({dr['class_distribution_counts']['1']:,} "
    f"records), and class 2 at {dr['class_distribution_pct']['2']:.1f}% "
    f"({dr['class_distribution_counts']['2']:,} records). A stratified 70/15/15 "
    f"split, Section III.A, produced training, validation, and held-out test sets; timestamps span "
    f"{dr['timestamp_min']} to {dr['timestamp_max']}.")

add_body(doc,
    "No data dictionary accompanies this competition. On direct inspection, train.csv contains zero "
    "missing values and zero duplicate rows. Four features \u2014 Login_Attempts, Session_Duration, "
    "Packet_Retry_Rate, and Payload_Entropy \u2014 contain negative values that are structurally "
    "impossible for a count, duration, rate, or entropy quantity and are winsorized; three further "
    "features (Malware_Score, User_Behavior_Score, Anomaly_Tendency) also contain negative values "
    "that, absent a data dictionary, cannot be confidently distinguished from a meaningful signed "
    "indicator and are left unaltered. Login_Attempts and Device_Response_Time additionally range "
    "across several orders of magnitude more than a plausible real count or latency value would "
    "suggest, consistent with the competition's own description noting extreme values in these "
    "columns. Feature selection was deliberately simple: rather than engineering complex derived "
    "features that might not generalize, the model uses the thirteen provided "
    "behavioral/network/threat/session features directly, plus seven temporal features (Section "
    "III.A), since the dataset provides pre-aggregated scores rather than raw logs to derive further "
    "signal from.")

add_heading2(doc, "B. Implementation Environment")
add_body(doc,
    "All experiments were run using Python 3.12 with scikit-learn 1.8.0, XGBoost 3.4.1, "
    "SHAP 0.52.0, NumPy 2.4.4, and Pandas 3.0.2, without a GPU. Training all five candidate models "
    f"on the full training split completes in under {R['model_comparison']['training_time_sec'].sum():.1f} "
    "seconds combined on a standard multi-core CPU; the counterfactual engine's 120-event evaluation "
    "across three methods, including a population-based global search, completes in under a minute.")

add_heading2(doc, "C. Evaluation Protocol")
add_body(doc,
    "Model selection used the validation split with macro F1 as the primary metric, since macro F1 "
    "weights the minority classes equally with the majority class rather than being dominated by it. "
    "The winning architecture was retrained on train+validation and given a single, final evaluation "
    "on the held-out test split, which was not used for model selection, to avoid optimistic bias. "
    "To confirm this single-split estimate was not favorable by chance, the same architecture was "
    "additionally evaluated with five-fold stratified cross-validation on the full labeled dataset, "
    "refitting preprocessing inside every fold independently to prevent leakage (Section V.A). "
    "Hyperparameters (300 estimators, maximum depth 6, learning rate 0.1) were set to standard "
    "defaults for gradient-boosted trees rather than extensively tuned, a scope limitation discussed "
    "in Section VI.C.")

doc.save(str(OUT))
print("Section IV done.")
