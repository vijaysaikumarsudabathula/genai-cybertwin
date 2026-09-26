# -*- coding: utf-8 -*-
import docx
from build_helpers import add_heading1, add_body, OUT

doc = docx.Document(str(OUT))

add_heading1(doc, "VIII. DATASET AVAILABILITY")

add_body(doc,
    "Dataset: KAUST Cyber Attack Detection dataset. Source: Kaggle competition "
    "\u201cKAUST-IOAI-L1-2026-Round2.\u201d Link: "
    "https://www.kaggle.com/competitions/kaust-ioai-l-1-2026-round-2/data. Access: publicly available "
    "subject to Kaggle account registration and acceptance of the competition rules; no additional "
    "license was required. This paper uses train.csv (10,684 labeled events), split as described in "
    "Section III.A; test.csv (2,672 unlabeled events) is part of the release but not used here.")

doc.save(str(OUT))
print("Dataset Availability section done.")
