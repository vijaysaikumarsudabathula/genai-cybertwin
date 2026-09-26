# GenAI-CyberTwin

A Lightweight Counterfactual Framework for Explainable Cyber-Threat Analysis
and Security Intervention Simulation, built on the KAUST Cyber Attack
Detection dataset (Kaggle: `kaust-ioai-l-1-2026-round-2`).

Every number produced by this project comes from actually running the code
against the real dataset. Where something can't be computed, the pipeline
writes `N/A` rather than a guess.

## What's here

- **ML pipeline** (`src/`): data validation → preprocessing → feature
  engineering → model training/comparison (5 models) → evaluation → SHAP
  explainability → a transparent Cyber Risk Engine → a 3-method
  counterfactual engine → ablation study → error analysis → GenAI evidence
  and narrative generation.
- **Interactive dashboard** (`dashboard/cybertwin_dashboard.html`): a single,
  self-contained HTML file (Plotly for charts, no server required). The
  What-If Simulator re-runs the *actual trained XGBoost model* client-side
  (900 real trees exported to JSON, verified to match Python's
  `predict_proba` to ~1e-4).
- **PDF research report** (`reports/final_research_report.pdf`): executive
  summary through limitations/future work, with every figure and number
  pulled from `results/`.

## Key results (actual, from the held-out test split)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority baseline | 0.503 | 0.223 |
| Logistic Regression | 0.886 | 0.851 |
| Decision Tree | 0.886 | 0.848 |
| Random Forest | 0.896 | 0.860 |
| **XGBoost (selected)** | **0.901** | **0.867** |

- Attack Type 0 is classified perfectly (0 errors); all 159 test-set errors
  are Type 1 <-> Type 2 confusions.
- `Device_Response_Time` dominates feature importance across all three
  independent measures (model-native, SHAP, permutation).
- Counterfactual methods show a genuine validity/sparsity trade-off: single-
  feature perturbation (42.5% validity, 1 feature changed) vs. differential
  evolution (90% validity, ~6.65 features changed).
- Temporal features add essentially nothing (+0.0009 Macro F1) — reported
  honestly rather than inflated.

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py          # runs the full pipeline, regenerates every result file
cd dashboard && python3 build_dashboard.py   # (re)builds the HTML dashboard from results
pytest                          # runs the test suite
```

Then open `dashboard/cybertwin_dashboard.html` directly in a browser — no
server needed.

## Project structure

```
cybertwin/
├── data/
│   ├── raw/            train.csv, test.csv, sample_submission.csv
│   └── reports/        dataset_report.json/md, feature_dictionary.md
├── models/              final_model.joblib, preprocessor.joblib, xgb_model_export.json
├── src/                 all pipeline modules (see run_pipeline.py for order)
├── dashboard/           template.html, build_dashboard.py, cybertwin_dashboard.html
├── results/              every CSV/JSON result the report and dashboard read from
├── reports/              final_research_report.pdf + figures/
├── tests/                pytest suite
├── requirements.txt
└── run_pipeline.py
```

## Honest limitations

- Class semantics for Attack Type 0/1/2 are not documented by the
  competition; all interpretation here is statistical, not confirmed ground
  truth.
- The counterfactual engine is a model-based simulation — there is no
  observed real-world intervention data to validate it against.
- The GenAI layer is Claude narrating pre-computed structured evidence in
  this static deliverable (no external LLM API key configured), not a live
  chat backend. Full narratives exist for 7 curated "Featured Events"; every
  other event gets a shorter, clearly-labeled auto-derived summary in the
  dashboard.
- Hyperparameters use reasonable defaults, not exhaustive tuning.

See `reports/final_research_report.pdf` for the complete write-up.

## Research paper

`paper/` contains an IEEE-format conference paper (`GenAI-CyberTwin.docx` /
`.pdf`, 8 pages, 18 references) reporting these results, plus the Python
scripts used to generate it. See `paper/README.md` for how to rebuild it.
