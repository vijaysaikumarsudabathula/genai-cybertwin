# IEEE Paper — Build Scripts

This folder contains the scripts used to generate `GenAI-CyberTwin.docx`, the
IEEE-format research paper reporting the results in this repository.

- `ExARG-template.docx` — the original IEEE-format template the paper's
  layout (fonts, two-column section, table borders) was built from.
- `build_helpers.py` — shared formatting helpers (headings, tables, figures,
  the Algorithm 1 pseudocode box, references) that replicate the template's
  exact IEEE styling via python-docx.
- `build_paper_main.py` → `clear_body.py` → `build_section1.py` ... →
  `build_section5c.py` → `build_section67.py` → `build_dataset_availability.py`
  → `build_references.py` — run in this order to regenerate the paper from
  scratch. Each script reads results from `../results/`, `../models/`, and
  `../data/reports/` and appends its section to the document.
- `GenAI-CyberTwin.docx` / `GenAI-CyberTwin.pdf` — the final paper (8 pages,
  IEEE two-column format).

## Rebuilding

```bash
cd paper
python3 build_paper_main.py
python3 clear_body.py
for f in build_section1 build_section2 build_section3 build_section4 \
         build_section5a build_section5b build_section5c build_section67 \
         build_dataset_availability build_references; do
    python3 $f.py
done
```

The output is written to `GenAI-CyberTwin.docx` (same filename each run).
