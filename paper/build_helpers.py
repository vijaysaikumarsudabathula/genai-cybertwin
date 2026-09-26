# -*- coding: utf-8 -*-
"""
build_paper.py
Rebuilds ExARG.docx in place into the GenAI-CyberTwin paper: same IEEE
two-column template (fonts, margins, section/column layout, table borders)
inherited from the original file, entirely new content. Every number below
was copied from /home/claude/cybertwin/results/*.json|csv -- see
CYBERTWIN_RESULTS dict for the loader.
"""
import json
import re
from pathlib import Path

import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = Path("/home/claude/paper/ExARG.docx")
OUT = Path("/home/claude/paper/GenAI-CyberTwin.docx")
CT_ROOT = Path("/home/claude/cybertwin")
FIG_DIR = CT_ROOT / "reports" / "figures"

# ---------------------------------------------------------------- load data
def load_json(p):
    return json.load(open(CT_ROOT / p))

R = {
    "dataset_report": load_json("data/reports/dataset_report.json"),
    "model_selection": load_json("results/model_selection.json"),
    "risk_summary": load_json("results/risk_engine_summary.json"),
    "cf_metrics": load_json("results/counterfactual_metrics.json"),
    "error_summary": load_json("results/error_analysis_summary.json"),
    "cv_summary": load_json("results/cross_validation_summary.json"),
    "ablation_note": (CT_ROOT / "results" / "ablation_note.txt").read_text(),
}
import pandas as pd
R["model_comparison"] = pd.read_csv(CT_ROOT / "results" / "model_comparison.csv")
R["shap_importance"] = pd.read_csv(CT_ROOT / "results" / "shap_importance.csv")
R["ablation"] = pd.read_csv(CT_ROOT / "results" / "ablation_results.csv")
R["intervention_ranking"] = pd.read_csv(CT_ROOT / "results" / "intervention_ranking.csv")

xgb_row = R["model_comparison"][R["model_comparison"].model == "XGBoost"].iloc[0]
rf_row = R["model_comparison"][R["model_comparison"].model == "RandomForest"].iloc[0]

# ============================================================== formatting
BODY_SIZE = Pt(10)
HEAD_SIZE = Pt(10.5)
CAP_SIZE = Pt(9)
FONT = "Times New Roman"


def set_run(run, bold=False, italic=False, size=BODY_SIZE, smallcaps=False, font=FONT, color=None):
    run.bold = bold
    run.italic = italic
    run.font.size = size
    run.font.name = font
    run.font.small_caps = smallcaps
    if color:
        run.font.color.rgb = RGBColor(*color)
    rFonts = run._element.rPr.rFonts if run._element.rPr is not None else None
    if rFonts is None:
        run._element.get_or_add_rPr()
    r = run._element.get_or_add_rPr().get_or_add_rFonts()
    r.set(qn('w:eastAsia'), font)


def add_runs(paragraph, segments, base_size=BODY_SIZE, base_italic=False):
    """segments: plain string with **bold** and *italic* markers (non-nested)."""
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*)', segments)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith('**') and tok.endswith('**'):
            r = paragraph.add_run(tok[2:-2])
            set_run(r, bold=True, italic=base_italic, size=base_size)
        elif tok.startswith('*') and tok.endswith('*'):
            r = paragraph.add_run(tok[1:-1])
            set_run(r, bold=False, italic=True, size=base_size)
        else:
            r = paragraph.add_run(tok)
            set_run(r, bold=False, italic=base_italic, size=base_size)


def add_heading1(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run(r, bold=False, smallcaps=True, size=HEAD_SIZE)
    return p


def add_heading2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_run(r, bold=True, italic=True, size=BODY_SIZE)
    return p


def add_body(doc, text, indent=True, justify=True):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    add_runs(p, text)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run(r, size=CAP_SIZE)
    return p


def add_table_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    set_run(r, size=CAP_SIZE, smallcaps=False)
    return p


def _set_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '4')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'auto')
        borders.append(el)
    tblPr.append(borders)


_TBLPR_ORDER = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
                'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders',
                'shd', 'tblLayout', 'tblCellMar', 'tblLook', 'tblCaption', 'tblDescription']


def _reorder_tblPr(tbl):
    """Sort tblPr's children into the canonical CT_TblPrBase sequence. Call this
    once after all ad-hoc tblPr mutations (tblW, borders, jc, tblLayout, ...)
    have been made in any order -- avoids manually tracking insertion order."""
    tblPr = tbl.tblPr

    def key(el):
        tag = el.tag.split('}')[-1]
        return _TBLPR_ORDER.index(tag) if tag in _TBLPR_ORDER else len(_TBLPR_ORDER)

    children = sorted(list(tblPr), key=key)
    for c in children:
        tblPr.remove(c)
    for c in children:
        tblPr.append(c)


def _shade_cell(cell, fill="D9D9D9"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def _set_cell_margins(cell, top=20, bottom=20, left=40, right=40):
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement('w:tcMar')
    for side, val in (('top', top), ('left', left), ('bottom', bottom), ('right', right)):
        el = OxmlElement(f'w:{side}')
        el.set(qn('w:w'), str(val))
        el.set(qn('w:type'), 'dxa')
        mar.append(el)
    tcPr.append(mar)


def add_ieee_table(doc, headers, rows, col_widths_in, header_bold=True, font_pt=7.5):
    n_cols = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    total_width = sum(col_widths_in)
    tbl = table._tbl
    old_tblW = tbl.tblPr.find(qn('w:tblW'))
    if old_tblW is not None:
        tbl.tblPr.remove(old_tblW)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), str(int(total_width * 1440)))
    tblW.set(qn('w:type'), 'dxa')
    tbl.tblPr.append(tblW)
    _set_table_borders(table)
    table.autofit = False
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.width = Inches(col_widths_in[j])
        _shade_cell(cell)
        _set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for k, line in enumerate(str(h).split("\n")):
            if k > 0:
                p.add_run().add_break()
            r = p.add_run(line)
            set_run(r, bold=header_bold, size=Pt(font_pt))
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.width = Inches(col_widths_in[j])
            _set_cell_margins(cell)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for k, line in enumerate(str(val).split("\n")):
                if k > 0:
                    p.add_run().add_break()
                r = p.add_run(line)
                set_run(r, size=Pt(font_pt))
    # force fixed column widths (grid) so autofit doesn't override
    grid = tbl.find(qn('w:tblGrid'))
    if grid is not None:
        tbl.remove(grid)
    grid = OxmlElement('w:tblGrid')
    for w in col_widths_in:
        gc = OxmlElement('w:gridCol')
        gc.set(qn('w:w'), str(int(w * 1440)))
        grid.append(gc)
    tbl.tblPr.addnext(grid)
    _reorder_tblPr(tbl)
    # keep the whole table together as one block (avoid an ugly split across
    # a column/page boundary): cantSplit on every row + keep-with-next on
    # every paragraph except those in the very last row.
    n_rows = len(table.rows)
    for ridx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement('w:cantSplit')
        trPr.append(cant_split)
        if ridx < n_rows - 1:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True
    return table


def add_figure(doc, filename, width_in, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(FIG_DIR / filename), width=Inches(width_in))
    add_caption(doc, caption)


def add_algorithm(doc, title, lines, width_in=3.3, font_pt=7.3):
    """Boxed pseudocode using a single-cell bordered table, monospace font."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl = table._tbl
    tblPr = tbl.tblPr
    old_tblW = tblPr.find(qn('w:tblW'))
    if old_tblW is not None:
        tblPr.remove(old_tblW)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), str(int(width_in * 1440)))
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '8')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'auto')
        borders.append(el)
    tblPr.append(borders)
    table.autofit = False
    grid = tbl.find(qn('w:tblGrid'))
    if grid is not None:
        tbl.remove(grid)
    grid = OxmlElement('w:tblGrid')
    gc = OxmlElement('w:gridCol')
    gc.set(qn('w:w'), str(int(width_in * 1440)))
    grid.append(gc)
    tbl.tblPr.addnext(grid)
    _reorder_tblPr(tbl)
    cell = table.cell(0, 0)
    cell.width = Inches(width_in)
    _set_cell_margins(cell, top=40, bottom=40, left=60, right=60)
    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_after = Pt(2)
    r = p0.add_run(title)
    set_run(r, bold=True, size=Pt(8.5), font="Times New Roman")
    hr = cell.add_paragraph()
    pPr = hr._p.get_or_add_pPr()
    pbdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1'); bottom.set(qn('w:color'), 'auto')
    pbdr.append(bottom)
    pPr.append(pbdr)
    hr.paragraph_format.space_after = Pt(3)
    for line in lines:
        lp = cell.add_paragraph()
        lp.paragraph_format.space_after = Pt(0)
        lp.paragraph_format.line_spacing = 1.0
        r = lp.add_run(line if line else "\u00a0")
        set_run(r, size=Pt(font_pt), font="Consolas")
    return table


def add_reference(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Pt(9)
    p.paragraph_format.first_line_indent = Pt(-9)
    p.paragraph_format.space_after = Pt(0.5)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    set_run(r, size=Pt(7.8))
    return p
