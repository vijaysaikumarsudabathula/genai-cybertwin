# -*- coding: utf-8 -*-
"""Remove everything from 'I. INTRODUCTION' onward (old ExARG body), keeping
the title/author/abstract/keywords block already rewritten, and keeping the
trailing sectPr (two-column section properties) intact."""
import docx
from docx.oxml.ns import qn
from build_helpers import OUT

doc = docx.Document(str(OUT))
body = doc.element.body

# find the paragraph whose text is 'I. INTRODUCTION'
start_idx = None
children = list(body)
for i, child in enumerate(children):
    if child.tag == qn('w:p'):
        text = "".join(node.text or "" for node in child.iter(qn('w:t')))
        if text.strip() == "I. INTRODUCTION":
            start_idx = i
            break

assert start_idx is not None, "Could not find 'I. INTRODUCTION' heading"

# the very last child is the trailing sectPr (not a paragraph) -- keep it
end_idx = len(children) - 1
assert children[-1].tag == qn('w:sectPr'), f"expected trailing sectPr, got {children[-1].tag}"

for child in children[start_idx:end_idx]:
    body.remove(child)

doc.save(str(OUT))
print(f"Removed body children [{start_idx}:{end_idx}) ; kept trailing sectPr")
print("Remaining children:", len(list(doc.element.body)))
