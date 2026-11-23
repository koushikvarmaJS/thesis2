"""Build the paper draft (Word) from the course template: python3 paper/build_paper.py

Fills "Paper template.docx" (one folder up from thesis2/) with the draft text in paper/content.py, adds a
contents page after the abstract, and writes "Paper draft.docx" next to the template. Numbers in the text come
from the report (Jetson_KV_Report_v2.pdf) and plots_v2/stats*.json; figures are the PNGs in plots_v2/.
"""
import copy
import re
import sys
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

TEMPLATE = ROOT.parent / 'Paper template.docx'
OUT = ROOT.parent / 'Paper draft.docx'
COL_W = 3.3   # inches, one column of the two-column body
PAGE_W = 7.0  # inches, full text width


def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


class Paper:
    def __init__(self):
        self.d = docx.Document(str(TEMPLATE))
        self.body = self.d.element.body
        self.cites = {}          # key -> number, in order of first citation
        self.tab_n = self.fig_n = 0
        self.labels = {}         # label -> "Table 3" / "Figure 2"
        self.first_body_section = True
        kids = list(self.body)
        self.sect_1col = copy.deepcopy(kids[8].find('.//' + qn('w:sectPr')))   # abstract/contents section
        self.sect_2col = copy.deepcopy(self.body.find(qn('w:sectPr')))         # body section
        # copies of these close the abstract/contents section and every body section: they must not restart the
        # page numbers or suppress the header on their first page (the template sets both on every section)
        for s in (self.sect_1col, self.sect_2col):
            for tag in ('w:type', 'w:pgNumType', 'w:titlePg'):
                for t in s.findall(qn(tag)):
                    s.remove(t)

    # ------------------------------------------------------------ text helpers
    def runs(self, p, text):
        """Inline markup: **bold**, *italic*, {cite:a,b}, {ref:label}, ^sup^."""
        text = re.sub(r'\{cite:([^}]+)\}', lambda m: self.cite(m[1]), text)
        text = re.sub(r'\{ref:([^}]+)\}', lambda m: self.labels.get(m[1], f'[{m[1]}?]'), text)
        for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^)', text):
            if not part:
                continue
            if part.startswith('**'):
                p.add_run(part[2:-2]).bold = True
            elif part.startswith('*'):
                p.add_run(part[1:-1]).italic = True
            elif part.startswith('^'):
                p.add_run(part[1:-1]).font.superscript = True
            else:
                p.add_run(part)
        return p

    def cite(self, keys):
        nums = []
        for k in [k.strip() for k in keys.split(',')]:
            if k not in C.REFS:
                raise KeyError(f'unknown reference {k}')
            self.cites.setdefault(k, len(self.cites) + 1)
            nums.append(self.cites[k])
        return '[' + ', '.join(str(n) for n in sorted(nums)) + ']'

    def para(self, text, style='Para'):
        return self.runs(self.d.add_paragraph(style=style), text)

    def head(self, text, level=1):
        return self.d.add_paragraph(text.replace(' ', ' ', 1), style=f'Head{level}')

    # ------------------------------------------------------------ sections and floats
    def _end_section(self, proto, new_page=False):
        """Close the running section with a copy of proto (its column layout applies to what came before)."""
        p = self.d.add_paragraph(style='Para')
        s = copy.deepcopy(proto)
        s.find(qn('w:pgSz')).addprevious(el('w:type', **{'w:val': 'nextPage' if new_page else 'continuous'}))
        p._p.get_or_add_pPr().append(s)
        pf = p.paragraph_format
        pf.space_before = pf.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(1)

    def wide_begin(self):
        self._end_section(self.sect_2col, new_page=self.first_body_section)
        self.first_body_section = False

    def wide_end(self):
        self._end_section(self.sect_1col)

    def figure(self, label, path, caption, wide=False, width=None):
        if wide:
            self.wide_begin()
        self.fig_n += 1
        p = self.d.add_paragraph(style='Image')
        p.add_run().add_picture(str(ROOT / path), width=Inches(width or (PAGE_W if wide else COL_W)))
        p.paragraph_format.keep_with_next = True
        self.runs(self.d.add_paragraph(style='FigureCaption'), f'Figure {self.fig_n}: ' + caption)
        if wide:
            self.wide_end()

    def table(self, label, caption, rows, widths, note=None, wide=False, size=7.5):
        if wide:
            self.wide_begin()
        self.tab_n += 1
        cap = self.runs(self.d.add_paragraph(style='TableCaption'), f'Table {self.tab_n}: ' + caption)
        cap.paragraph_format.space_before = Pt(10)
        cap.paragraph_format.space_after = Pt(4)
        cap.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=len(rows), cols=len(rows[0]))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tw = sum(widths)
        tblPr = t._tbl.tblPr
        tblW = tblPr.find(qn('w:tblW'))
        tblW.set(qn('w:w'), str(int(tw * 1440)))
        tblW.set(qn('w:type'), 'dxa')
        borders = el('w:tblBorders')
        for side in ('top', 'bottom'):
            borders.append(el(f'w:{side}', **{'w:val': 'single', 'w:sz': 8, 'w:space': 0, 'w:color': '000000'}))
        after = tblPr.find(qn('w:jc')) if tblPr.find(qn('w:jc')) is not None else tblW
        after.addnext(borders)
        grid = t._tbl.tblGrid
        for gc, w in zip(grid.findall(qn('w:gridCol')), widths):
            gc.set(qn('w:w'), str(int(w * 1440)))
        for i, row in enumerate(rows):
            tr = t.rows[i]
            if i == 0:
                trPr = tr._tr.get_or_add_trPr()
                trPr.append(el('w:tblHeader'))
            for j, val in enumerate(row):
                cell = tr.cells[j]
                cell.width = Inches(widths[j])
                tcPr = cell._tc.get_or_add_tcPr()
                if i == 0:
                    b = el('w:tcBorders')
                    b.append(el('w:bottom', **{'w:val': 'single', 'w:sz': 4, 'w:space': 0, 'w:color': '000000'}))
                    tcPr.append(b)
                mar = el('w:tcMar')
                for side in ('left', 'right'):
                    mar.append(el(f'w:{side}', **{'w:w': 50, 'w:type': 'dxa'}))
                tcPr.append(mar)
                p = cell.paragraphs[0]
                p.style = self.d.styles['Normal']
                pf = p.paragraph_format
                pf.space_before = pf.space_after = Pt(1)
                pf.line_spacing = 1.0
                p.alignment = 0 if j == 0 else 1
                self.runs(p, str(val))
                for r in p.runs:
                    r.font.size = Pt(size)
                    if i == 0:
                        r.bold = True
                p.paragraph_format.keep_with_next = i < len(rows) - 1
        if note:
            n = self.runs(self.d.add_paragraph(style='TableFootnote'), note)
            n.paragraph_format.space_after = Pt(8)
        else:
            self.d.add_paragraph(style='Para').paragraph_format.space_after = Pt(2)
        if wide:
            self.wide_end()

    # ------------------------------------------------------------ front matter
    def front(self):
        kids = list(self.body)
        kids[0].getparent()
        p_title, p_sub = self.d.paragraphs[0], self.d.paragraphs[1]
        p_title.text, p_sub.text = C.TITLE, C.SUBTITLE
        authors = [p for p in self.d.paragraphs[3:6]]
        for i, (p, a) in enumerate(zip(authors, C.AUTHORS + [None] * 3)):
            for r in list(p._p):
                if r.tag != qn('w:pPr'):
                    p._p.remove(r)
            if a is None:
                continue
            lines = a.split('\n')
            p.paragraph_format.space_before = Pt(6)  # same top for all three columns
            # one author per column: Word balances a continuous 3-column section by height, which lines the
            # authors up only when every block has the same number of lines. A column break at the *start* of
            # authors 2 and 3 forces it (at the end of a block it would leave an empty line atop the next column)
            if i > 0:
                p.add_run().add_break(WD_BREAK.COLUMN)
            for k, line in enumerate(lines):
                run = p.add_run(line)
                run.font.size = Pt(12 if k == 0 else 9 if '@' in line else 10)
                if k < len(lines) - 1:
                    run.add_break()
        # the third author box stays empty if there are only two authors
        self.d.paragraphs[6].text = ''
        self.d.paragraphs[7].text = ''
        absp = self.d.paragraphs[10]
        absp.text = ''
        self.runs(absp, C.ABSTRACT[0])
        prev = absp._p
        for t in C.ABSTRACT[1:]:
            np_ = self.runs(self.d.add_paragraph(style='Abstract'), t)._p
            prev.addnext(np_)
            prev = np_
        # remove the template's body (Introduction ... metadata), keep the final sectPr
        start = list(self.body).index(absp._p) + 1 + len(C.ABSTRACT) - 1
        for e in list(self.body)[start:]:
            if e.tag != qn('w:sectPr'):
                self.body.remove(e)
        # the abstract section (1 column) now ends after the contents page: move its sectPr there
        sp = list(self.body)[8].find('.//' + qn('w:sectPr'))
        sp.getparent().remove(sp)

    def keywords_and_contents(self):
        self.d.add_paragraph('KEYWORDS', style='KeyWordHead')
        self.para(C.KEYWORDS, 'KeyWords')
        p = self.d.add_paragraph(style='AbsHead')
        p.add_run().add_break(WD_BREAK.PAGE)
        p.add_run('CONTENTS')
        # TOC field: Word fills in the page numbers (it asks to update fields when the file opens)
        p = self.d.add_paragraph(style='TOC1')
        r = p.add_run()
        r._r.append(el('w:fldChar', **{'w:fldCharType': 'begin'}))
        r = p.add_run()
        it = el('w:instrText', **{'xml:space': 'preserve'})
        it.text = ' TOC \\o "1-2" \\h \\z \\u \\t "Head1,1,Head2,2" '
        r._r.append(it)
        r = p.add_run()
        r._r.append(el('w:fldChar', **{'w:fldCharType': 'separate'}))
        # placeholder entries until Word updates the field
        first = True
        for level, text in self.toc_entries:
            q = p if first else self.d.add_paragraph(style='TOC1' if level == 1 else 'TOC2')
            first = False
            q.paragraph_format.left_indent = Inches(0 if level == 1 else 0.25)
            q.paragraph_format.space_after = Pt(2 if level == 2 else 4)
            q.paragraph_format.space_before = Pt(0 if level == 2 else 4)
            run = q.add_run(text)
            run.bold = level == 1
            run.font.size = Pt(10 if level == 1 else 9)
        end = self.d.add_paragraph(style='TOC2')
        r = end.add_run()
        r._r.append(el('w:fldChar', **{'w:fldCharType': 'end'}))
        note = self.para('Page numbers appear when Word updates this table (it asks when the file opens; or '
                         'right-click the table and choose Update Field).', 'TableFootnote')
        note.paragraph_format.space_before = Pt(10)
        s = copy.deepcopy(self.sect_1col)
        s.find(qn('w:pgSz')).addprevious(el('w:type', **{'w:val': 'continuous'}))  # abstract follows the authors
        note._p.get_or_add_pPr().append(s)

    def headers(self):
        """Replace the template's placeholder running heads (conference and author names)."""
        # the strings sit in tables/frames inside the header parts, so edit every text node of every header part
        for rel in self.d.part.rels.values():
            if rel.reltype.endswith('/header'):
                for t in rel.target_part.element.iter(qn('w:t')):
                    for old, new in C.RUNNING_HEADS.items():
                        if t.text and old in t.text:
                            t.text = t.text.replace(old, new)

    def finish(self):
        self.headers()
        # the body's last section: no page-number restart, no title-page header suppression
        last = self.body.find(qn('w:sectPr'))
        for tag in ('w:pgNumType', 'w:titlePg'):
            for t in last.findall(qn(tag)):
                last.remove(t)
        # outline levels so the headings feed the contents page and Word's navigation pane
        for name, lvl in (('Head1', 0), ('Head2', 1)):
            pPr = self.d.styles[name].element.get_or_add_pPr()
            pPr.append(el('w:outlineLvl', **{'w:val': lvl}))
        st = self.d.settings.element
        later = ('hdrShapeDefaults', 'footnotePr', 'endnotePr', 'compat', 'docVars', 'rsids', 'mathPr',
                 'attachedSchema', 'themeFontLang', 'clrSchemeMapping', 'doNotIncludeSubdocsInStats',
                 'doNotAutoCompressPictures', 'forceUpgrade', 'captions', 'readModeInkLockDown', 'smartTagType',
                 'schemaLibrary', 'shapeDefaults', 'doNotEmbedSmartTags', 'decimalSymbol', 'listSeparator')
        nxt = next((c for c in st if c.tag.split('}')[1] in later), None)
        uf = el('w:updateFields', **{'w:val': 'true'})
        nxt.addprevious(uf) if nxt is not None else st.append(uf)
        # the body's last section starts on a new page if no wide float has closed one before
        last = self.body.find(qn('w:sectPr'))
        if self.first_body_section:
            last.find(qn('w:pgSz')).addprevious(el('w:type', **{'w:val': 'nextPage'}))
        cp = self.d.core_properties
        cp.title, cp.author = C.TITLE, C.AUTHOR_META
        self.d.save(str(OUT))
        print('wrote', OUT)


def main():
    P = Paper()
    # first pass collects the headings for the contents page
    P.toc_entries = [(lvl, txt) for kind, *rest in C.BODY if kind == 'h' for lvl, txt in [(rest[1], rest[0])]]
    nf = nt = 0
    for kind, *a in C.BODY:  # number tables and figures up front so text can refer forward
        if kind == 'fig':
            nf += 1
            P.labels[a[0]] = str(nf)
        elif kind == 'tab':
            nt += 1
            P.labels[a[0]] = str(nt)
    P.front()
    P.keywords_and_contents()
    for kind, *a in C.BODY:
        if kind == 'h':
            P.head(a[0], a[1])
        elif kind == 'p':
            P.para(a[0])
        elif kind == 'pc':
            P.para(a[0], 'ParaContinue')
        elif kind == 'fig':
            P.figure(*a[:3], **(a[3] if len(a) > 3 else {}))
        elif kind == 'tab':
            P.table(*a[:4], **(a[4] if len(a) > 4 else {}))
        elif kind == 'refs':
            P.d.add_paragraph('ACKNOWLEDGMENTS', style='AckHead')
            P.para(C.ACK, 'AckPara')
            P.d.add_paragraph('REFERENCES', style='ReferenceHead')
            for k, n in sorted(P.cites.items(), key=lambda x: x[1]):
                P.para(f'[{n}]\t' + C.REFS[k], 'Bib_entry')
            unused = set(C.REFS) - set(P.cites)
            if unused:
                print('unused references:', sorted(unused))
    P.finish()


if __name__ == '__main__':
    main()
