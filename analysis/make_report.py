"""Build Jetson_KV_Report_v2.pdf from plots_v2/stats.json (run analyze_v2.py first).

Needs reportlab and matplotlib. Usage: python3 analysis/make_report.py
Every number in the results sections is read from stats.json; nothing is typed in by hand
except the v1 numbers quoted for comparison and the text explaining them.
"""
import json
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / 'plots_v2'
S = json.loads((PLOTS / 'stats.json').read_text())

FONT_DIR = Path('/System/Library/Fonts/Supplemental')
pdfmetrics.registerFont(TTFont('Arial', str(FONT_DIR / 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', str(FONT_DIR / 'Arial Bold.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Italic', str(FONT_DIR / 'Arial Italic.ttf')))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic')

INK, MUTED, ACCENT, RULE, SHADE, HIL = (colors.HexColor(c) for c in
                                         ('#10201C', '#4A5568', '#B65A17', '#D9D4C7', '#EFEBE2', '#F6E3D3'))
st_body = ParagraphStyle('body', fontName='Arial', fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5)
st_small = ParagraphStyle('small', parent=st_body, fontSize=8.2, leading=11, textColor=MUTED)
st_cell = ParagraphStyle('cell', parent=st_body, fontSize=8.2, leading=10.6, spaceAfter=0)
st_cellb = ParagraphStyle('cellb', parent=st_cell, fontName='Arial-Bold')
st_h1 = ParagraphStyle('h1', parent=st_body, fontName='Arial-Bold', fontSize=15, leading=19, spaceBefore=12,
                       spaceAfter=6, keepWithNext=1)
st_h2 = ParagraphStyle('h2', parent=st_body, fontName='Arial-Bold', fontSize=11.5, leading=15, spaceBefore=9,
                       spaceAfter=4, keepWithNext=1)
st_title = ParagraphStyle('title', parent=st_body, fontName='Arial-Bold', fontSize=23, leading=28, spaceAfter=10)
st_sub = ParagraphStyle('sub', parent=st_body, fontSize=12, leading=16, textColor=MUTED, spaceAfter=8)
st_box = ParagraphStyle('box', parent=st_body, backColor=SHADE, borderPadding=7, spaceBefore=6, spaceAfter=10)
st_code = ParagraphStyle('code', parent=st_body, fontName='Courier', fontSize=7.8, leading=10, textColor=INK,
                         backColor=SHADE, borderPadding=5, spaceAfter=8, alignment=TA_LEFT)

LL, QW = 'Llama-3.2-1B-Instruct-Q8_0', 'qwen2.5-0.5b-instruct-q8_0'
LL4, QW4, L3 = 'Llama-3.2-1B-Instruct-Q4_K_M', 'qwen2.5-0.5b-instruct-q4_k_m', 'Llama-3.2-3B-Instruct-Q4_K_M'
NAME = {LL: 'Llama-1B', QW: 'Qwen-0.5B', LL4: 'Llama-1B (Q4_K_M weights)', QW4: 'Qwen-0.5B (Q4_K_M weights)',
        L3: 'Llama-3B (Q4_K_M weights)'}
KVN = {'f16': 'f16', 'f16@off': 'f16, flash attn off', 'q8_0': '8-bit', 'q4_0': '4-bit'}
CTX = [1024, 4096, 8192, 16384]


# ---------------------------------------------------------------- data access

def get(metric, tag, model, kv, ctx, np_=1):
    return S.get(metric, {}).get(f'{tag}|{model}|{kv}|{ctx}|{np_}')


def mean(metric, tag, model, kv, ctx, np_=1):
    v = get(metric, tag, model, kv, ctx, np_)
    return v[0] if v else float('nan')


def chg(metric, tag, model, kv, ctx, np_=1, show_ci=True):
    """'+14.7% ± 1.9' (paired, 95% CI), with '(n.s.)' when the CI includes zero."""
    v = S.get(f'change_{metric}', {}).get(f'{tag}|{model}|{kv}|{ctx}|{np_}')
    if not v:
        return '–'
    m, half, n, sig = v
    s = f'{m:+.1f}%' + (f' ± {half:.1f}' if show_ci and half == half else '')
    return s if sig else f'{s} (n.s.)'


def sig(metric, tag, model, kv, ctx, np_=1):
    v = S.get(f'change_{metric}', {}).get(f'{tag}|{model}|{kv}|{ctx}|{np_}')
    return bool(v and v[3])


def f(x, d=1):
    return '–' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:,.{d}f}'


# ---------------------------------------------------------------- layout helpers

def P(t, s=st_body):
    return Paragraph(t, s)


def table(rows, widths, header=True, hl_rows=(), font=st_cell):
    data = [[c if not isinstance(c, str) else Paragraph(c, st_cellb if (header and i == 0) else font)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), 0.4, RULE),
             ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
             ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        style.append(('BACKGROUND', (0, 0), (-1, 0), SHADE))
    for r in hl_rows:
        style.append(('BACKGROUND', (0, r), (-1, r), HIL))
    t.setStyle(TableStyle(style))
    return t


def figure(path, caption, width=170):
    if not Path(path).exists():
        return [P(f'<i>(figure missing: {Path(path).name})</i>', st_small)]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width * mm
    return [KeepTogether([Image(str(path), width=w, height=w * ih / iw), P(caption, st_small)])]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Arial', 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, 'KV-cache quantization on Jetson Orin Nano: revised report (v2)')
    canvas.drawRightString(192 * mm, 10 * mm, f'Page {doc.page}')
    canvas.restoreState()


def build(story, out):
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title='KV-Cache Quantization on Jetson Orin Nano (v2)',
                            author='Koushik Bhupathiraju')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('wrote', out)


if __name__ == '__main__':
    import report_content  # noqa: E402  (the sections; kept separate so this file stays about layout)
    build(report_content.story(), ROOT / 'Jetson_KV_Report_v2.pdf')
