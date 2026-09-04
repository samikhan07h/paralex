"""
Dev utility: generates a sample financial statements PDF containing REAL
tables (built with reportlab's Table flowable, which draws actual grid
lines and cell boundaries) rather than text formatted to merely look like
a table. This distinction matters: pdfplumber's extract_tables() detects
tables by finding grid/ruling lines in the PDF's underlying structure — a
document with text manually aligned into columns (e.g. via spaces) has no
such structure and won't be detected as a table at all. A genuine reportlab
Table is what makes this a meaningful test of table EXTRACTION rather than
a document we already know can't exercise that code path.

Not part of the production pipeline — same category as
scripts/generate_sample_docs.py from Phase 1.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()

doc = SimpleDocTemplate(
    str(OUT_DIR / "sample_financial_statements.pdf"),
    pagesize=letter,
    topMargin=50, bottomMargin=50,
)
story = []

story.append(Paragraph("Beacon Industries, Inc. — Consolidated Financial Statements (FY2024)", styles["Title"]))
story.append(Spacer(1, 16))

# --- Income Statement table ---
story.append(Paragraph("Consolidated Statement of Operations (in millions)", styles["Heading2"]))
story.append(Spacer(1, 8))

income_statement_data = [
    ["Line Item", "FY2024", "FY2023"],
    ["Net revenue", "$3,150", "$2,780"],
    ["Cost of revenue", "$1,260", "$1,120"],
    ["Gross profit", "$1,890", "$1,660"],
    ["Research and development", "$410", "$360"],
    ["Sales and marketing", "$380", "$340"],
    ["General and administrative", "$290", "$260"],
    ["Total operating expenses", "$1,080", "$960"],
    ["Operating income", "$810", "$700"],
    ["Net income", "$645", "$560"],
]

income_table = Table(income_statement_data, colWidths=[220, 100, 100])
income_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
]))
story.append(income_table)
story.append(Spacer(1, 24))

# --- Balance Sheet table ---
story.append(Paragraph("Consolidated Balance Sheet (in millions)", styles["Heading2"]))
story.append(Spacer(1, 8))

balance_sheet_data = [
    ["Line Item", "FY2024", "FY2023"],
    ["Cash and cash equivalents", "$920", "$780"],
    ["Accounts receivable", "$410", "$360"],
    ["Total current assets", "$1,780", "$1,520"],
    ["Property and equipment, net", "$650", "$600"],
    ["Total assets", "$3,050", "$2,650"],
    ["Accounts payable", "$240", "$210"],
    ["Total current liabilities", "$520", "$470"],
    ["Long-term debt", "$560", "$520"],
    ["Total liabilities", "$1,180", "$1,090"],
    ["Total stockholders' equity", "$1,870", "$1,560"],
]

balance_table = Table(balance_sheet_data, colWidths=[220, 100, 100])
balance_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
]))
story.append(balance_table)
story.append(Spacer(1, 24))

story.append(Paragraph(
    "Note: Figures are illustrative sample data for testing purposes and do not represent an actual company.",
    styles["Italic"],
))

doc.build(story)
print(f"Created {OUT_DIR / 'sample_financial_statements.pdf'}")
