"""
Dev utility: generates synthetic-but-realistic sample PDFs for testing the
ingestion pipeline before real client documents are available.

NOTE: These are placeholder documents so Phase 1 has something concrete to
test against end-to-end. Swap in real (or real-looking public) contracts,
leases, and 10-K excerpts in data/sample_docs/ before the final GitHub demo
for authenticity. This script is not part of the production pipeline and
its dependency (fpdf2) is intentionally NOT in requirements.txt.
"""

from fpdf import FPDF
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_pdf(filename: str, title: str, body_lines: list[str]):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    for line in body_lines:
        pdf.multi_cell(0, 7, line)
        pdf.ln(2)
    pdf.output(str(OUT_DIR / filename))
    print(f"Created {OUT_DIR / filename}")


lease_body = [
    "1. PARTIES. This Residential Lease Agreement (\"Agreement\") is entered into "
    "between Meridian Properties LLC (\"Landlord\") and the undersigned tenant "
    "(\"Tenant\") for the property located at 48 Harrington Street, Unit 3B.",

    "2. TERM. The lease term shall commence on January 1, 2025 and terminate on "
    "December 31, 2025, unless renewed or terminated earlier in accordance with "
    "Section 9 of this Agreement.",

    "3. RENT. Tenant shall pay Landlord monthly rent of $2,400.00, due on the 1st "
    "day of each month. A late fee of $75.00 applies to payments received after "
    "the 5th day of the month.",

    "4. SECURITY DEPOSIT. Tenant shall pay a security deposit of $2,400.00 prior "
    "to occupancy. The deposit shall be returned within 30 days of move-out, less "
    "any deductions for damages beyond normal wear and tear.",

    "5. UTILITIES. Tenant is responsible for electricity, gas, and internet. "
    "Landlord is responsible for water, sewer, and trash collection.",

    "6. MAINTENANCE. Tenant shall promptly notify Landlord of any needed repairs. "
    "Landlord shall address emergency repairs within 24 hours and non-emergency "
    "repairs within 7 business days.",

    "7. PETS. No pets are permitted without prior written consent of Landlord. "
    "If consent is granted, an additional pet deposit of $500.00 shall apply.",

    "8. SUBLETTING. Tenant shall not sublet the premises, in whole or in part, "
    "without prior written consent of Landlord.",

    "9. TERMINATION. Either party may terminate this Agreement upon 60 days' "
    "written notice. Early termination by Tenant prior to the end of the term "
    "shall incur a fee equal to two (2) months' rent.",

    "10. GOVERNING LAW. This Agreement shall be governed by the laws of the "
    "State of New York.",
]

loan_body = [
    "1. LOAN AMOUNT. Lender agrees to loan Borrower the principal sum of "
    "$150,000.00 (the \"Loan\"), subject to the terms of this Agreement.",

    "2. INTEREST RATE. The Loan shall bear interest at a fixed annual rate of "
    "6.75%, calculated on the outstanding principal balance.",

    "3. REPAYMENT SCHEDULE. Borrower shall repay the Loan in 60 equal monthly "
    "installments of $2,946.31, beginning 30 days after the Effective Date.",

    "4. PREPAYMENT. Borrower may prepay all or part of the outstanding principal "
    "at any time without penalty.",

    "5. LATE PAYMENT. Any payment not received within 10 days of its due date "
    "shall incur a late fee of 5% of the overdue amount.",

    "6. DEFAULT. Borrower shall be in default if any payment is more than 30 "
    "days past due, or upon breach of any other material term of this Agreement. "
    "Upon default, Lender may declare the entire unpaid balance immediately due.",

    "7. COLLATERAL. This Loan is secured by the commercial equipment described "
    "in Exhibit A, in which Borrower grants Lender a security interest.",

    "8. REPRESENTATIONS. Borrower represents that it has full authority to enter "
    "into this Agreement and that the Loan proceeds will be used solely for "
    "business operating purposes.",

    "9. GOVERNING LAW. This Agreement shall be governed by the laws of the "
    "State of Delaware.",
]

tenk_body = [
    "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND "
    "RESULTS OF OPERATIONS",

    "Overview. Total net revenue for fiscal year 2024 was $4.82 billion, an "
    "increase of 11.3% compared to $4.33 billion in fiscal year 2023. The "
    "increase was primarily driven by growth in our subscription services "
    "segment, which grew 18.4% year-over-year.",

    "Gross Profit. Gross profit for fiscal 2024 was $2.91 billion, representing "
    "a gross margin of 60.4%, compared to a gross margin of 58.7% in fiscal "
    "2023. The margin improvement reflects increased operating leverage in our "
    "cloud infrastructure.",

    "Operating Expenses. Total operating expenses increased to $1.98 billion in "
    "fiscal 2024 from $1.76 billion in fiscal 2023, primarily due to increased "
    "headcount in research and development, which grew from 3,200 to 3,850 "
    "employees.",

    "Net Income. Net income for fiscal 2024 was $612 million, or $2.14 per "
    "diluted share, compared to net income of $498 million, or $1.76 per "
    "diluted share, in fiscal 2023.",

    "Liquidity and Capital Resources. As of the end of fiscal 2024, the Company "
    "had cash and cash equivalents of $1.15 billion and total debt of $800 "
    "million. The Company believes its existing cash balances, together with "
    "cash generated from operations, will be sufficient to meet working capital "
    "and capital expenditure needs for at least the next 12 months.",

    "Risk Factors Summary. The Company's results may be affected by factors "
    "including foreign currency fluctuations, customer concentration in the "
    "enterprise segment, and increased competition in the cloud infrastructure "
    "market.",
]

make_pdf("sample_lease_agreement.pdf", "Residential Lease Agreement", lease_body)
make_pdf("sample_loan_agreement.pdf", "Commercial Loan Agreement", loan_body)
make_pdf("sample_10k_excerpt.pdf", "Form 10-K Excerpt - Item 7 MD&A (Sample Corp.)", tenk_body)
