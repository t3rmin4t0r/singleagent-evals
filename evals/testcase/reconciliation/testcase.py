"""Reconciliation test case — reconcile vendor invoices against QBO expense report."""

from pathlib import Path

import pandas as pd

from evals.models import ToolDef
from evals.testcase.base import TestCase
from evals.testcase.reconciliation.tools import make_tools

SYSTEM_PROMPT = """\
You are a financial reconciliation analyst. The input files include a QBO expense report \
and vendor invoices. You also have a tool to save reconciliation results. \
Your job is to read invoices and the QBO expense report, then match vendors across the two \
sources and compare amounts.

Steps:
1. Examine the uploaded files: the Excel expense report (expenses_by_vendor_summary.xlsx) \
   and the invoice PDFs.
2. From the QBO expense report, identify vendor spending by month.
3. From each invoice PDF, extract vendor name, period, and total amount.
4. Match invoice vendors to QBO vendors — names may differ between sources. \
   Use your judgment to map them.
5. For each matched pair, note the invoice amount and the corresponding QBO amount.
6. Save the reconciliation results — the tool will compute exact/partial/unmatched status.

Real-world QBO exports are messy. Vendor names may be abbreviated or inconsistent, \
expenses may be split across line items, payments can post in the wrong month, and \
refunds or credits may appear as separate rows. Account for these when matching.

You are responsible for vendor name matching and period mapping. The save tool handles \
the numerical comparison and status classification.

When saving results, use the invoice vendor name and format period as "Mon YYYY" \
(e.g. "Jan 2025", "Feb 2025").
"""

TASK_PROMPT = """\
I have uploaded our Q1 2025 vendor invoices and our QBO expense report.

Reconcile the invoices against the QBO report.

Review the uploaded files, extract the relevant data from the expense report and invoices, \
and use the save tool to record your reconciliation results.
"""


class ReconciliationTestCase(TestCase):

    def update_pdf_metadata(self) -> None:
        """Update both PDF and XLSX metadata to bypass caching."""
        super().update_pdf_metadata()
        self.update_xlsx_metadata()

    @property
    def name(self) -> str:
        return "reconciliation"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def task_prompt(self) -> str:
        return TASK_PROMPT

    @property
    def followup_prompt(self) -> str:
        return "Check the results more carefully, this is wrong"

    @property
    def input_files(self) -> list[str]:
        """Specify which files to upload for this test case."""
        excel_file = f"expenses_by_vendor_summary_{self._version}.xlsx"
        return [
            # Excel expense report
            excel_file,  # v1 or v2 version
            # Invoice PDFs
            "2025-01_blueshield_BSC-2025-01-4421.pdf",
            "2025-01_comcast_0142-2025-01.pdf",
            "2025-01_gcp_INV-GCP-2501-8821.pdf",
            "2025-02_blueshield_BSC-2025-02-4422.pdf",
            "2025-02_comcast_0287-2025-02.pdf",
            "2025-02_comcast_0287-2025-02.pdf",  # duplicate
            "2025-02_gcp_INV-GCP-2502-9012.pdf",
            "2025-03_blueshield_BSC-2025-03-4423.pdf",
            "2025-03_comcast_0398-2025-03.pdf",
            "2025-03_gcp_INV-GCP-2503-9234.pdf",
        ]

    def get_tools(self, output_dir: Path) -> list[ToolDef]:
        return make_tools(self.data_dir, output_dir)

    def verify(self, output_dir: Path) -> dict:
        """Verify reconciliation output against golden baseline."""
        results: dict = {"structural": {}, "numerical": {}}

        # Structural checks
        csv_files = list(output_dir.glob("reconciliation_output.csv"))
        results["structural"]["csv_exists"] = len(csv_files) > 0

        # Numerical: compare against golden baseline
        recon_path = output_dir / "reconciliation_output.csv"
        golden_path = self.data_dir / f"golden_reconciliation_{self._version}.csv"

        if recon_path.exists() and golden_path.exists():
            recon = pd.read_csv(recon_path)
            golden = pd.read_csv(golden_path)

            results["numerical"]["output_rows"] = len(recon)
            results["numerical"]["golden_rows"] = len(golden)

            def _norm(s: str) -> str:
                return str(s).lower().strip()

            output_lookup: dict[tuple[str, str], pd.Series] = {}
            for _, row in recon.iterrows():
                key = (_norm(row.get("vendor", "")), _norm(row.get("period", "")))
                output_lookup[key] = row

            mismatches: list[str] = []

            for _, grow in golden.iterrows():
                gv = _norm(grow["vendor"])
                gp = _norm(grow["period"])

                orow = output_lookup.get((gv, gp))
                if orow is None:
                    for (ov, op), candidate in output_lookup.items():
                        if gp == op and (gv in ov or ov in gv):
                            orow = candidate
                            break

                if orow is None:
                    mismatches.append(f"{grow['vendor']} {grow['period']}: row missing from output")
                    continue

                inv_ok = abs(float(orow.get("invoice_amount", 0)) - float(grow["invoice_amount"])) < 0.01
                qbo_ok = abs(float(orow.get("qbo_amount", 0)) - float(grow["qbo_amount"])) < 0.01
                status_ok = _norm(orow.get("status", "")) == _norm(grow["status"])

                if not inv_ok or not qbo_ok or not status_ok:
                    fields: list[str] = []
                    if not inv_ok:
                        fields.append(f"invoice_amount: got {orow.get('invoice_amount')}, expected {grow['invoice_amount']}")
                    if not qbo_ok:
                        fields.append(f"qbo_amount: got {orow.get('qbo_amount')}, expected {grow['qbo_amount']}")
                    if not status_ok:
                        fields.append(f"status: got {orow.get('status')}, expected {grow['status']}")
                    mismatches.append(f"{grow['vendor']} {grow['period']}: {'; '.join(fields)}")

            n = len(golden)
            matched = n - len(mismatches)
            results["numerical"]["rows_matched"] = matched
            results["numerical"]["rows_total"] = n
            results["numerical"]["mismatches"] = mismatches

        # Top-level pass/score — binary: perfect match or fail
        num = results["numerical"]
        if num.get("rows_matched") == num.get("rows_total") and num.get("rows_total", 0) > 0:
            results["passed"] = True
            results["score"] = "PASS"
        elif "mismatches" in num:
            results["passed"] = False
            m = num["rows_matched"]
            n = num["rows_total"]
            results["score"] = f"FAIL ({m}/{n} rows)"
            results["reason"] = num["mismatches"]
        else:
            results["passed"] = False
            results["score"] = "FAIL (no output)"

        return results
