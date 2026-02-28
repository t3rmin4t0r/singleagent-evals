"""Reconciliation-specific tools — save results and create Excel report.

Input files are uploaded via provider-specific APIs (OpenAI Files API, Claude files, etc.)
and referenced directly in the system prompt. This file only contains tools for saving
and formatting reconciliation results.
"""

import json
from difflib import unified_diff
from pathlib import Path

import pandas as pd
import xlsxwriter

from bench.models import ToolDef

# JSON Schemas
SAVE_RECONCILIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reconciliation_json": {
            "type": "string",
            "description": (
                'JSON array of matched records. Each: '
                '{"vendor": "Comcast Business", "period": "Jan 2025", "invoice_amount": 189.00, "qbo_amount": 189.00}'
            ),
        },
    },
    "required": ["reconciliation_json"],
}

CREATE_RECONCILIATION_EXCEL_SCHEMA = {
    "type": "object",
    "properties": {
        "reconciliation_csv_path": {"type": "string", "default": "output/reconciliation_output.csv"},
        "output_filename": {"type": "string", "default": "reconciliation_report.xlsx"},
    },
    "required": [],
}

GENERATE_RECONCILIATION_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "reconciliation_csv_path": {"type": "string", "default": "output/reconciliation_output.csv"},
    },
    "required": [],
}


def make_tools(data_dir: Path, output_dir: Path) -> list[ToolDef]:
    """Create reconciliation tool definitions with data_dir/output_dir captured in closures."""

    def _resolve_to_output(path_str: str) -> Path:
        """Resolve paths that the LLM provides for output files."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        parts = p.parts
        if parts and parts[0] == "output":
            return output_dir / Path(*parts[1:])
        return output_dir / p

    # -- save_reconciliation --
    def save_reconciliation(reconciliation_json: str) -> str:
        try:
            records = json.loads(reconciliation_json)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"

        if not isinstance(records, list):
            return "Error: reconciliation_json must be a JSON array."

        results = []
        for rec in records:
            vendor = rec.get("vendor", "")
            period = rec.get("period", "")
            inv_amt = float(rec.get("invoice_amount", 0) or 0)
            qbo_amt = float(rec.get("qbo_amount", 0) or 0)

            diff = round(inv_amt - qbo_amt, 2)
            if qbo_amt == 0 and inv_amt > 0:
                status = "Unmatched"
                notes = "No matching QBO entry"
            elif abs(diff) < 0.01:
                status = "Matched"
                notes = "Exact match"
            else:
                status = "Partial"
                if diff > 0:
                    notes = f"${abs(diff):.2f} short in QBO"
                else:
                    notes = f"${abs(diff):.2f} over in QBO"

            results.append({
                "vendor": vendor,
                "period": period,
                "invoice_amount": inv_amt,
                "qbo_amount": qbo_amt,
                "status": status,
                "notes": notes,
            })

        out_df = pd.DataFrame(results)
        out_path = output_dir / "reconciliation_output.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        matched = sum(1 for r in results if r["status"] == "Matched")
        partial = sum(1 for r in results if r["status"] == "Partial")
        unmatched = sum(1 for r in results if r["status"] == "Unmatched")
        total_inv = sum(r["invoice_amount"] for r in results)
        total_qbo = sum(r["qbo_amount"] for r in results)

        lines = [
            f"Reconciliation saved: {len(results)} line items",
            f"  Matched: {matched}  |  Partial: {partial}  |  Unmatched: {unmatched}",
            f"  Total invoiced: ${total_inv:,.2f}",
            f"  Total in QBO:   ${total_qbo:,.2f}",
            f"  Difference:     ${total_inv - total_qbo:,.2f}",
            f"Saved to: {out_path}",
            "",
            out_df.to_csv(index=False),
        ]
        return "\n".join(lines)

    # -- generate_reconciliation_report --
    def generate_reconciliation_report(reconciliation_csv_path: str = "output/reconciliation_output.csv") -> str:
        """Generate a unified diff-style report of the reconciliation results."""
        path = _resolve_to_output(reconciliation_csv_path)
        if not path.exists():
            return f"Error: {reconciliation_csv_path} not found."

        df = pd.read_csv(path)

        # Create expected (input) and actual (output) line representations
        expected_lines = ["Reconciliation Report", "=" * 80, ""]
        actual_lines = ["Reconciliation Report", "=" * 80, ""]

        # Group by vendor for better organization
        vendors = df["vendor"].unique()
        for vendor in sorted(vendors):
            vendor_df = df[df["vendor"] == vendor].sort_values("period")
            actual_lines.append(f"\nVendor: {vendor}")
            actual_lines.append("-" * 40)

            for _, row in vendor_df.iterrows():
                period = row["period"]
                inv_amt = f"${row['invoice_amount']:,.2f}"
                qbo_amt = f"${row['qbo_amount']:,.2f}"
                status = row["status"]
                notes = row["notes"]

                actual_lines.append(f"  {period:12} | Invoice: {inv_amt:>12} | QBO: {qbo_amt:>12} | {status:10} | {notes}")

        # Summary section
        actual_lines.append("")
        actual_lines.append("=" * 80)
        actual_lines.append("Summary")
        actual_lines.append("-" * 40)

        matched = sum(1 for _, r in df.iterrows() if r["status"] == "Matched")
        partial = sum(1 for _, r in df.iterrows() if r["status"] == "Partial")
        unmatched = sum(1 for _, r in df.iterrows() if r["status"] == "Unmatched")
        total_inv = df["invoice_amount"].sum()
        total_qbo = df["qbo_amount"].sum()

        actual_lines.append(f"Total Records:  {len(df)}")
        actual_lines.append(f"  Matched:      {matched}")
        actual_lines.append(f"  Partial:      {partial}")
        actual_lines.append(f"  Unmatched:    {unmatched}")
        actual_lines.append(f"Total Invoiced: ${total_inv:,.2f}")
        actual_lines.append(f"Total in QBO:   ${total_qbo:,.2f}")
        actual_lines.append(f"Difference:     ${total_inv - total_qbo:,.2f}")

        # Generate unified diff
        diff = unified_diff(
            expected_lines,
            actual_lines,
            fromfile="expected (input)",
            tofile="actual (reconciliation results)",
            lineterm="",
        )

        report_path = output_dir / "reconciliation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write("\n".join(diff))

        return f"Report generated: {report_path}"

    # -- create_reconciliation_excel --
    def create_reconciliation_excel(
        reconciliation_csv_path: str = "output/reconciliation_output.csv",
        output_filename: str = "reconciliation_report.xlsx",
    ) -> str:
        path = _resolve_to_output(reconciliation_csv_path)
        if not path.exists():
            return f"Error: {reconciliation_csv_path} not found. Run save_reconciliation first."

        df = pd.read_csv(path)

        out_path = output_dir / output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wb = xlsxwriter.Workbook(str(out_path))

        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#2E5090", "font_color": "#FFFFFF",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        })
        currency_fmt = wb.add_format({"num_format": "$#,##0.00", "border": 1})
        text_fmt = wb.add_format({"border": 1})
        matched_fmt = wb.add_format({"bg_color": "#E2EFDA", "border": 1})
        partial_fmt = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
        unmatched_fmt = wb.add_format({"bg_color": "#FCE4EC", "border": 1})

        ws = wb.add_worksheet("Reconciliation")
        cols = list(df.columns)
        for c, col in enumerate(cols):
            ws.write(0, c, col, header_fmt)

        for r, (_, row) in enumerate(df.iterrows(), 1):
            status = str(row.get("status", ""))
            if status == "Matched":
                status_cell_fmt = matched_fmt
            elif status == "Partial":
                status_cell_fmt = partial_fmt
            else:
                status_cell_fmt = unmatched_fmt

            ws.write(r, 0, row.get("vendor", ""), text_fmt)
            ws.write(r, 1, row.get("period", ""), text_fmt)
            ws.write(r, 2, row.get("invoice_amount", 0), currency_fmt)
            ws.write(r, 3, row.get("qbo_amount", 0), currency_fmt)
            ws.write(r, 4, status, status_cell_fmt)
            ws.write(r, 5, row.get("notes", ""), text_fmt)

        ws.freeze_panes(1, 0)
        ws.set_column(0, 0, 28)
        ws.set_column(1, 1, 12)
        ws.set_column(2, 3, 18)
        ws.set_column(4, 4, 12)
        ws.set_column(5, 5, 25)

        wb.close()
        return f"Excel report created: {out_path} ({len(df)} rows)"

    # -- Assemble tool list --
    return [
        ToolDef(
            name="save_reconciliation",
            description="Save reconciliation results. You provide matched pairs (vendor, period, invoice_amount, qbo_amount); the tool computes match status and saves CSV.",
            parameters=SAVE_RECONCILIATION_SCHEMA,
            fn=save_reconciliation,
        ),
        ToolDef(
            name="create_reconciliation_excel",
            description="Create a styled Excel reconciliation report with color-coded match status.",
            parameters=CREATE_RECONCILIATION_EXCEL_SCHEMA,
            fn=create_reconciliation_excel,
        ),
    ]
