"""NRR test case — prompt, tools, verification."""

import json
from pathlib import Path

import pandas as pd

from bench.models import ToolDef
from bench.testcase.base import TestCase
from bench.testcase.nrr.tools import make_tools

SYSTEM_PROMPT = """\
You are a financial data analyst. You have tools to read data files and compute MRR/NRR.

The calculate_mrr tool is a configurable engine — you must choose all parameters:
event types, currency, revenue source, billing frequency filter, deduplication strategy, \
proration method, and credit handling. Read the data carefully before deciding. \
The billing data may have overlapping events, mixed frequencies, and other complexities \
that require you to choose the right configuration.

Always read and understand the data before calling calculate_mrr. Wrong parameters \
will produce wrong results.

IMPORTANT: After reading the billing summary, compare the total billed amount against \
the total subscription ACV. If the billed amount is significantly larger, billing events \
likely contain pre-billed duplicates — use revenue_source=subscriptions instead.
"""

TASK_PROMPT = """\
Analyze the billing data in input.xlsx and the product pricing in pricing.pdf.

Generate monthly MRR and Net Revenue Retention (NRR) for the full date range in the data.

Produce:
1. A styled Excel file with MRR and NRR sheets
2. An ECharts JSON chart showing the NRR trend over time

Start by reading the data files to understand the billing structure, then calculate.
"""


class NRRTestCase(TestCase):

    @property
    def name(self) -> str:
        return "nrr"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def task_prompt(self) -> str:
        return TASK_PROMPT

    def get_tools(self, output_dir: Path) -> list[ToolDef]:
        return make_tools(self.data_dir, output_dir)

    def verify(self, output_dir: Path) -> dict:
        """Verify NRR outputs against golden baselines."""
        results: dict = {"structural": {}, "numerical": {}}

        # Structural checks
        excel_files = list(output_dir.glob("*.xlsx"))
        results["structural"]["excel_exists"] = len(excel_files) > 0

        chart_files = list(output_dir.glob("*.json"))
        results["structural"]["chart_exists"] = len(chart_files) > 0

        if chart_files:
            try:
                with open(chart_files[0]) as f:
                    chart = json.load(f)
                results["structural"]["chart_valid_json"] = True
                results["structural"]["chart_has_series"] = "series" in chart
            except (json.JSONDecodeError, OSError):
                results["structural"]["chart_valid_json"] = False

        # Numerical: MRR comparison
        mrr_path = output_dir / "mrr_output.csv"
        golden_mrr_path = self.data_dir / "golden_mrr.csv"

        if mrr_path.exists() and golden_mrr_path.exists():
            mrr = pd.read_csv(mrr_path)
            golden = pd.read_csv(golden_mrr_path)

            results["numerical"]["mrr_rows"] = len(mrr)
            results["numerical"]["golden_mrr_rows"] = len(golden)

            mrr_totals = mrr.groupby("month")["mrr"].sum()
            golden_totals = golden.groupby("month")["mrr"].sum()

            common_months = set(mrr_totals.index) & set(golden_totals.index)
            if common_months:
                diffs = []
                for m in sorted(common_months):
                    diff_pct = abs(mrr_totals[m] - golden_totals[m]) / golden_totals[m] if golden_totals[m] != 0 else 0
                    diffs.append(diff_pct)
                results["numerical"]["mrr_avg_diff_pct"] = round(sum(diffs) / len(diffs) * 100, 2)
                results["numerical"]["mrr_max_diff_pct"] = round(max(diffs) * 100, 2)
                results["numerical"]["mrr_months_matched"] = len(common_months)

        # Numerical: NRR comparison
        nrr_path = output_dir / "nrr_output.csv"
        golden_nrr_path = self.data_dir / "golden_nrr.csv"

        if nrr_path.exists() and golden_nrr_path.exists():
            nrr = pd.read_csv(nrr_path)
            golden_nrr = pd.read_csv(golden_nrr_path)

            results["numerical"]["nrr_months"] = len(nrr)
            results["numerical"]["golden_nrr_months"] = len(golden_nrr)

            nrr_by_month = dict(zip(nrr["month"], nrr["nrr"]))
            golden_by_month = dict(zip(golden_nrr["month"], golden_nrr["nrr"]))

            common = set(nrr_by_month) & set(golden_by_month)
            if common:
                nrr_diffs = []
                for m in sorted(common):
                    diff = abs(nrr_by_month[m] - golden_by_month[m])
                    nrr_diffs.append(diff)
                results["numerical"]["nrr_avg_abs_diff"] = round(sum(nrr_diffs) / len(nrr_diffs), 4)
                results["numerical"]["nrr_max_abs_diff"] = round(max(nrr_diffs), 4)
                results["numerical"]["nrr_within_5pct"] = all(d < 0.05 for d in nrr_diffs)

        # Top-level pass/score — binary: all NRR diffs < 5% or fail
        num = results["numerical"]
        if num.get("nrr_within_5pct"):
            results["passed"] = True
            results["score"] = "PASS"
        elif "nrr_max_abs_diff" in num:
            results["passed"] = False
            results["score"] = f"FAIL (nrr_max_diff={num['nrr_max_abs_diff']}, mrr_avg_diff={num.get('mrr_avg_diff_pct', '?')}%)"
        elif "mrr_avg_diff_pct" in num:
            results["passed"] = False
            results["score"] = f"FAIL (mrr_avg_diff={num['mrr_avg_diff_pct']}%, no NRR output)"
        else:
            results["passed"] = False
            results["score"] = "FAIL (no output)"

        return results
