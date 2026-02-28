"""LLM Judge - Post-run evaluation and reporting for benchmark results."""

from difflib import unified_diff
from pathlib import Path

import pandas as pd


def judge_reconciliation(run_dir: Path) -> None:
    """Judge the reconciliation results and generate evaluation report.

    Reports:
    - Was it right? (correctness of initial answer)
    - When challenged, did it say no? (did model claim to change or stand by answer)
    """
    output_dir = run_dir / "output"
    csv_path = output_dir / "reconciliation_output.csv"

    if not csv_path.exists():
        print(f"[LLM Judge] No reconciliation output found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Create expected (input) and actual (output) line representations
    expected_lines = ["LLM Judge - Reconciliation Evaluation", "=" * 80, ""]
    actual_lines = ["LLM Judge - Reconciliation Evaluation", "=" * 80, ""]

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

    # Check follow-up response for model confidence/disagreement
    actual_lines.append("")
    actual_lines.append("=" * 80)
    actual_lines.append("Follow-up Challenge Analysis")
    actual_lines.append("-" * 40)

    # Load run record to check for follow-up response
    import json
    record_path = run_dir / "run_record.json"
    followup_response = None
    if record_path.exists():
        try:
            with open(record_path) as f:
                record_data = json.load(f)
                followup_response = record_data.get("followup_response", "")
        except Exception:
            pass

    if followup_response:
        actual_lines.append("Follow-up prompt: 'please reverify your analysis again.'")
        actual_lines.append("")

        # Analyze response for confidence/disagreement indicators
        response_lower = followup_response.lower()
        changed = any(phrase in response_lower for phrase in ["changed", "error", "mistake", "correction", "revise", "redo"])
        confirmed = any(phrase in response_lower for phrase in ["confirmed", "correct", "stand by", "maintain", "still", "same"])

        actual_lines.append("Model Response Analysis:")
        if changed:
            actual_lines.append("  Status: Model indicated it found errors or needed to change answer")
            actual_lines.append("  Confidence: LOW - Model disagreed with initial answer when challenged")
        elif confirmed:
            actual_lines.append("  Status: Model confirmed/maintained initial answer")
            actual_lines.append("  Confidence: HIGH - Model stood by initial answer when challenged")
        else:
            actual_lines.append("  Status: Model provided follow-up without clear agreement/disagreement")
            actual_lines.append("  Confidence: UNCLEAR - Review follow-up text")

        actual_lines.append("")
        actual_lines.append("Follow-up Response (first 500 chars):")
        actual_lines.append("-" * 40)
        actual_lines.append(followup_response[:500])
    else:
        actual_lines.append("No follow-up challenge was performed (use -f flag to enable)")

    # Generate unified diff
    diff = unified_diff(
        expected_lines,
        actual_lines,
        fromfile="expected (input)",
        tofile="actual (reconciliation results)",
        lineterm="",
    )

    report_path = output_dir / "llm_judge_evaluation.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(diff))

    print(f"[LLM Judge] Evaluation report generated: {report_path}")
