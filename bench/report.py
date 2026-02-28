"""Summary reporting — console table and JSON export."""

import json
from pathlib import Path


def run_result(run: dict) -> str:
    """Determine PASS/FAIL/ERROR/DONE for a single run."""
    if run.get("error"):
        return "ERROR"
    fc = run.get("followup_comparison")
    if fc and fc.get("match") != "100%":
        return "FAIL"
    v = run.get("verification", {})
    if "passed" not in v:
        return "DONE"
    return "PASS" if v["passed"] else "FAIL"


def _followup_status(fc: dict | None) -> str:
    if fc is None:
        return "-"
    return fc.get("match", "CHANGED")


def print_summary(all_runs: list[dict]) -> None:
    """Print a clean summary report across all iterations."""
    if not all_runs:
        return

    first = all_runs[0]
    n = len(all_runs)

    print(f"\n{'=' * 80}")
    print(f"  {first['testcase']}  |  {first['provider']} / {first['model']}  |  {n} run(s)  |  followup={'yes' if first.get('followup') else 'no'}")
    print(f"{'=' * 80}")

    print(f"\n  {'#':<4} {'Time':>7} {'Calls':>6} {'Tokens':>10} {'Score':>10} {'Followup':>10} {'Result':>8}")
    print(f"  {'-' * 65}")

    for i, run in enumerate(all_runs, 1):
        score = run.get("verification", {}).get("score", "-")
        result = run_result(run)
        fu = _followup_status(run.get("followup_comparison"))
        print(f"  {i:<4} {run['wall_time_seconds']:>6.1f}s {run['total_tool_calls']:>6} {run['total_tokens']:>10,} {score:>10} {fu:>10} {result:>8}")

    avg_time = sum(r["wall_time_seconds"] for r in all_runs) / n
    avg_calls = sum(r["total_tool_calls"] for r in all_runs) / n
    avg_tokens = sum(r["total_tokens"] for r in all_runs) / n
    pass_count = sum(1 for r in all_runs if run_result(r) == "PASS")
    print(f"  {'-' * 65}")
    print(f"  {'AVG':<4} {avg_time:>6.1f}s {avg_calls:>6.1f} {avg_tokens:>10,.0f} {'':>10} {'':>10} {pass_count}/{n} pass")

    # Verification details
    print(f"\n  Verification:")
    for i, run in enumerate(all_runs, 1):
        num = run.get("verification", {}).get("numerical", {})
        if num:
            parts = [f"{k}={v}" for k, v in num.items()]
            print(f"    Run {i}: {', '.join(parts)}")

    # Follow-up details
    comparisons = [(i + 1, r["followup_comparison"]) for i, r in enumerate(all_runs) if r.get("followup_comparison")]
    if comparisons:
        print(f"\n  Followup comparison:")
        for num, fc in comparisons:
            print(f"    Run {num}: {fc.get('match', '?')} ({fc.get('pct', '?')}%) — {fc.get('explanation', '')}")

    print()


def save_summary(all_runs: list[dict], path: Path) -> None:
    """Save summary report as JSON."""
    if not all_runs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "testcase": all_runs[0]["testcase"],
        "provider": all_runs[0]["provider"],
        "iterations": len(all_runs),
        "runs": all_runs,
    }, indent=2))
    print(f"Summary saved: {path.absolute()}")
