"""CLI entrypoint for the single-agent eval. No os.chdir anywhere."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from evals.compare import compare_outputs
from evals.config import DEFAULT_MODELS, RunConfig
from evals.llm_judge import judge_reconciliation
from evals.report import print_summary, save_summary
from evals.runner import create_runner
from evals.testcase import get_testcase, list_testcases
from evals.tracing import save_run_trace


def _run_single(config: RunConfig, run_dir: Path) -> dict:
    """Run one iteration. Returns the full record dict."""
    tc = get_testcase(config.testcase)
    tc.set_version(config.version)
    model = config.resolved_model()
    runner = create_runner(config.provider)

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging if debug is off
    if not config.debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Allow DEBUG messages through to handlers
        # Keep console handler at WARNING
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.WARNING)
        file_handler = logging.FileHandler(run_dir / "debug.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    record = runner.run(
        testcase=tc,
        model=model,
        output_dir=output_dir,
        followup=config.followup,
        max_turns=config.max_turns,
        debug=config.debug,
        text=config.text,
    )

    print("\n" + "=" * 60)
    print(record.summary())

    # Verification
    print("\n--- Verification ---")
    verification = tc.verify(output_dir)
    print(json.dumps(verification, indent=2))

    # Follow-up comparison
    followup_comparison = None
    pre_dir = run_dir / "pre_followup"
    if config.followup and pre_dir.exists():
        print("\n--- Follow-up Comparison ---")
        followup_comparison = compare_outputs(pre_dir, output_dir)
        print(json.dumps(followup_comparison, indent=2))

    # Run LLM Judge to evaluate results (after follow-up to compare initial vs challenged answers)
    print("\n--- LLM Judge ---")
    if config.testcase == "reconciliation":
        judge_reconciliation(run_dir)

    # Save traces (run_record.json and trace.jsonl)
    record_dict = save_run_trace(run_dir, record, config, verification, followup_comparison)
    return record_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-agent eval for LLM tool-calling",
        prog="evals",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output (dump file contents to context)")
    parser.add_argument("--text", action="store_true", help="Convert all file inputs to text in prompt instead of uploading")
    parser.add_argument("--testcase", "-t", help="Test case name (e.g. nrr, reconciliation)")
    parser.add_argument("--provider", "-p", choices=["claude", "openai", "both"], help="LLM provider")
    parser.add_argument("--model", "-m", default="", help="Override model name")
    parser.add_argument("--followup", "-f", action="store_true", help="Send follow-up verification prompt")
    parser.add_argument("--iterations", "-n", type=int, default=1, help="Number of iterations (default: 1)")
    parser.add_argument("--max-turns", type=int, default=30, help="Max tool-call turns per phase (default: 30)")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="Data version: v1 or v2 (default: v2)")
    parser.add_argument("--output-root", "-o", default="output", help="Root output directory (default: output)")
    parser.add_argument("--list", action="store_true", help="List available test cases and exit")
    args = parser.parse_args()

    # Configure logging based on debug flag
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.list:
        print("Available test cases:")
        for name in list_testcases():
            print(f"  {name}")
        return

    if not args.testcase or not args.provider:
        parser.error("--testcase and --provider are required (use --list to see available test cases)")

    base_dir = Path.cwd()
    providers = ["claude", "openai"] if args.provider == "both" else [args.provider]

    for provider in providers:
        config = RunConfig(
            testcase=args.testcase,
            provider=provider,
            model=args.model or DEFAULT_MODELS.get(provider, ""),
            followup=args.followup,
            iterations=args.iterations,
            max_turns=args.max_turns,
            output_root=args.output_root,
            debug=args.debug,
            text=args.text,
            version=args.version,
        )

        all_runs: list[dict] = []

        for i in range(config.iterations):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = base_dir / config.output_root / config.testcase / provider / ts

            if config.iterations > 1:
                print(f"\n{'#' * 60}")
                print(f"  Iteration {i + 1}/{config.iterations}")
                print(f"{'#' * 60}")

            record = _run_single(config, run_dir)
            all_runs.append(record)

        if all_runs:
            print_summary(all_runs)
            summary_path = base_dir / config.output_root / config.testcase / provider / "summary_report.json"
            save_summary(all_runs, summary_path)


if __name__ == "__main__":
    main()
