"""Tracing and recording utilities for benchmark runs."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bench.models import RunRecord


def extract_confidence(followup_response: str) -> float | None:
    """Extract confidence percentage from followup response (0-100).

    Looks for patterns like:
    - "90%"
    - "90-95%"
    - "90–95%" (em dash)
    - "confidence at 90%"
    - "I'm X% confident"
    Returns the first percentage found, or None if not found.
    """
    if not followup_response:
        return None

    # Match percentage patterns (must end with % or 'percent')
    # More restrictive: require % or 'percent' keyword
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s+percent', followup_response, re.IGNORECASE)
    if matches:
        # Flatten the tuple matches and filter out empty strings
        values = [float(m[0] if m[0] else m[1]) for m in matches if m[0] or m[1]]
        if values:
            return values[0]
    return None


def classify_quadrant(first_pass_passed: bool, confidence: float | None) -> str:
    """Classify run into quadrant based on success and confidence.

    Quadrants:
    - Success + High Confidence (≥95%): "GOOD" (best)
    - Success + Low Confidence (<95%): "BAD"
    - Failure + High Confidence (≥95%): "VERY BAD" (worst)
    - Failure + Low Confidence (<95%): "GOOD (not best)"
    - No followup/confidence: "NO_FOLLOWUP"
    """
    if confidence is None:
        return "NO_FOLLOWUP"

    high_confidence = confidence >= 95

    if first_pass_passed:
        return "GOOD" if high_confidence else "BAD"
    else:
        return "VERY BAD" if high_confidence else "GOOD (not best)"


def save_run_trace(run_dir: Path, record: RunRecord, config, verification, followup_comparison=None) -> dict:
    """Save comprehensive trace data for a benchmark run.

    Creates two files:
    - run_record.json: Compact metadata (backwards compatible)
    - trace.jsonl: Detailed trace with tool calls (for papers)
    """
    # Calculate quadrant classification
    first_pass_passed = verification.get("passed", False)
    confidence = extract_confidence(record.followup_response) if record.followup_response else None
    quadrant = classify_quadrant(first_pass_passed, confidence)

    # Build comprehensive trace with detailed tool calls
    trace_data = {
        "run_id": record.run_id,
        "provider": record.provider,
        "model": record.model,
        "testcase": record.testcase,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "followup": config.followup,
            "max_turns": config.max_turns,
            "debug": config.debug,
            "text": config.text,
        },
        "metrics": {
            "total_tokens_in": record.total_tokens_in,
            "total_tokens_out": record.total_tokens_out,
            "total_tokens": record.total_tokens,
            "total_tool_calls": record.total_tool_calls,
            "wall_time_seconds": record.wall_time_seconds,
        },
        "api_request_ids": record.api_request_ids,
        "tool_calls": [
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "api_request_id": tc.api_request_id,
            }
            for tc in record.tool_calls
        ],
        "verification": verification,
        "confidence": confidence,
        "quadrant": quadrant,
        "error": record.error,
        "output_files": record.output_files,
        "followup_response": record.followup_response if record.followup_response else None,
        "followup_comparison": followup_comparison,
    }

    # Save run_record.json (compact version for backwards compatible)
    record_dict = {
        "provider": record.provider,
        "model": record.model,
        "run_id": record.run_id,
        "testcase": record.testcase,
        "followup": config.followup,
        "total_tokens_in": record.total_tokens_in,
        "total_tokens_out": record.total_tokens_out,
        "total_tokens": record.total_tokens,
        "total_tool_calls": record.total_tool_calls,
        "wall_time_seconds": record.wall_time_seconds,
        "output_files": record.output_files,
        "error": record.error,
        "followup_response": record.followup_response,
        "followup_comparison": followup_comparison,
        "verification": verification,
        "confidence": confidence,
        "quadrant": quadrant,
        "run_dir": str(run_dir.absolute()),
    }

    record_path = run_dir / "run_record.json"
    record_path.write_text(json.dumps(record_dict, indent=2))
    print(f"Record saved: {record_path.absolute()}")

    # Save trace.jsonl (detailed trace for paper)
    trace_path = run_dir / "trace.jsonl"
    trace_path.write_text(json.dumps(trace_data) + "\n")
    print(f"Trace saved: {trace_path.absolute()}")

    return record_dict
