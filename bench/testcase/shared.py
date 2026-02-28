"""Shared utilities for test case tools."""

from pathlib import Path

import fitz
import pandas as pd

from bench.config import MAX_TOOL_RESULT_LENGTH


def resolve_path(file_path: str, data_dir: Path) -> Path:
    """Resolve a file path relative to data_dir with fallback to filename-only.

    Handles LLM providing paths like "data/input.xlsx" or just "input.xlsx".
    """
    p = Path(file_path)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to data_dir
    candidate = data_dir / p
    if candidate.exists():
        return candidate
    # Fallback: just the filename in data_dir
    fallback = data_dir / p.name
    if fallback.exists():
        return fallback
    return candidate  # Return the first attempt for error messaging


def read_pdf_text(file_path: str, data_dir: Path, max_chars: int = MAX_TOOL_RESULT_LENGTH) -> str:
    """Read a PDF and return its text content, truncated to max_chars."""
    path = resolve_path(file_path, data_dir)
    if not path.exists():
        return f"Error: file not found: {file_path}"
    doc = fitz.open(str(path))
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text[:max_chars]


def read_excel_overview(path: Path, max_chars: int = MAX_TOOL_RESULT_LENGTH) -> str:
    """Read Excel file: sheet names, row counts, and first 5 rows per sheet.

    Single read per sheet (no double read for row count).
    """
    xls = pd.ExcelFile(path)
    parts = [f"Sheets: {xls.sheet_names}"]
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        parts.append(f"\n--- {sheet_name} ({len(df)} rows, {df.shape[1]} cols) ---")
        parts.append(df.head(5).to_csv(index=False))
    return "\n".join(parts)[:max_chars]
