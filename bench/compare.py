"""Cell-level CSV comparison for follow-up verification."""

from pathlib import Path

import pandas as pd


def compare_outputs(pre_dir: Path, post_dir: Path) -> dict:
    """Compare pre-followup and post-followup CSVs at cell level.

    Returns: {"match": "100%|partial|no", "pct": int, "explanation": str}
    """
    pre_csvs = sorted(pre_dir.glob("*.csv"))
    post_csvs = {f.name: f for f in post_dir.rglob("*.csv")}

    if not pre_csvs:
        return {"match": "100%", "pct": 100, "explanation": "No CSV output to compare."}

    total_cells = 0
    matching_cells = 0
    diffs: list[str] = []

    for pre_file in pre_csvs:
        post_file = post_csvs.get(pre_file.name)
        if not post_file:
            diffs.append(f"{pre_file.name}: missing after followup")
            continue

        try:
            pre_df = pd.read_csv(pre_file)
            post_df = pd.read_csv(post_file)
        except Exception:
            diffs.append(f"{pre_file.name}: could not parse")
            continue

        if pre_df.shape != post_df.shape:
            diffs.append(f"{pre_file.name}: shape {pre_df.shape} -> {post_df.shape}")
            # Count cells from the larger shape as total, smaller overlap as partial
            total_cells += max(pre_df.size, post_df.size)
            matching_cells += min(pre_df.size, post_df.size) // 2
            continue

        if list(pre_df.columns) != list(post_df.columns):
            diffs.append(f"{pre_file.name}: columns changed")
            total_cells += pre_df.size
            continue

        # Cell-level comparison
        file_cells = pre_df.shape[0] * pre_df.shape[1]
        total_cells += file_cells
        matched = 0
        for col in pre_df.columns:
            if pd.api.types.is_numeric_dtype(pre_df[col]) and pd.api.types.is_numeric_dtype(post_df[col]):
                close = (pre_df[col].fillna(0) - post_df[col].fillna(0)).abs() < 0.01
                matched += close.sum()
            else:
                matched += (pre_df[col].astype(str) == post_df[col].astype(str)).sum()
        matching_cells += matched

    if total_cells == 0:
        return {"match": "100%", "pct": 100, "explanation": "No data cells to compare."}

    pct = int(100 * matching_cells / total_cells)
    if pct == 100 and not diffs:
        return {"match": "100%", "pct": 100, "explanation": "All output files identical."}
    elif pct >= 90:
        return {"match": "partial", "pct": pct, "explanation": "; ".join(diffs) if diffs else f"{pct}% of cells match."}
    else:
        return {"match": "no", "pct": pct, "explanation": "; ".join(diffs) if diffs else f"Only {pct}% of cells match."}
