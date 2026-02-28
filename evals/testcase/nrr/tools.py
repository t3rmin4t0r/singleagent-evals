"""NRR-specific tools — configurable MRR/NRR computation engine.

The LLM reads the data, understands the billing structure, and decides HOW to
calculate MRR by choosing enum parameters. The tool executes the chosen config.

All tools are created via make_tools() which captures data_dir and output_dir
in closures — no os.chdir() needed.
"""

import json
from calendar import monthrange
from datetime import datetime
from pathlib import Path

import pandas as pd
import xlsxwriter

from evals.config import MAX_TOOL_RESULT_LENGTH
from evals.models import ToolDef
from evals.testcase.shared import read_excel_overview, read_pdf_text, resolve_path

# Valid enum values for calculate_mrr parameters
VALID_EVENT_TYPES = {"invoice", "charge", "credit", "refund"}
VALID_CURRENCIES = {"USD", "EUR", "ALL"}
VALID_REVENUE_SOURCE = {"billing_events", "subscriptions"}
VALID_BILLING_FREQ_FILTER = {"all", "annual_only", "monthly_only"}
VALID_DEDUP_FIELD = {"none", "subscription_id", "contract_id"}
VALID_DEDUP_KEEP = {"longest_period", "shortest_period", "highest_amount", "lowest_amount", "latest_bill_date"}
VALID_PRORATION = {"daily", "monthly_equal", "full_in_bill_month"}
VALID_CREDIT_METHOD = {"subtract_in_credit_month", "prorate_over_service_period", "ignore"}


# ---------------------------------------------------------------------------
# Internal computation helpers (no I/O path assumptions)
# ---------------------------------------------------------------------------

def _validate_enum(value: str, valid_set: set[str], param_name: str) -> str | None:
    if value not in valid_set:
        return f"Error: {param_name} must be one of {sorted(valid_set)}, got '{value}'"
    return None


def _dedup_events(df: pd.DataFrame, field: str, keep: str) -> pd.DataFrame:
    """Deduplicate overlapping billing events within each group."""
    groups = []
    for _, group in df.groupby(field):
        if len(group) <= 1:
            groups.append(group)
            continue

        group = group.sort_values("service_start_date").reset_index(drop=True)

        # Find clusters of overlapping service periods
        clusters: list[list[int]] = []
        cluster = [0]
        for i in range(1, len(group)):
            prev_end = group.iloc[cluster[-1]]["service_end_date"]
            curr_start = group.iloc[i]["service_start_date"]
            if curr_start <= prev_end:
                cluster.append(i)
            else:
                clusters.append(cluster)
                cluster = [i]
        clusters.append(cluster)

        kept = []
        for cluster in clusters:
            if len(cluster) == 1:
                kept.append(group.iloc[cluster[0]])
                continue

            candidates = group.iloc[cluster]
            if keep == "longest_period":
                durations = (candidates["service_end_date"] - candidates["service_start_date"]).dt.days
                winner = durations.idxmax()
            elif keep == "shortest_period":
                durations = (candidates["service_end_date"] - candidates["service_start_date"]).dt.days
                winner = durations.idxmin()
            elif keep == "highest_amount":
                winner = candidates["amount"].idxmax()
            elif keep == "lowest_amount":
                winner = candidates["amount"].idxmin()
            elif keep == "latest_bill_date":
                winner = candidates["bill_date"].idxmax()
            else:
                winner = cluster[0]
            kept.append(group.loc[winner])

        groups.append(pd.DataFrame(kept))

    return pd.concat(groups, ignore_index=True)


def _mrr_from_subscriptions(
    contracts: pd.DataFrame,
    subs: pd.DataFrame,
    start_month: str,
    end_month: str,
    billing_freq_filter: str,
    proration: str,
) -> list[dict]:
    """Compute MRR from contracts + subscriptions tables."""
    filtered_subs = subs.copy()
    if billing_freq_filter == "annual_only":
        filtered_subs = filtered_subs[filtered_subs["billing_freq"] == "annual"]
    elif billing_freq_filter == "monthly_only":
        filtered_subs = filtered_subs[filtered_subs["billing_freq"] == "monthly"]

    contract_acv = filtered_subs.groupby("contract_id").apply(
        lambda x: (x["unit_price"] * x["quantity"]).sum(), include_groups=False
    ).to_dict()

    rows = []
    y, m = map(int, start_month.split("-"))
    while True:
        ym = f"{y:04d}-{m:02d}"
        if ym > end_month:
            break

        days_in_month = monthrange(y, m)[1]
        month_start = pd.Timestamp(datetime(y, m, 1))
        month_end = pd.Timestamp(datetime(y, m, days_in_month))

        active = contracts[
            (contracts["start_date"] <= month_end) & (contracts["end_date"] >= month_start)
        ]

        for _, c in active.iterrows():
            acv = contract_acv.get(c["contract_id"], 0)
            if acv <= 0:
                continue

            cust = c["customer_id"]

            if proration == "daily":
                start_in_month = max(c["start_date"], month_start)
                end_in_month = min(c["end_date"], month_end)
                days_active = (end_in_month - start_in_month).days + 1
                mrr = acv * days_active / 365
            elif proration == "monthly_equal":
                mrr = acv / 12
            elif proration == "full_in_bill_month":
                mrr = acv
            else:
                mrr = acv * days_in_month / 365

            rows.append({"customer_id": cust, "month": ym, "mrr": round(mrr, 2)})

        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

    return rows


def _mrr_from_billing_events(
    path: Path,
    contracts: pd.DataFrame,
    subs: pd.DataFrame,
    start_month: str,
    end_month: str,
    event_types: list[str],
    currency: str,
    billing_freq_filter: str,
    dedup_field: str,
    dedup_keep: str,
    proration: str,
) -> list[dict]:
    """Compute MRR from billing_events table."""
    billing = pd.read_excel(path, sheet_name="billing_events")

    for col in ["bill_date", "service_start_date", "service_end_date"]:
        billing[col] = pd.to_datetime(billing[col])
    billing["amount"] = pd.to_numeric(billing["amount"], errors="coerce")

    merged = billing.merge(contracts[["contract_id", "customer_id"]], on="contract_id", how="left")
    merged = merged.merge(subs[["subscription_id", "billing_freq"]], on="subscription_id", how="left")

    filtered = merged[merged["event_type"].isin(event_types)].copy()

    if currency != "ALL":
        filtered = filtered[filtered["currency"] == currency]

    if billing_freq_filter == "annual_only":
        filtered = filtered[filtered["billing_freq"] == "annual"]
    elif billing_freq_filter == "monthly_only":
        filtered = filtered[filtered["billing_freq"] == "monthly"]

    if dedup_field != "none":
        filtered = _dedup_events(filtered, dedup_field, dedup_keep)

    rows = []
    for _, row in filtered.iterrows():
        svc_start = row["service_start_date"]
        svc_end = row["service_end_date"]
        amount = row["amount"]
        cust = row["customer_id"]

        if pd.isna(svc_start) or pd.isna(svc_end) or pd.isna(amount) or amount <= 0:
            continue

        if proration == "full_in_bill_month":
            ym = row["bill_date"].strftime("%Y-%m")
            if start_month <= ym <= end_month:
                rows.append({"customer_id": cust, "month": ym, "mrr": round(amount, 2)})
            continue

        total_days = (svc_end - svc_start).days + 1
        if total_days <= 0:
            continue

        if proration == "monthly_equal":
            months_in_period = 0
            cur = datetime(svc_start.year, svc_start.month, 1)
            while cur <= svc_end:
                months_in_period += 1
                cur = datetime(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
            per_month = amount / max(months_in_period, 1)

        current = datetime(svc_start.year, svc_start.month, 1)
        while current <= svc_end:
            ym = current.strftime("%Y-%m")
            if start_month <= ym <= end_month:
                if proration == "daily":
                    ms = max(svc_start, pd.Timestamp(current))
                    me_day = monthrange(current.year, current.month)[1]
                    me = min(svc_end, pd.Timestamp(datetime(current.year, current.month, me_day)))
                    days = (me - ms).days + 1
                    if days > 0:
                        prorated = amount * (days / total_days)
                        rows.append({"customer_id": cust, "month": ym, "mrr": round(prorated, 2)})
                elif proration == "monthly_equal":
                    rows.append({"customer_id": cust, "month": ym, "mrr": round(per_month, 2)})

            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

    return rows


# ---------------------------------------------------------------------------
# Tool factory — captures data_dir and output_dir in closures
# ---------------------------------------------------------------------------

# JSON Schemas (module-level constants)
READ_SPREADSHEET_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to .xlsx file"},
        "sheet_name": {"type": "string", "description": "Sheet name. Empty = overview of all sheets.", "default": ""},
    },
    "required": ["file_path"],
}

READ_PDF_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to PDF file"},
    },
    "required": ["file_path"],
}

GET_BILLING_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to .xlsx file with billing_events sheet"},
    },
    "required": ["file_path"],
}

CALCULATE_MRR_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to .xlsx file"},
        "start_month": {"type": "string", "description": "Start month YYYY-MM (e.g. '2024-01')"},
        "end_month": {"type": "string", "description": "End month YYYY-MM inclusive (e.g. '2025-07')"},
        "event_types": {
            "type": "array",
            "items": {"type": "string", "enum": ["invoice", "charge", "credit", "refund"]},
            "description": "Billing event types to include.",
        },
        "currency": {"type": "string", "enum": ["USD", "EUR", "ALL"], "description": "Currency filter."},
        "revenue_source": {
            "type": "string",
            "enum": ["billing_events", "subscriptions"],
            "description": "billing_events uses billed amounts; subscriptions uses contracted prices with contract active periods.",
        },
        "billing_freq_filter": {"type": "string", "enum": ["all", "annual_only", "monthly_only"]},
        "dedup_field": {"type": "string", "enum": ["none", "subscription_id", "contract_id"]},
        "dedup_keep": {
            "type": "string",
            "enum": ["longest_period", "shortest_period", "highest_amount", "lowest_amount", "latest_bill_date"],
        },
        "proration": {"type": "string", "enum": ["daily", "monthly_equal", "full_in_bill_month"]},
        "credit_method": {
            "type": "string",
            "enum": ["subtract_in_credit_month", "prorate_over_service_period", "ignore"],
        },
    },
    "required": [
        "file_path", "start_month", "end_month",
        "event_types", "currency", "revenue_source", "billing_freq_filter",
        "dedup_field", "dedup_keep", "proration", "credit_method",
    ],
}

CALCULATE_NRR_SCHEMA = {
    "type": "object",
    "properties": {
        "mrr_csv_path": {
            "type": "string",
            "description": "Path to MRR CSV from calculate_mrr. Default: output/mrr_output.csv",
            "default": "output/mrr_output.csv",
        },
    },
    "required": [],
}

CREATE_STYLED_EXCEL_SCHEMA = {
    "type": "object",
    "properties": {
        "mrr_csv_path": {"type": "string", "default": "output/mrr_output.csv"},
        "nrr_csv_path": {"type": "string", "default": "output/nrr_output.csv"},
        "output_filename": {"type": "string", "default": "nrr_analysis.xlsx", "description": "Output filename"},
    },
    "required": [],
}

CREATE_ECHART_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "nrr_csv_path": {"type": "string", "default": "output/nrr_output.csv"},
        "title": {"type": "string", "default": "Net Revenue Retention (NRR)"},
        "output_filename": {"type": "string", "default": "nrr_chart.json"},
    },
    "required": [],
}


def make_tools(data_dir: Path, output_dir: Path) -> list[ToolDef]:
    """Create NRR tool definitions with data_dir/output_dir captured in closures."""

    def _resolve_to_output(path_str: str) -> Path:
        """Resolve paths that the LLM provides for output files (e.g. 'output/mrr_output.csv')."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        # If it starts with 'output/', strip that prefix and resolve relative to output_dir
        parts = p.parts
        if parts and parts[0] == "output":
            return output_dir / Path(*parts[1:])
        return output_dir / p

    # -- read_spreadsheet --
    def read_spreadsheet(file_path: str, sheet_name: str = "") -> str:
        path = resolve_path(file_path, data_dir)
        if not path.exists():
            return f"Error: file not found: {file_path}"
        if not sheet_name:
            return read_excel_overview(path)
        df = pd.read_excel(path, sheet_name=sheet_name)
        return f"Sheet: {sheet_name} | {len(df)} rows x {len(df.columns)} cols\n{df.to_csv(index=False)}"[:MAX_TOOL_RESULT_LENGTH]

    # -- read_pdf --
    def read_pdf(file_path: str) -> str:
        return read_pdf_text(file_path, data_dir, max_chars=MAX_TOOL_RESULT_LENGTH)

    # -- get_billing_summary --
    def get_billing_summary(file_path: str) -> str:
        path = resolve_path(file_path, data_dir)
        if not path.exists():
            return f"Error: file not found: {file_path}"

        billing = pd.read_excel(path, sheet_name="billing_events")
        contracts = pd.read_excel(path, sheet_name="contracts")
        subs = pd.read_excel(path, sheet_name="subscriptions")

        billing["bill_date"] = pd.to_datetime(billing["bill_date"])
        billing["service_start_date"] = pd.to_datetime(billing["service_start_date"])
        billing["service_end_date"] = pd.to_datetime(billing["service_end_date"])
        billing["amount"] = pd.to_numeric(billing["amount"], errors="coerce")
        contracts["start_date"] = pd.to_datetime(contracts["start_date"])
        contracts["end_date"] = pd.to_datetime(contracts["end_date"])

        merged = billing.merge(contracts[["contract_id", "customer_id"]], on="contract_id", how="left")

        lines = [
            f"Billing events: {len(billing)}",
            f"Contracts: {len(contracts)}",
            f"Subscriptions: {len(subs)}",
            f"Unique customers: {merged['customer_id'].nunique()}",
            f"Bill date range: {billing['bill_date'].min().date()} to {billing['bill_date'].max().date()}",
            f"Service period range: {billing['service_start_date'].min().date()} to {billing['service_end_date'].max().date()}",
            f"Contract period range: {contracts['start_date'].min().date()} to {contracts['end_date'].max().date()}",
            f"Total billed amount: ${billing['amount'].sum():,.2f}",
            f"Total subscription ACV: ${(subs['unit_price'] * subs['quantity']).sum():,.2f}",
            f"Event types: {dict(billing['event_type'].value_counts())}",
            f"Currencies: {dict(billing['currency'].value_counts())}",
            f"Amount range: ${billing['amount'].min():,.2f} to ${billing['amount'].max():,.2f}",
        ]

        if "is_prebill" in billing.columns:
            prebill_count = (billing["is_prebill"] == "Y").sum()
            lines.append(f"Pre-billed events: {prebill_count}/{len(billing)} ({100*prebill_count/len(billing):.0f}%)")

        events_per_sub = billing.groupby("subscription_id").size()
        lines.append(f"Billing events per subscription: min={events_per_sub.min()}, median={events_per_sub.median():.0f}, max={events_per_sub.max()}")

        return "\n".join(lines)

    # -- calculate_mrr --
    def calculate_mrr(
        file_path: str,
        start_month: str,
        end_month: str,
        event_types: list[str],
        currency: str,
        revenue_source: str,
        billing_freq_filter: str,
        dedup_field: str,
        dedup_keep: str,
        proration: str,
        credit_method: str,
    ) -> str:
        # Validate enums
        for evt in event_types:
            err = _validate_enum(evt, VALID_EVENT_TYPES, "event_types")
            if err:
                return err
        for param, val, valid in [
            ("currency", currency, VALID_CURRENCIES),
            ("revenue_source", revenue_source, VALID_REVENUE_SOURCE),
            ("billing_freq_filter", billing_freq_filter, VALID_BILLING_FREQ_FILTER),
            ("dedup_field", dedup_field, VALID_DEDUP_FIELD),
            ("dedup_keep", dedup_keep, VALID_DEDUP_KEEP),
            ("proration", proration, VALID_PRORATION),
            ("credit_method", credit_method, VALID_CREDIT_METHOD),
        ]:
            err = _validate_enum(val, valid, param)
            if err:
                return err

        path = resolve_path(file_path, data_dir)
        if not path.exists():
            return f"Error: file not found: {file_path}"

        contracts = pd.read_excel(path, sheet_name="contracts")
        customers = pd.read_excel(path, sheet_name="customers")
        subs = pd.read_excel(path, sheet_name="subscriptions")
        credits_df = pd.read_excel(path, sheet_name="credits_adjustments")

        contracts["start_date"] = pd.to_datetime(contracts["start_date"])
        contracts["end_date"] = pd.to_datetime(contracts["end_date"])
        credits_df["credit_date"] = pd.to_datetime(credits_df["credit_date"])
        credits_df["amount"] = pd.to_numeric(credits_df["amount"], errors="coerce")

        config_summary = (
            f"Config: events={event_types}, currency={currency}, source={revenue_source}, "
            f"freq={billing_freq_filter}, dedup={dedup_field}/{dedup_keep}, "
            f"proration={proration}, credits={credit_method}"
        )

        start_dt = datetime.strptime(start_month, "%Y-%m")
        end_year, end_mo = map(int, end_month.split("-"))
        end_dt = datetime(end_year, end_mo, monthrange(end_year, end_mo)[1])

        if revenue_source == "subscriptions":
            rows = _mrr_from_subscriptions(contracts, subs, start_month, end_month, billing_freq_filter, proration)
        else:
            rows = _mrr_from_billing_events(
                path, contracts, subs, start_month, end_month,
                event_types, currency, billing_freq_filter, dedup_field, dedup_keep, proration,
            )

        mrr_df = pd.DataFrame(rows)
        if mrr_df.empty:
            return f"Error: no MRR data produced. Check parameters.\n{config_summary}"

        mrr_agg = mrr_df.groupby(["customer_id", "month"], as_index=False)["mrr"].sum()

        # Credits
        credits_applied = 0
        if credit_method != "ignore":
            credits_merged = credits_df.merge(contracts[["contract_id", "customer_id"]], on="contract_id", how="left")
            credits_in_range = credits_merged[
                (credits_merged["credit_date"] >= start_dt) & (credits_merged["credit_date"] <= end_dt)
            ].copy()

            if credit_method == "subtract_in_credit_month":
                credits_in_range["month"] = credits_in_range["credit_date"].dt.strftime("%Y-%m")
                credits_by_cm = credits_in_range.groupby(["customer_id", "month"], as_index=False)["amount"].sum()
                credits_by_cm.rename(columns={"amount": "credit"}, inplace=True)
                mrr_agg = mrr_agg.merge(credits_by_cm, on=["customer_id", "month"], how="left")
                mrr_agg["credit"] = mrr_agg["credit"].fillna(0)
                mrr_agg["mrr"] = mrr_agg["mrr"] + mrr_agg["credit"]
                mrr_agg.drop(columns=["credit"], inplace=True)
            elif credit_method == "prorate_over_service_period":
                for _, cr in credits_in_range.iterrows():
                    cust = cr["customer_id"]
                    cust_months = mrr_agg[mrr_agg["customer_id"] == cust]["month"].unique()
                    if len(cust_months) > 0:
                        per_month_credit = cr["amount"] / len(cust_months)
                        mask = mrr_agg["customer_id"] == cust
                        mrr_agg.loc[mask, "mrr"] = mrr_agg.loc[mask, "mrr"] + per_month_credit

            credits_applied = len(credits_in_range)

        mrr_agg = mrr_agg.merge(customers[["customer_id", "customer_name"]], on="customer_id", how="left")
        mrr_agg = mrr_agg.sort_values(["month", "customer_id"]).reset_index(drop=True)

        # Save to output_dir (captured in closure)
        out_path = output_dir / "mrr_output.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mrr_agg.to_csv(out_path, index=False)

        monthly_totals = mrr_agg.groupby("month")["mrr"].sum()
        summary_lines = [
            config_summary,
            f"MRR calculated: {len(mrr_agg)} rows, {mrr_agg['customer_id'].nunique()} customers, "
            f"{mrr_agg['month'].nunique()} months",
            f"Credits applied: {credits_applied}",
            f"Saved to: {out_path}",
            "",
            "Monthly total MRR:",
        ]
        for m, total in monthly_totals.items():
            summary_lines.append(f"  {m}: ${total:,.2f}")

        return "\n".join(summary_lines)[:MAX_TOOL_RESULT_LENGTH]

    # -- calculate_nrr --
    def calculate_nrr(mrr_csv_path: str = "output/mrr_output.csv") -> str:
        path = _resolve_to_output(mrr_csv_path)
        if not path.exists():
            return f"Error: file not found: {mrr_csv_path}. Run calculate_mrr first."

        mrr_df = pd.read_csv(path)
        months = sorted(mrr_df["month"].unique())

        if len(months) < 2:
            return "Error: need at least 2 months of MRR data for NRR calculation."

        results = []
        for i in range(1, len(months)):
            prev_month = months[i - 1]
            curr_month = months[i]

            prev = mrr_df[mrr_df["month"] == prev_month]
            curr = mrr_df[mrr_df["month"] == curr_month]

            cohort = set(prev["customer_id"])
            cohort_prev_mrr = prev[prev["customer_id"].isin(cohort)]["mrr"].sum()
            cohort_curr = curr[curr["customer_id"].isin(cohort)]
            cohort_curr_mrr = cohort_curr["mrr"].sum()

            expansion = 0.0
            contraction = 0.0
            churn = 0.0

            for cust in cohort:
                prev_val = prev[prev["customer_id"] == cust]["mrr"].sum()
                curr_val = cohort_curr[cohort_curr["customer_id"] == cust]["mrr"].sum()
                diff = curr_val - prev_val
                if curr_val == 0:
                    churn += prev_val
                elif diff > 0:
                    expansion += diff
                elif diff < 0:
                    contraction += abs(diff)

            nrr = cohort_curr_mrr / cohort_prev_mrr if cohort_prev_mrr > 0 else 0

            results.append({
                "month": curr_month,
                "baseline": round(cohort_prev_mrr, 2),
                "expansion": round(expansion, 2),
                "contraction": round(contraction, 2),
                "churn": round(churn, 2),
                "nrr": round(nrr, 4),
            })

        nrr_df = pd.DataFrame(results)
        out_path = output_dir / "nrr_output.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nrr_df.to_csv(out_path, index=False)

        lines = [f"NRR calculated: {len(nrr_df)} months", f"Saved to: {out_path}", ""]
        for _, row in nrr_df.iterrows():
            lines.append(
                f"  {row['month']}: NRR={row['nrr']:.4f}  "
                f"baseline=${row['baseline']:,.0f}  expansion=${row['expansion']:,.0f}  "
                f"contraction=${row['contraction']:,.0f}  churn=${row['churn']:,.0f}"
            )

        return "\n".join(lines)[:MAX_TOOL_RESULT_LENGTH]

    # -- create_styled_excel --
    def create_styled_excel(
        mrr_csv_path: str = "output/mrr_output.csv",
        nrr_csv_path: str = "output/nrr_output.csv",
        output_filename: str = "nrr_analysis.xlsx",
    ) -> str:
        mrr_path = _resolve_to_output(mrr_csv_path)
        nrr_path = _resolve_to_output(nrr_csv_path)

        if not mrr_path.exists():
            return f"Error: {mrr_csv_path} not found. Run calculate_mrr first."
        if not nrr_path.exists():
            return f"Error: {nrr_csv_path} not found. Run calculate_nrr first."

        mrr_df = pd.read_csv(mrr_path)
        nrr_df = pd.read_csv(nrr_path)

        out_path = output_dir / output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wb = xlsxwriter.Workbook(str(out_path))

        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#3583FC", "font_color": "#FFFFFF",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        })
        currency_fmt = wb.add_format({"num_format": "$#,##0.00"})
        pct_fmt = wb.add_format({"num_format": "0.0000"})

        # MRR sheet
        ws_mrr = wb.add_worksheet("MRR")
        cols = ["customer_id", "customer_name", "month", "mrr"]
        for c, col in enumerate(cols):
            ws_mrr.write(0, c, col, header_fmt)
        for r, (_, row) in enumerate(mrr_df.iterrows(), 1):
            ws_mrr.write(r, 0, row.get("customer_id", ""))
            ws_mrr.write(r, 1, row.get("customer_name", ""))
            ws_mrr.write(r, 2, row.get("month", ""))
            ws_mrr.write(r, 3, row.get("mrr", 0), currency_fmt)
        ws_mrr.freeze_panes(1, 0)
        ws_mrr.set_column(0, 0, 14)
        ws_mrr.set_column(1, 1, 20)
        ws_mrr.set_column(2, 2, 12)
        ws_mrr.set_column(3, 3, 15)

        # NRR sheet
        ws_nrr = wb.add_worksheet("NRR")
        nrr_cols = ["month", "baseline", "expansion", "contraction", "churn", "nrr"]
        for c, col in enumerate(nrr_cols):
            ws_nrr.write(0, c, col, header_fmt)
        for r, (_, row) in enumerate(nrr_df.iterrows(), 1):
            ws_nrr.write(r, 0, row.get("month", ""))
            ws_nrr.write(r, 1, row.get("baseline", 0), currency_fmt)
            ws_nrr.write(r, 2, row.get("expansion", 0), currency_fmt)
            ws_nrr.write(r, 3, row.get("contraction", 0), currency_fmt)
            ws_nrr.write(r, 4, row.get("churn", 0), currency_fmt)
            ws_nrr.write(r, 5, row.get("nrr", 0), pct_fmt)
        ws_nrr.freeze_panes(1, 0)
        ws_nrr.set_column(0, 0, 12)
        ws_nrr.set_column(1, 4, 15)
        ws_nrr.set_column(5, 5, 10)

        wb.close()
        return f"Excel created: {out_path} ({len(mrr_df)} MRR rows, {len(nrr_df)} NRR rows)"

    # -- create_echart_json --
    def create_echart_json(
        nrr_csv_path: str = "output/nrr_output.csv",
        title: str = "Net Revenue Retention (NRR)",
        output_filename: str = "nrr_chart.json",
    ) -> str:
        path = _resolve_to_output(nrr_csv_path)
        if not path.exists():
            return f"Error: {nrr_csv_path} not found. Run calculate_nrr first."

        nrr_df = pd.read_csv(path)

        config = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 18}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "legend": {"data": ["NRR"], "bottom": 10},
            "grid": {"left": "10%", "right": "10%", "bottom": "15%"},
            "xAxis": {
                "type": "category",
                "data": nrr_df["month"].tolist(),
                "axisLabel": {"rotate": 45},
            },
            "yAxis": {"type": "value", "name": "NRR", "axisLabel": {"formatter": "{value}"}},
            "series": [
                {
                    "name": "NRR",
                    "type": "line",
                    "smooth": True,
                    "data": [round(v, 4) for v in nrr_df["nrr"].tolist()],
                    "markLine": {"data": [{"yAxis": 1.0, "name": "100%"}]},
                    "symbol": "circle",
                    "symbolSize": 6,
                }
            ],
        }

        out_path = output_dir / output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(config, indent=2))

        return f"ECharts JSON created: {out_path} ({len(nrr_df)} data points)"

    # -- Assemble tool list --
    return [
        ToolDef(
            name="read_spreadsheet",
            description="Read an Excel spreadsheet. Returns sheet list + preview, or full sheet data.",
            parameters=READ_SPREADSHEET_SCHEMA,
            fn=read_spreadsheet,
        ),
        ToolDef(
            name="read_pdf",
            description="Read a PDF and return its text content.",
            parameters=READ_PDF_SCHEMA,
            fn=read_pdf,
        ),
        ToolDef(
            name="get_billing_summary",
            description="Get summary statistics of billing data: date ranges, amounts, event types.",
            parameters=GET_BILLING_SUMMARY_SCHEMA,
            fn=get_billing_summary,
        ),
        ToolDef(
            name="calculate_mrr",
            description="Calculate Monthly Recurring Revenue with proration across service periods.",
            parameters=CALCULATE_MRR_SCHEMA,
            fn=calculate_mrr,
        ),
        ToolDef(
            name="calculate_nrr",
            description="Calculate cohort-based Net Revenue Retention from MRR data.",
            parameters=CALCULATE_NRR_SCHEMA,
            fn=calculate_nrr,
        ),
        ToolDef(
            name="create_styled_excel",
            description="Create a styled Excel file with MRR and NRR sheets.",
            parameters=CREATE_STYLED_EXCEL_SCHEMA,
            fn=create_styled_excel,
        ),
        ToolDef(
            name="create_echart_json",
            description="Create an ECharts JSON chart configuration for NRR visualization.",
            parameters=CREATE_ECHART_JSON_SCHEMA,
            fn=create_echart_json,
        ),
    ]
