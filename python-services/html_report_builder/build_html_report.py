import os
import argparse
from pathlib import Path
from typing import Tuple
import math

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

# Load .env so os.getenv works reliably
load_dotenv()

# DQ metadata lives here (actual data is in SALES_DQ)
DQ_DB = "DQ_DB"
DQ_SCHEMA = "DQ_SCHEMA"


# ---------------- Snowflake connection ---------------- #

def get_snowflake_connection():
    """
    Create a Snowflake connection using credentials from environment variables.

    Expected env vars (from .env or shell):
      - SNOWFLAKE_USER
      - SNOWFLAKE_PASSWORD
      - SNOWFLAKE_ACCOUNT
      - SNOWFLAKE_WAREHOUSE
      - SNOWFLAKE_ROLE (optional)
    """
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        role=os.getenv("SNOWFLAKE_ROLE", None),
        database=DQ_DB,
        schema=DQ_SCHEMA,
    )


# ---------------- Load latest run + results ---------------- #

def _safe_int(value) -> int:
    """
    Safely convert a value coming from Snowflake → pandas into an int.

    - Returns 0 for NULL / NaN / unconvertible values.
    - This is convenient for counts like TOTAL_ROWS, FAILED_ROWS.
    """
    if value is None:
        return 0
    try:
        if isinstance(value, float) and math.isnan(value):
            return 0
        return int(value)
    except Exception:
        return 0


def load_latest_run(dataset_name: str) -> Tuple[pd.Timestamp, pd.DataFrame]:
    """
    Load the latest DQ run for a dataset that has at least one row
    in DQ_CONSTRAINT_RESULTS.

    Returns:
        (run_ts, df)
        - run_ts: STARTED_AT of the run (pd.Timestamp)
        - df: dataframe with joined run + constraint result rows
    """
    ds = dataset_name.upper()
    conn = get_snowflake_connection()

    try:
        # 1) Find the latest run for this dataset that actually has constraint results
        sql_run = f"""
            WITH runs_with_results AS (
              SELECT
                r.RUN_ID,
                r.DATASET_NAME,
                r.RUN_TYPE,
                r.STARTED_AT,
                r.FINISHED_AT,
                r.STATUS,
                r.TOTAL_ROWS,
                r.FAILED_ROWS,
                r.MESSAGE,
                ROW_NUMBER() OVER (
                  PARTITION BY r.DATASET_NAME
                  ORDER BY r.STARTED_AT DESC
                ) AS rn
              FROM {DQ_DB}.{DQ_SCHEMA}.DQ_RUNS r
              JOIN {DQ_DB}.{DQ_SCHEMA}.DQ_CONSTRAINT_RESULTS c
                ON c.RUN_ID = r.RUN_ID
              WHERE r.DATASET_NAME = %s
            )
            SELECT *
            FROM runs_with_results
            WHERE rn = 1
        """
        runs_df = pd.read_sql(sql_run, conn, params=[ds])

        if runs_df.empty:
            raise ValueError(f"No runs with constraint results found for dataset {ds}")

        run_id = runs_df.loc[0, "RUN_ID"]
        run_ts = runs_df.loc[0, "STARTED_AT"]

        # 2) Fetch all constraint results for that run
        sql_verif = f"""
            SELECT
                r.DATASET_NAME,
                r.RUN_ID,
                r.RUN_TYPE,
                r.STARTED_AT,
                r.FINISHED_AT,
                r.STATUS       AS RUN_STATUS,
                r.TOTAL_ROWS,
                r.FAILED_ROWS,
                r.MESSAGE      AS RUN_MESSAGE,
                c.RULE_ID,
                c.RULE_TYPE,
                c.COLUMN_NAME,
                c.STATUS       AS CONSTRAINT_STATUS,
                c.MESSAGE      AS CONSTRAINT_MESSAGE,
                c.METRIC_NAME,
                c.METRIC_VALUE
            FROM {DQ_DB}.{DQ_SCHEMA}.DQ_RUNS r
            JOIN {DQ_DB}.{DQ_SCHEMA}.DQ_CONSTRAINT_RESULTS c
              ON c.RUN_ID = r.RUN_ID
            WHERE r.RUN_ID = %s
            ORDER BY c.RULE_ID, c.COLUMN_NAME
        """
        df = pd.read_sql(sql_verif, conn, params=[run_id])

        if df.empty:
            raise ValueError(f"No constraint results found for run_id={run_id} / dataset={ds}")

        return run_ts, df

    finally:
        conn.close()


# ---------------- HTML builder ---------------- #

def build_html(dataset: str, run_ts, df: pd.DataFrame) -> str:
    """
    Build a simple HTML report for a single dataset and a single run.
    """
    dataset = dataset.upper()

    total_rows = _safe_int(df["TOTAL_ROWS"].iloc[0]) if "TOTAL_ROWS" in df.columns else None
    failed_rows = _safe_int(df["FAILED_ROWS"].iloc[0]) if "FAILED_ROWS" in df.columns else None
    passed_rows = (total_rows - failed_rows) if total_rows is not None and failed_rows is not None else None

    total_checks = len(df)
    failed_checks = (df["CONSTRAINT_STATUS"] != "Success").sum()
    passed_checks = total_checks - failed_checks

    failed_df = df[df["CONSTRAINT_STATUS"] != "Success"].copy()

    preferred_cols = [
        "RULE_TYPE",
        "COLUMN_NAME",
        "CONSTRAINT_STATUS",
        "CONSTRAINT_MESSAGE",
        "METRIC_NAME",
        "METRIC_VALUE",
    ]
    available_cols = [c for c in preferred_cols if c in failed_df.columns]

    if not failed_df.empty and available_cols:
        failed_html_table = failed_df[available_cols].to_html(
            index=False, escape=True, classes="dq-table"
        )
    elif failed_df.empty:
        failed_html_table = "<p>No failed constraints 🎉</p>"
    else:
        failed_html_table = failed_df.to_html(index=False, escape=True, classes="dq-table")

    # Build optional metrics cleanly
    extra_metric_rows = ""

    if total_rows is not None:
        extra_metric_rows += f"""
        <div class="metric">
            <div class="metric-label">Rows scanned</div>
            <div class="metric-value">{total_rows}</div>
        </div>
        """

    if failed_rows is not None:
        extra_metric_rows += f"""
        <div class="metric">
            <div class="metric-label">Rows with violations</div>
            <div class="metric-value status-failure">{failed_rows}</div>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Data Quality Report – {dataset}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 20px;
      line-height: 1.5;
    }}
    h1, h2, h3 {{
      color: #222;
    }}
    .overview-card {{
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 24px;
      background: #fafafa;
    }}
    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 8px 12px;
      border-radius: 6px;
      background: #fff;
      border: 1px solid #eee;
    }}
    .metric-label {{
      font-size: 12px;
      text-transform: uppercase;
      color: #666;
    }}
    .metric-value {{
      font-size: 18px;
      font-weight: 600;
    }}
    .dq-table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 8px;
    }}
    .dq-table th, .dq-table td {{
      border: 1px solid #ddd;
      padding: 6px 8px;
      font-size: 13px;
    }}
    .dq-table th {{
      background: #f3f3f3;
      text-align: left;
    }}
    .status-failure {{
      color: #b3261e;
      font-weight: 600;
    }}
  </style>
</head>
<body>

  <h1>Data Quality Report – {dataset}</h1>

  <div class="overview-card">
    <h2>Overview</h2>
    <p><strong>Run timestamp:</strong> {run_ts}</p>
    <div class="overview-grid">
      <div class="metric">
        <div class="metric-label">Total checks</div>
        <div class="metric-value">{total_checks}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Passed checks</div>
        <div class="metric-value">{passed_checks}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Failed checks</div>
        <div class="metric-value status-failure">{failed_checks}</div>
      </div>
      {extra_metric_rows}
    </div>
  </div>

  <h2>Failed Constraints</h2>
  {failed_html_table}

</body>
</html>
    """

    return html



# ---------------- main() ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. fact_sales)")
    args = parser.parse_args()

    dataset = args.dataset
    ds_upper = dataset.upper()

    print(f"Building HTML report for {dataset}...")
    print(f"Loading latest verification run from Snowflake for dataset '{ds_upper}'...")

    run_ts, df = load_latest_run(dataset)
    print(f"Loaded {len(df)} rows for latest run_ts={run_ts}")

    html = build_html(dataset, run_ts, df)

    # Write to reports/<dataset>/latest.html under project root
    project_root = Path(__file__).resolve().parents[2]
    report_dir = project_root / "reports" / dataset.lower()
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "latest.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote HTML report to: {output_path}")
    print()
    print(f"Expected latest report path: {output_path}")


if __name__ == "__main__":
    main()
