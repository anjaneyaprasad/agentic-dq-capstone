import argparse
import os
from datetime import datetime

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Root of the repo: dq-agentic-capstone
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Spark Deequ outputs
SPARK_OUTPUT_DIR = os.path.join(ROOT_DIR, "dq-spark-project", "output")
METRICS_PATH = os.path.join(SPARK_OUTPUT_DIR, "dq_metrics_all")
VERIF_PATH = os.path.join(SPARK_OUTPUT_DIR, "dq_verification_all")

# Templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

def load_latest_run(dataset_name: str) -> datetime:
    """
    Load latest run_ts for given dataset.
    Prefer verification runs (constraints), fall back to metrics if needed.
    """
    verif_df = pd.read_parquet(VERIF_PATH)

    df_verif = verif_df[verif_df["dataset_name"] == dataset_name]
    if not df_verif.empty:
        if not pd.api.types.is_datetime64_any_dtype(df_verif["run_ts"]):
            df_verif["run_ts"] = pd.to_datetime(df_verif["run_ts"])
        latest_ts = df_verif["run_ts"].max()
        return latest_ts

    # fallback to metrics
    metrics_df = pd.read_parquet(METRICS_PATH)
    df_metrics = metrics_df[metrics_df["dataset_name"] == dataset_name]
    if df_metrics.empty:
        raise ValueError(
            f"No metrics or verification found for dataset '{dataset_name}' "
            f"in {METRICS_PATH} or {VERIF_PATH}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df_metrics["run_ts"]):
        df_metrics["run_ts"] = pd.to_datetime(df_metrics["run_ts"])

    latest_ts = df_metrics["run_ts"].max()
    return latest_ts


def load_data_for_run(dataset_name: str, verif_run_ts: datetime):
    """
    Load profiling metrics + verification results for a given dataset.
    We:
      - use verification rows with exact run_ts
      - use metrics from the latest run_ts <= verification run_ts
        (or latest overall if none are <=).
    """
    metrics_df = pd.read_parquet(METRICS_PATH)
    verif_df = pd.read_parquet(VERIF_PATH)

    metrics_df["run_ts"] = pd.to_datetime(metrics_df["run_ts"])
    verif_df["run_ts"] = pd.to_datetime(verif_df["run_ts"])

    # 1) Verification for this run_ts (from load_latest_run)
    verif_run = verif_df[
        (verif_df["dataset_name"] == dataset_name)
        & (verif_df["run_ts"] == verif_run_ts)
    ].copy()

    if verif_run.empty:
        print(f"[WARN] No verification results for dataset '{dataset_name}' and run_ts '{verif_run_ts}'")
        verif_run = pd.DataFrame(
            columns=[
                "dataset_name",
                "run_ts",
                "check",
                "check_level",
                "constraint",
                "constraint_status",
                "constraint_message",
            ]
        )

    # 2) Metrics for this dataset – choose best matching run_ts
    metrics_ds = metrics_df[metrics_df["dataset_name"] == dataset_name].copy()
    if metrics_ds.empty:
        raise ValueError(f"No metrics found for dataset '{dataset_name}' in {METRICS_PATH}")

    # Prefer metrics run_ts <= verification run_ts
    candidates = metrics_ds[metrics_ds["run_ts"] <= verif_run_ts]
    if not candidates.empty:
        metrics_ts = candidates["run_ts"].max()
    else:
        # If all metrics are after verif_run_ts, just use latest available
        metrics_ts = metrics_ds["run_ts"].max()

    metrics_run = metrics_ds[metrics_ds["run_ts"] == metrics_ts].copy()

    if metrics_run.empty:
        raise ValueError(
            f"No metrics rows found for dataset '{dataset_name}' at chosen metrics_ts '{metrics_ts}'"
        )

    print(f"[INFO] Using metrics run_ts {metrics_ts} for verification run_ts {verif_run_ts}")

    return metrics_run, verif_run


def build_context(dataset_name: str, run_ts: datetime, metrics_df: pd.DataFrame, verif_df: pd.DataFrame):
    """
    Build context dict for the Jinja2 template.
    """
    profiling_metrics = [
        {
            "entity": row.get("entity"),
            "name": row.get("name"),
            "instance": row.get("instance"),
            "value": row.get("value"),
        }
        for _, row in metrics_df.iterrows()
    ]

    constraints = [
        {
            "check": row.get("check"),
            "check_level": row.get("check_level"),
            "constraint": row.get("constraint"),
            "constraint_status": row.get("constraint_status"),
            "constraint_message": row.get("constraint_message"),
        }
        for _, row in verif_df.iterrows()
    ]

    total_constraints = len(constraints)
    failed = sum(
        1
        for c in constraints
        if c["constraint_status"] not in (None, "", "Success")
    )
    passed = total_constraints - failed

    summary = {
        "total_constraints": total_constraints,
        "passed": passed,
        "failed": failed,
    }

    context = {
        "dataset_name": dataset_name,
        "run_ts": run_ts,
        "profiling_metrics": profiling_metrics,
        "constraints": constraints,
        "summary": summary,
    }
    return context


def render_report(context, output_path: str):
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )

    template = env.get_template("report.html")
    html = template.render(**context)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build DQ HTML report from Deequ outputs.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match dataset_name used in Deequ jobs)",
    )
    parser.add_argument(
        "--run-ts",
        required=False,
        help="Optional run timestamp (ISO format). If omitted, latest verification run is used.",
    )
    parser.add_argument(
        "--out-dir",
        required=False,
        default=os.path.join(ROOT_DIR, "reports"),
        help="Base output directory for HTML reports.",
    )

    args = parser.parse_args()

    dataset_name = args.dataset

    if args.run_ts:
        run_ts = pd.to_datetime(args.run_ts)
    else:
        run_ts = load_latest_run(dataset_name)
        print(f"[INFO] Using latest run_ts for {dataset_name}: {run_ts}")

    metrics_df, verif_df = load_data_for_run(dataset_name, run_ts)

    context = build_context(dataset_name, run_ts, metrics_df, verif_df)

    safe_ts = run_ts.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.out_dir, dataset_name)
    output_path = os.path.join(output_dir, f"dq_report_{safe_ts}.html")

    # Write timestamped report
    render_report(context, output_path)

    # Also keep a 'latest.html' for convenience
    latest_path = os.path.join(output_dir, "latest.html")
    render_report(context, latest_path)


if __name__ == "__main__":
    main()
