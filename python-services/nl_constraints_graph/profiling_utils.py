# python-services/nl_constraints_graph/profiling_utils.py

from __future__ import annotations
import os
from typing import Dict, Any

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METRICS_DIR = os.path.join(PROJECT_ROOT, "output", "dq_metrics_all")


def load_latest_profile(dataset: str) -> Dict[str, Dict[str, Any]] | None:
    """
    Build a column->metrics dict from the latest profiling runs.

    Returns:
        {
          "amount": {
            "completeness": 0.98,
            "min": 0,
            "max": 12000,
            ...
          },
          ...
        }
    """
    if not os.path.exists(METRICS_DIR):
        return None

    df = pd.read_parquet(METRICS_DIR)
    if "dataset_name" not in df.columns:
        return None

    df_ds = df[df["dataset_name"] == dataset].copy()
    if df_ds.empty:
        return None

    # assume: run_ts, column, metric_name, metric_value
    required_cols = {"column", "metric_name", "metric_value", "run_ts"}
    if not required_cols.issubset(set(df_ds.columns)):
        return None

    # Only use latest run_ts
    latest_ts = df_ds["run_ts"].max()
    df_latest = df_ds[df_ds["run_ts"] == latest_ts]

    summary: Dict[str, Dict[str, Any]] = {}
    for col in sorted(df_latest["column"].unique()):
        sub = df_latest[df_latest["column"] == col]
        metrics: Dict[str, Any] = {}
        for _, row in sub.iterrows():
            metrics[row["metric_name"]] = row["metric_value"]
        summary[col] = metrics

    return summary