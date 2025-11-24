# python-services/nl_constraints_graph/anomaly_utils.py

from __future__ import annotations
import os
from typing import List

import pandas as pd

# Adjust this if your folder layout changes
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METRICS_DIR = os.path.join(PROJECT_ROOT, "output", "dq_metrics_all")


def detect_anomalies(
    dataset: str,
    metric_name: str = "completeness",
    drop_threshold: float = 0.1,
) -> List[str]:
    """
    Very simple anomaly detector:
      - Reads dq_metrics_all parquet
      - For each column, looks at time series of a given metric_name
      - If latest value dropped by >= drop_threshold compared to previous run,
        emit a human-readable anomaly message.

    Returns:
        List of strings (messages).
    """
    if not os.path.exists(METRICS_DIR):
        return []

    df = pd.read_parquet(METRICS_DIR)
    if "dataset_name" not in df.columns:
        # Adjust this if your schema differs
        return []

    df_ds = df[df["dataset_name"] == dataset].copy()
    if df_ds.empty:
        return []

    # Ensure we have the required columns
    required_cols = {"column", "metric_name", "metric_value", "run_ts"}
    if not required_cols.issubset(set(df_ds.columns)):
        return []

    messages: List[str] = []

    for col in sorted(df_ds["column"].unique()):
        sub = df_ds[(df_ds["column"] == col) & (df_ds["metric_name"] == metric_name)]
        if sub.empty:
            continue

        sub = sub.sort_values("run_ts")
        if len(sub) < 2:
            # Need at least 2 runs to see a drop
            continue

        prev = float(sub.iloc[-2]["metric_value"])
        latest = float(sub.iloc[-1]["metric_value"])

        if prev > 0 and (prev - latest) >= drop_threshold:
            messages.append(
                f"Anomaly: {metric_name} for '{col}' dropped from {prev:.3f} to {latest:.3f} "
                f"between the last two runs."
            )

    return messages
