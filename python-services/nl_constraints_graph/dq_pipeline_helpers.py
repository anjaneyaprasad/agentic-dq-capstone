from __future__ import annotations

import os
import sys
import subprocess
from typing import Tuple

import pandas as pd

from .dq_brain import get_snowflake_connection

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PY_SERVICES_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PY_SERVICES_ROOT, ".."))

SPARK_PROJECT_DIR = os.path.join(PROJECT_ROOT, "dq-spark-project")

DQ_DB = "DQ_DB"
DQ_SCHEMA = "DQ_SCHEMA"


def _run_cmd(cmd: list[str], cwd: str | None = None, timeout: int | None = None) -> str:
    """Run a shell command and capture stdout+stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,   # <-- timeout added here
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired as e:
        return f"[ERROR] Command timed out after {timeout} seconds:\n{cmd}\n\nPartial output:\n{e.stdout or ''}\n{e.stderr or ''}"


def run_spark_jobs(dataset: str) -> str:
    """
    Run ProfilingJob + ValidationJob for the given dataset.
    This is the same logic you use in Streamlit, just without any UI bits.
    """
    ds = dataset.upper()
    logs: list[str] = []

    jar_path = "target/scala-2.12/dq-spark-project-assembly-0.1.0-SNAPSHOT.jar"

    # 1) Build fat JAR
    logs.append("Building Spark fat JAR with sbt assembly...\n")
    logs.append(
        _run_cmd(
            ["sbt", "clean", "assembly"],
            cwd=SPARK_PROJECT_DIR,
        )
    )

    # 2) ProfilingJob
    logs.append(f"\nRunning ProfilingJob for {ds} via spark-submit...\n")
    logs.append(
        _run_cmd(
            [
                "spark-submit",
                "--master",
                "local[4]",
                "--class",
                "com.anjaneya.dq.DqProfilingJob",
                "--packages",
                "net.snowflake:spark-snowflake_2.12:2.16.0-spark_3.4",
                jar_path,
                ds,
            ],
            cwd=SPARK_PROJECT_DIR,
        )
    )

    # 3) ValidationJob
    logs.append(f"\nRunning ValidationJob for {ds} via spark-submit...\n")
    logs.append(
        _run_cmd(
            [
                "spark-submit",
                "--master",
                "local[4]",
                "--class",
                "com.anjaneya.dq.DqValidationJob",
                "--packages",
                "net.snowflake:spark-snowflake_2.12:2.16.0-spark_3.4",
                jar_path,
                ds,
            ],
            cwd=SPARK_PROJECT_DIR,
        )
    )

    return "\n".join(logs)


def load_latest_verification_run(dataset: str) -> Tuple[pd.Timestamp, str, pd.DataFrame]:
    """
    Load the latest run for a dataset that has at least one row
    in DQ_CONSTRAINT_RESULTS.
    Returns (run_ts, run_id, df).
    """
    ds = dataset.upper()
    conn = get_snowflake_connection()
    try:
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

        return run_ts, run_id, df
    finally:
        conn.close()


def load_bad_rows_for_run(dataset: str, run_id: str) -> pd.DataFrame:
    """
    Load bad rows for a given run and dataset from DQ_BAD_ROWS.
    """
    ds = dataset.upper()
    conn = get_snowflake_connection()
    try:
        sql = f"""
            SELECT
              RUN_ID,
              DATASET_NAME,
              RULE_ID,
              PRIMARY_KEY,
              ROW_JSON,
              VIOLATION_MSG,
              CREATED_AT
            FROM {DQ_DB}.{DQ_SCHEMA}.DQ_BAD_ROWS
            WHERE DATASET_NAME = %s
              AND RUN_ID       = %s
            ORDER BY RULE_ID, PRIMARY_KEY
        """
        return pd.read_sql(sql, conn, params=[ds, run_id])
    finally:
        conn.close()
