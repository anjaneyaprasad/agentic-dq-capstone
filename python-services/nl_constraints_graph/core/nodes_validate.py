"""
Snowflake-backed dataset/column resolver.

This module is *not* used directly by tests. It just provides the
Snowflake implementations, which the top-level `nl_constraints_graph.nodes_validate`
module delegates to.

Metadata: DQ_DB.DQ_SCHEMA.DQ_DATASETS (OBJECT_NAME)
Data:     physical table pointed to by OBJECT_NAME (e.g. SALES_DQ.PUBLIC.FACT_SALES)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from nl_constraints_graph.dq_brain import get_snowflake_connection


@lru_cache(maxsize=256)
def list_available_datasets_sf() -> List[str]:
    """
    Return active DATASET_NAME values from DQ_DB.DQ_SCHEMA.DQ_DATASETS.
    """
    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DATASET_NAME
                FROM DQ_DB.DQ_SCHEMA.DQ_DATASETS
                WHERE COALESCE(IS_ACTIVE, TRUE)
                ORDER BY DATASET_NAME
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [r[0] for r in rows]


@lru_cache(maxsize=256)
def get_dataset_columns_sf(dataset_name: str) -> List[str]:
    """
    Resolve dataset -> OBJECT_NAME via DQ_DATASETS and then fetch column names
    by doing a zero-row SELECT on the physical table.

    Example row in DQ_DATASETS:
      DATASET_NAME = 'FACT_SALES'
      OBJECT_NAME  = 'SALES_DQ.PUBLIC.FACT_SALES'

    We then run:
      SELECT * FROM SALES_DQ.PUBLIC.FACT_SALES WHERE 1=0
    and read cursor.description to get the column names.
    """
    ds = dataset_name.upper()

    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            # 1) Find the physical object for this dataset
            cur.execute(
                """
                SELECT OBJECT_NAME
                FROM DQ_DB.DQ_SCHEMA.DQ_DATASETS
                WHERE UPPER(DATASET_NAME) = %s
                  AND COALESCE(IS_ACTIVE, TRUE)
                """,
                (ds,),
            )
            row = cur.fetchone()

            if not row:
                raise ValueError(
                    f"No active dataset definition found in DQ_DB.DQ_SCHEMA.DQ_DATASETS "
                    f"for DATASET_NAME='{ds}'."
                )

            object_name = row[0]  # e.g. "SALES_DQ.PUBLIC.FACT_SALES"

            # 2) Zero-row select to introspect columns
            sql_cols = f"SELECT * FROM {object_name} WHERE 1=0"
            cur.execute(sql_cols)

            if not cur.description:
                raise ValueError(
                    f"Could not introspect columns for table '{object_name}'. "
                    "Cursor description is empty."
                )

            columns = [col[0] for col in cur.description]

    finally:
        conn.close()

    if not columns:
        raise ValueError(
            f"No columns found for table '{object_name}'. "
            "Check that the table exists and the Snowflake role has SELECT privilege."
        )

    return columns


if __name__ == "__main__":
    # Handy for manual debugging
    print("Active datasets (from DQ_DATASETS):", list_available_datasets_sf())
    try:
        print("FACT_SALES columns:", get_dataset_columns_sf("FACT_SALES"))
    except Exception as e:
        print("Error:", e)
