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


# """
# Snowflake-backed dataset/column resolver + validator node for the NL → YAML agent.

# - Metadata: DQ_DB.DQ_SCHEMA.DQ_DATASETS (OBJECT_NAME)
# - Data:     physical table pointed to by OBJECT_NAME (e.g. SALES_DQ.PUBLIC.FACT_SALES)

# We avoid INFORMATION_SCHEMA and instead:
#   SELECT * FROM <OBJECT_NAME> WHERE 1=0
# and use cursor.description to get column names.

# This version is GraphState-aware (LangGraph passes a GraphState, not a dict).
# """

# from __future__ import annotations

# import os
# from functools import lru_cache
# from typing import List

# import snowflake.connector

# from nl_constraints_graph.models import GraphState  # <- Pydantic state model
# from nl_constraints_graph.dq_brain import get_snowflake_connection

# SUPPORTED_RULE_TYPES: list[str] = [
#     "COMPLETENESS",
#     "UNIQUENESS",
#     "IS_UNIQUE",
#     "RANGE",
#     "PATTERN",
#     "REGEX",
# ]


# # -------------------------------------------------------------------
# # Snowflake connection helper (self-contained)
# # -------------------------------------------------------------------


# # -------------------------------------------------------------------
# # Dataset registry helpers (from DQ_DATASETS)
# # -------------------------------------------------------------------

# @lru_cache(maxsize=256)
# def list_available_datasets() -> List[str]:
#     """
#     Return active DATASET_NAME values from DQ_DB.DQ_SCHEMA.DQ_DATASETS.
#     """
#     sql = """
#         SELECT DATASET_NAME
#         FROM DQ_DB.DQ_SCHEMA.DQ_DATASETS
#         WHERE COALESCE(IS_ACTIVE, TRUE)
#         ORDER BY DATASET_NAME
#     """

#     conn = get_snowflake_connection()
#     try:
#         with conn.cursor() as cur:
#             cur.execute(sql)
#             rows = cur.fetchall()
#     finally:
#         conn.close()

#     return [r[0] for r in rows]


# def dataset_exists(dataset: str) -> bool:
#     return dataset.upper() in {d.upper() for d in list_available_datasets()}


# # -------------------------------------------------------------------
# # Column metadata via OBJECT_NAME + cursor.description
# # -------------------------------------------------------------------

# @lru_cache(maxsize=256)
# def get_dataset_columns(dataset_name: str) -> List[str]:
#     """
#     Resolve dataset -> OBJECT_NAME via DQ_DATASETS and then fetch column names
#     by doing a zero-row SELECT on the physical table.

#     Example row in DQ_DATASETS:
#       DATASET_NAME = 'FACT_SALES'
#       OBJECT_NAME  = 'SALES_DQ.PUBLIC.FACT_SALES'

#     We then run:
#       SELECT * FROM SALES_DQ.PUBLIC.FACT_SALES WHERE 1=0
#     and read cursor.description to get the column names.
#     """
#     ds = dataset_name.upper()

#     conn = get_snowflake_connection()
#     try:
#         with conn.cursor() as cur:
#             # 1) Find the physical object for this dataset
#             cur.execute(
#                 """
#                 SELECT OBJECT_NAME
#                 FROM DQ_DB.DQ_SCHEMA.DQ_DATASETS
#                 WHERE UPPER(DATASET_NAME) = %s
#                   AND COALESCE(IS_ACTIVE, TRUE)
#                 """,
#                 (ds,),
#             )
#             row = cur.fetchone()

#             if not row:
#                 raise ValueError(
#                     f"No active dataset definition found in DQ_DB.DQ_SCHEMA.DQ_DATASETS "
#                     f"for DATASET_NAME='{ds}'."
#                 )

#             object_name = row[0]  # e.g. "SALES_DQ.PUBLIC.FACT_SALES"

#             # 2) Zero-row select to introspect columns
#             sql_cols = f"SELECT * FROM {object_name} WHERE 1=0"
#             cur.execute(sql_cols)

#             if not cur.description:
#                 raise ValueError(
#                     f"Could not introspect columns for table '{object_name}'. "
#                     "Cursor description is empty."
#                 )

#             columns = [col[0] for col in cur.description]

#     finally:
#         conn.close()

#     if not columns:
#         raise ValueError(
#             f"No columns found for table '{object_name}'. "
#             "Check that the table exists and the Snowflake role has SELECT privilege."
#         )

#     return columns

# # -------------------------------------------------------------------
# # Rule validation helpers (used by tests)
# # -------------------------------------------------------------------

# def _extract_rule_column(rule: Any) -> str | None:
#     """
#     Tests use DummyRule(column=..., type_=...)
#     Real rules might use columnName / column.
#     """
#     return (
#         getattr(rule, "column", None)
#         or getattr(rule, "columnName", None)
#     )


# def _extract_rule_type(rule: Any) -> str | None:
#     """
#     Tests use DummyRule(type_=...)
#     Real rules might use type / rule_type / ruleType.
#     """
#     return (
#         getattr(rule, "type_", None)
#         or getattr(rule, "rule_type", None)
#         or getattr(rule, "ruleType", None)
#         or getattr(rule, "type", None)
#     )


# def validate_rules(dataset: str, rules: Sequence[Any]) -> Tuple[bool, List[str]]:
#     """
#     Validate that:
#       - rule type is supported (in SUPPORTED_RULE_TYPES)
#       - referenced column exists in the dataset

#     Returns (ok, messages).

#     tests/nl_constraints_graph/test_nodes_validate.py calls this directly.
#     """
#     ok = True
#     messages: List[str] = []

#     dataset_cols = {c.upper() for c in get_dataset_columns(dataset)}

#     for r in rules:
#         col = _extract_rule_column(r)
#         r_type = _extract_rule_type(r)

#         # Normalize type for comparison
#         r_type_norm = (r_type or "").upper()

#         if not r_type_norm or r_type_norm not in SUPPORTED_RULE_TYPES:
#             ok = False
#             messages.append(f"Unsupported rule type: {r_type} for column {col}")
#             # continue, but still check columns

#         if col and col.upper() not in dataset_cols:
#             ok = False
#             messages.append(
#                 f"Column '{col}' not found in dataset '{dataset}'. "
#                 f"Available columns: {sorted(dataset_cols)}"
#             )

#     return ok, messages


# def detect_anomalies(dataset: str) -> List[str]:
#     """
#     Stub anomaly detector – tests monkeypatch this.

#     In production you might use profiling metrics from DQ_PROFILING_METRICS
#     or other signals. For tests, they override this with a lambda.
#     """
#     return []


# # -------------------------------------------------------------------
# # LangGraph validator node (GraphState-aware)
# # -------------------------------------------------------------------

# def validator_node(state: GraphState) -> GraphState:
#     """
#     Validates inferred rules + columns and attaches messages / flags
#     to the GraphState.

#     tests expect, at minimum:
#       - when there are no rules, validation_messages contains
#         "No rules inferred to validate."
#       - detect_anomalies(...) is called (they monkeypatch it)
#     """
#     request = state.request
#     dataset = (request.dataset or "").upper()

#     # Inferred rules from earlier graph steps
#     rules = list(getattr(state, "inferred_rules", []) or [])

#     # Existing messages (if any)
#     messages: List[str] = list(getattr(state, "validation_messages", []) or [])
#     anomaly_messages: List[str] = list(getattr(state, "anomaly_messages", []) or [])

#     if not rules:
#         messages.append("No rules inferred to validate.")
#         # Run anomaly detection anyway
#         anomaly_messages.extend(detect_anomalies(dataset))

#         state.validation_messages = messages
#         state.anomaly_messages = anomaly_messages
#         # If GraphState defines validation_ok, set it to False
#         if hasattr(state, "validation_ok"):
#             state.validation_ok = False  # type: ignore[attr-defined]
#         return state

#     # We have some rules -> validate them against dataset metadata
#     ok, rule_messages = validate_rules(dataset, rules)
#     messages.extend(rule_messages)

#     # Attach anomaly info
#     anomaly_messages.extend(detect_anomalies(dataset))

#     state.validation_messages = messages
#     state.anomaly_messages = anomaly_messages
#     if hasattr(state, "validation_ok"):
#         state.validation_ok = ok  # type: ignore[attr-defined]

#     return state

# # def validator_node(state: GraphState) -> GraphState:
# #     """
# #     LangGraph node used by graph_nl_to_yaml:

# #     - Ensures dataset exists in DQ_DATASETS
# #     - Ensures any referenced columns exist on the physical table
# #     - Returns the same GraphState (we already populate `columns` in Streamlit)
# #     """
# #     request = state.request
# #     dataset = (request.dataset or "").upper()
# #     requested_columns = state.columns or []

# #     if not dataset:
# #         raise ValueError("No dataset name provided in request.")

# #     if not dataset_exists(dataset):
# #         raise ValueError(
# #             f"Dataset '{dataset}' is not registered or not active in DQ_DATASETS."
# #         )

# #     actual_columns = set(get_dataset_columns(dataset))
# #     missing = [c for c in requested_columns if c and c.upper() not in actual_columns]

# #     if missing:
# #         raise ValueError(
# #             f"Column(s) not found in dataset '{dataset}': {missing}. "
# #             f"Available columns: {sorted(actual_columns)}"
# #         )

# #     # IMPORTANT: do NOT add new attributes to GraphState here.
# #     # Just return the original state after validation.
# #     return state


# # -------------------------------------------------------------------
# # Debug usage
# # -------------------------------------------------------------------

# if __name__ == "__main__":
#     print("Available datasets:", list_available_datasets())
#     try:
#         print("FACT_SALES columns:", get_dataset_columns("FACT_SALES"))
#     except Exception as e:
#         print("Error:", e)
