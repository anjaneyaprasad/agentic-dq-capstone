from __future__ import annotations

import json
import uuid
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from nl_constraints_graph.models import GraphState, NLRequest
from nl_constraints_graph.graph_nl_to_yaml import build_graph
from nl_constraints_graph.nodes_validate import get_dataset_columns
from nl_constraints_graph.dq_brain import get_snowflake_connection

# ---------------------------------------------------------------------
# Path / env bootstrap
# ---------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from nl_constraints_graph.dq_brain import (
    get_snowflake_connection,
    load_latest_profiling,
    summarize_profiling,
)

from nl_constraints_graph.streamlit_app import (
    run_spark_jobs,
    load_latest_verification_run,
    load_bad_rows_for_run,
)

# OpenAI client for NL -> SQL
from openai import OpenAI

mcp = FastMCP("dq_mcp_server")
openai_client = OpenAI()

RULE_TYPE_ALIASES = {
    "unique": "uniqueness",
    "uniqueness": "uniqueness",
    "completeness": "completeness",
    "completeness_threshold": "completeness",
    "non_negative": "non_negative",
    "domain": "domain",
    "size_greater_than": "size_greater_than",
    "min_value": "min_value",
    "max_value": "max_value",
}

# ---------------------------------------------------------------------
# NL -> SQL prompts & few-shot examples for DQ_DB.DQ_SCHEMA
# ---------------------------------------------------------------------

DQ_NL_SQL_SYSTEM_PROMPT = """
You are a Snowflake SQL generation agent for a Data Quality (DQ) platform.

Your ONLY job: given a natural language question, generate a SINGLE Snowflake SELECT
query over the DQ metadata tables that answers the question.

STRICT RULES
------------
- You MUST ONLY query tables in database DQ_DB, schema DQ_SCHEMA.
- You MUST NOT reference any other database or schema (NO SALES_DQ, SALES_DB, PUBLIC, etc.).
- You MUST NOT query raw data tables such as FACT tables or source datasets.
- When the user asks about uniqueness, duplicates, null checks, ranges, thresholds,
  rule failures, bad rows, or profiling, interpret it as a **metadata lookup**
  in DQ_DB.DQ_SCHEMA (RULES, RUNS, CONSTRAINT_RESULTS, BAD_ROWS, PROFILING_METRICS),
  NOT as a query on raw fact tables.
- Output ONLY valid Snowflake SQL.
- No explanations, no comments, no markdown fences. Just the SQL text.
- The query MUST start with SELECT and MUST end with a semicolon.
- NEVER generate DDL/DML (no CREATE, ALTER, DROP, INSERT, UPDATE, DELETE, MERGE, TRUNCATE).


DQ METADATA SCHEMA (IMPORTANT)
------------------------------
All DQ metadata lives in database DQ_DB, schema DQ_SCHEMA:

1) DQ_DB.DQ_SCHEMA.DQ_DATASETS
   - DATASET_NAME (PK)
   - SOURCE_TYPE
   - CONNECTION_NAME
   - OBJECT_NAME        -- physical table/view name in source
   - DESCRIPTION
   - IS_ACTIVE
   - CREATED_BY
   - CREATED_AT
   - UPDATED_AT
   - PRIMARY_KEY_COLUMN

2) DQ_DB.DQ_SCHEMA.DQ_RULESETS
   - RULESET_ID (PK)
   - DATASET_NAME (FK -> DQ_DATASETS.DATASET_NAME)
   - VERSION
   - IS_ACTIVE
   - PROMPT_TEXT
   - DESCRIPTION
   - CREATED_BY
   - CREATED_AT
   - UNIQUE(DATASET_NAME, VERSION)

3) DQ_DB.DQ_SCHEMA.DQ_RULES
   - RULE_ID (PK)
   - RULESET_ID (FK -> DQ_RULESETS.RULESET_ID)
   - DATASET_NAME (FK -> DQ_DATASETS.DATASET_NAME)
   - RULE_TYPE
   - COLUMN_NAME        -- can be NULL for dataset-level rules
   - LEVEL              -- e.g. TABLE, COLUMN, ROW
   - THRESHOLD
   - MIN_VALUE
   - MAX_VALUE
   - ALLOWED_VALUES     -- VARIANT
   - PARAMS_JSON        -- VARIANT
   - CREATED_AT

4) DQ_DB.DQ_SCHEMA.DQ_RUNS
   - RUN_ID (PK)
   - DATASET_NAME (FK -> DQ_DATASETS.DATASET_NAME)
   - RULESET_ID (FK -> DQ_RULESETS.RULESET_ID)
   - RUN_TYPE           -- e.g. PROFILING, VERIFICATION
   - STARTED_AT
   - FINISHED_AT
   - STATUS             -- e.g. SUCCESS, FAILED, RUNNING
   - TOTAL_ROWS
   - FAILED_ROWS
   - MESSAGE

5) DQ_DB.DQ_SCHEMA.DQ_CONSTRAINT_RESULTS
   - RUN_ID    (FK -> DQ_RUNS.RUN_ID)
   - RULE_ID   (FK -> DQ_RULES.RULE_ID)
   - RULE_TYPE
   - COLUMN_NAME
   - STATUS          -- e.g. PASSED, FAILED
   - MESSAGE
   - METRIC_NAME
   - METRIC_VALUE
   - CREATED_AT

6) DQ_DB.DQ_SCHEMA.DQ_BAD_ROWS
   - RUN_ID       (FK -> DQ_RUNS.RUN_ID)
   - DATASET_NAME (FK -> DQ_DATASETS.DATASET_NAME)
   - RULE_ID      (FK -> DQ_RULES.RULE_ID)
   - PRIMARY_KEY
   - ROW_JSON     -- full failing row
   - VIOLATION_MSG
   - CREATED_AT

7) DQ_DB.DQ_SCHEMA.DQ_PROFILING_METRICS
   - DATASET_NAME (FK -> DQ_DATASETS.DATASET_NAME)
   - COLUMN_NAME
   - METRIC_NAME
   - METRIC_VALUE
   - RUN_TS
   
   

INTERPRETATION RULES
--------------------
- "dq rules" ALWAYS refers to DQ_DB.DQ_SCHEMA.DQ_RULES (optionally joined with RULESETS/DATASETS),
  not to Snowflake physical constraints.
- NEVER use INFORMATION_SCHEMA.TABLE_CONSTRAINTS to answer questions about "dq rules".
- Prefer queries over DQ_DB.DQ_SCHEMA.*. Use INFORMATION_SCHEMA only if the user explicitly
  asks about generic Snowflake metadata (e.g. list schemas).

- When the user says "sales table" or similar:
  - Prefer matching DATASET_NAME or OBJECT_NAME using ILIKE with '%SALES%'.
  - Example: DATASET_NAME ILIKE '%SALES%' OR OBJECT_NAME ILIKE '%SALES%'.

- "active rules" -> filter RULESETS.IS_ACTIVE = TRUE.
- "latest run" -> most recent STARTED_AT in DQ_RUNS (or RUN_TS in profiling metrics).

OUTPUT FORMAT
-------------
- Output exactly one SELECT query, terminated with a semicolon.
- No extra text.
"""



SALES_NL_SQL_SYSTEM_PROMPT = """
You are a Snowflake SQL assistant for the SALES analytics warehouse.

Your job: given a natural language question, generate a SINGLE Snowflake SELECT
query over the SALES_DQ.PUBLIC schema that answers the question.

DATA SCHEMA (simplified)
------------------------
- SALES_DQ.PUBLIC.FACT_SALES(
    SALE_ID,
    CUSTOMER_ID,
    PRODUCT_ID,
    STORE_ID,
    SALE_TS,
    QUANTITY,
    AMOUNT
)
- SALES_DQ.PUBLIC.DIM_CUSTOMERS(
    CUSTOMER_ID,
    CUSTOMER_NAME,
    SEGMENT,
    CITY,
    STATE
)
- SALES_DQ.PUBLIC.DIM_PRODUCTS(
    PRODUCT_ID,
    PRODUCT_NAME,
    CATEGORY,
    SUB_CATEGORY
)
- SALES_DQ.PUBLIC.DIM_STORES(
    STORE_ID,
    STORE_NAME,
    CITY,
    STATE,
    REGION
)

RULES
-----
- Output ONLY valid Snowflake SQL.
- No explanations, no comments, no markdown fences. Only the SQL text.
- The query MUST start with SELECT and MUST end with a semicolon.
- NEVER generate DDL/DML (no CREATE, ALTER, DROP, INSERT, UPDATE, DELETE, MERGE, TRUNCATE).
- You MAY join FACT_SALES with DIM_* tables as needed.
- You MUST NOT reference DQ_DB.DQ_SCHEMA.* here. Those are DQ metadata tables
  handled by a separate agent.

INTERPRETATION
--------------
- Questions about counts, sums, aggregates, top-N, filters, date ranges → query FACT_SALES / DIM_*.
- If user says things like "uniqueness", "duplicates", "nulls", you may generate
  queries that actually check the data (e.g. COUNT(*), COUNT(DISTINCT), GROUP BY HAVING).
"""


DQ_NL_SQL_FEW_SHOTS: List[Dict[str, str]] = [
    # 1) dq rules on sales table
    {
        "role": "user",
        "content": "show me dq rules on sales table",
    },
    {
        "role": "assistant",
        "content": """
SELECT
    r.RULE_ID,
    r.DATASET_NAME,
    r.RULE_TYPE,
    r.COLUMN_NAME,
    r.LEVEL,
    r.THRESHOLD,
    r.MIN_VALUE,
    r.MAX_VALUE,
    r.ALLOWED_VALUES,
    rs.RULESET_ID,
    rs.VERSION,
    rs.IS_ACTIVE
FROM DQ_DB.DQ_SCHEMA.DQ_RULES r
JOIN DQ_DB.DQ_SCHEMA.DQ_RULESETS rs
  ON r.RULESET_ID = rs.RULESET_ID
JOIN DQ_DB.DQ_SCHEMA.DQ_DATASETS d
  ON r.DATASET_NAME = d.DATASET_NAME
WHERE (r.DATASET_NAME ILIKE '%SALES%'
       OR d.OBJECT_NAME ILIKE '%SALES%')
ORDER BY rs.IS_ACTIVE DESC, rs.VERSION DESC, r.RULE_ID;
""".strip(),
    },
    # 2) active dq rules for a dataset
    {
        "role": "user",
        "content": "list active dq rules for dataset sales_orders",
    },
    {
        "role": "assistant",
        "content": """
SELECT
    r.RULE_ID,
    r.DATASET_NAME,
    r.RULE_TYPE,
    r.COLUMN_NAME,
    r.LEVEL,
    r.THRESHOLD,
    r.MIN_VALUE,
    r.MAX_VALUE,
    r.ALLOWED_VALUES,
    rs.VERSION,
    rs.IS_ACTIVE
FROM DQ_DB.DQ_SCHEMA.DQ_RULES r
JOIN DQ_DB.DQ_SCHEMA.DQ_RULESETS rs
  ON r.RULESET_ID = rs.RULESET_ID
WHERE r.DATASET_NAME = 'SALES_ORDERS'
  AND rs.IS_ACTIVE = TRUE
ORDER BY r.RULE_ID;
""".strip(),
    },
    # 3) last 5 dq runs for sales
    {
        "role": "user",
        "content": "show last 5 dq runs for the sales dataset",
    },
    {
        "role": "assistant",
        "content": """
SELECT
    r.RUN_ID,
    r.DATASET_NAME,
    r.RUN_TYPE,
    r.STARTED_AT,
    r.FINISHED_AT,
    r.STATUS,
    r.TOTAL_ROWS,
    r.FAILED_ROWS,
    r.MESSAGE
FROM DQ_DB.DQ_SCHEMA.DQ_RUNS r
JOIN DQ_DB.DQ_SCHEMA.DQ_DATASETS d
  ON r.DATASET_NAME = d.DATASET_NAME
WHERE r.DATASET_NAME ILIKE '%SALES%'
   OR d.OBJECT_NAME ILIKE '%SALES%'
ORDER BY r.STARTED_AT DESC
LIMIT 5;
""".strip(),
    },
    # 4) failed rows for latest run on sales
    {
        "role": "user",
        "content": "show failed rows for latest dq run on sales table",
    },
    {
        "role": "assistant",
        "content": """
WITH latest_run AS (
    SELECT
        r.RUN_ID,
        r.DATASET_NAME
    FROM DQ_DB.DQ_SCHEMA.DQ_RUNS r
    JOIN DQ_DB.DQ_SCHEMA.DQ_DATASETS d
      ON r.DATASET_NAME = d.DATASET_NAME
    WHERE r.DATASET_NAME ILIKE '%SALES%'
       OR d.OBJECT_NAME ILIKE '%SALES%'
    ORDER BY r.STARTED_AT DESC
    LIMIT 1
)
SELECT
    b.RUN_ID,
    b.DATASET_NAME,
    b.RULE_ID,
    b.PRIMARY_KEY,
    b.ROW_JSON,
    b.VIOLATION_MSG,
    b.CREATED_AT
FROM DQ_DB.DQ_SCHEMA.DQ_BAD_ROWS b
JOIN latest_run lr
  ON b.RUN_ID = lr.RUN_ID
ORDER BY b.CREATED_AT DESC;
""".strip(),
    },
    # 5) summary of rule failures for last run
    {
        "role": "user",
        "content": "give me summary of dq rule failures for the last dq run on the sales dataset",
    },
    {
        "role": "assistant",
        "content": """
WITH latest_run AS (
    SELECT
        r.RUN_ID,
        r.DATASET_NAME
    FROM DQ_DB.DQ_SCHEMA.DQ_RUNS r
    JOIN DQ_DB.DQ_SCHEMA.DQ_DATASETS d
      ON r.DATASET_NAME = d.DATASET_NAME
    WHERE r.DATASET_NAME ILIKE '%SALES%'
       OR d.OBJECT_NAME ILIKE '%SALES%'
    ORDER BY r.STARTED_AT DESC
    LIMIT 1
)
SELECT
    cr.RULE_ID,
    rl.RULE_TYPE,
    rl.COLUMN_NAME,
    cr.STATUS,
    COUNT(*) AS RESULT_COUNT
FROM DQ_DB.DQ_SCHEMA.DQ_CONSTRAINT_RESULTS cr
JOIN latest_run lr
  ON cr.RUN_ID = lr.RUN_ID
JOIN DQ_DB.DQ_SCHEMA.DQ_RULES rl
  ON cr.RULE_ID = rl.RULE_ID
GROUP BY
    cr.RULE_ID,
    rl.RULE_TYPE,
    rl.COLUMN_NAME,
    cr.STATUS
ORDER BY RESULT_COUNT DESC;
""".strip(),
    },
    {
    "role": "user",
    "content": "give me query to check uniqueness of sales_id on sales table"
    },
    {
        "role": "assistant",
        "content": """
    SELECT
        RULE_ID,
        DATASET_NAME,
        COLUMN_NAME,
        RULE_TYPE,
        THRESHOLD,
        PARAMS_JSON,
        LEVEL,
        CREATED_AT
    FROM DQ_DB.DQ_SCHEMA.DQ_RULES
    WHERE RULE_TYPE ILIKE '%UNIQUE%'
    AND COLUMN_NAME ILIKE '%SALES_ID%'
    AND (DATASET_NAME ILIKE '%SALES%' OR COLUMN_NAME ILIKE '%SALES%')
    ORDER BY CREATED_AT DESC;
    """.strip()
    }
]

SALES_NL_SQL_FEW_SHOTS: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "show top 10 products by total sales amount in the last 30 days",
    },
    {
        "role": "assistant",
        "content": """
SELECT
    p.PRODUCT_ID,
    p.PRODUCT_NAME,
    SUM(f.AMOUNT) AS total_amount
FROM SALES_DQ.PUBLIC.FACT_SALES f
JOIN SALES_DQ.PUBLIC.DIM_PRODUCTS p
  ON f.PRODUCT_ID = p.PRODUCT_ID
WHERE f.SALE_TS >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY p.PRODUCT_ID, p.PRODUCT_NAME
ORDER BY total_amount DESC
LIMIT 10;
""".strip(),
    },
    {
        "role": "user",
        "content": "check uniqueness of SALE_ID in FACT_SALES",
    },
    {
        "role": "assistant",
        "content": """
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT SALE_ID) AS distinct_sale_ids
FROM SALES_DQ.PUBLIC.FACT_SALES;
""".strip(),
    },
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _build_sales_nl_sql_messages(user_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SALES_NL_SQL_SYSTEM_PROMPT}
    ]
    messages.extend(SALES_NL_SQL_FEW_SHOTS)
    messages.append({"role": "user", "content": user_prompt})
    return messages

def _generate_sales_sql_from_nl(user_prompt: str) -> str:
    """
    NL -> SQL for SALES_DQ.PUBLIC data tables.
    """
    messages = _build_sales_nl_sql_messages(user_prompt)

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )

    raw_sql = (completion.choices[0].message.content or "").strip()

    # Strip ```sql fences if present
    if raw_sql.startswith("```"):
        raw_sql = raw_sql.strip("`")
        if raw_sql.lower().startswith("sql"):
            raw_sql = raw_sql.split("\n", 1)[1].strip()

    if not raw_sql.rstrip().endswith(";"):
        raw_sql = raw_sql.rstrip() + ";"

    upper = raw_sql.upper()

    # Guardrails: only SELECT and only SALES_DQ.PUBLIC.*, no DQ_DB
    if not upper.lstrip().startswith("SELECT"):
        raise ValueError(f"Generated SQL is not a SELECT statement: {raw_sql}")

    if "DQ_DB.DQ_SCHEMA" in upper:
        raise ValueError(f"Generated SQL incorrectly references DQ_DB: {raw_sql}")

    if "SALES_DQ.PUBLIC" not in upper:
        # Be a bit flexible, but at least enforce SALES_DQ presence
        if "SALES_DQ." not in upper:
            raise ValueError(f"Generated SQL does not reference SALES_DQ: {raw_sql}")

    return raw_sql

def _df_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Safely convert a DataFrame to a JSON-serializable list[dict].
    """
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _build_nl_sql_messages(user_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DQ_NL_SQL_SYSTEM_PROMPT}
    ]
    messages.extend(DQ_NL_SQL_FEW_SHOTS)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _generate_sql_from_nl(user_prompt: str) -> str:
    
    print(f"[DEBUG] _generate_sql_from_nl CALLED with prompt: {user_prompt}", file=sys.stderr, flush=True)
    
    """
    Convert natural language to SQL via OpenAI, with hard guardrails:
    - Only DQ_DB.DQ_SCHEMA.* allowed
    - No SALES_DQ / SALES_DB / PUBLIC fact tables
    """
    messages = _build_nl_sql_messages(user_prompt)

    def _call_llm(msgs: List[Dict[str, str]]) -> str:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # strip ```sql fences if present
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("sql"):
                raw = raw.split("\n", 1)[1].strip()

        if not raw.rstrip().endswith(";"):
            raw = raw.rstrip() + ";"

        return raw

    sql = _call_llm(messages)
    upper = sql.upper()

    # If it references forbidden DBs or doesn't touch DQ_DB.DQ_SCHEMA at all, try once more with a correction
    forbidden = ("SALES_DQ.", "SALES_DB.", " PUBLIC.", "FROM FACT_", "FROM SALES_")
    if any(f in upper for f in forbidden) or "DQ_DB.DQ_SCHEMA" not in upper:
        # Add a very explicit corrective message and retry ONCE
        messages.append({
            "role": "assistant",
            "content": (
                "You must not query SALES_DQ, SALES_DB, PUBLIC or any raw data tables. "
                "Rewrite the query so it ONLY uses the metadata tables in DQ_DB.DQ_SCHEMA "
                "(DQ_RULES, DQ_RULESETS, DQ_DATASETS, DQ_RUNS, DQ_CONSTRAINT_RESULTS, "
                "DQ_BAD_ROWS, DQ_PROFILING_METRICS). Output only the SQL."
            ),
        })
        messages.append({"role": "user", "content": user_prompt})
        sql = _call_llm(messages)
        upper = sql.upper()

    # If it's STILL bad, fail fast instead of returning wrong SQL
    if any(f in upper for f in forbidden) or "DQ_DB.DQ_SCHEMA" not in upper:
        raise ValueError(
            f"Generated SQL references non-DQ_DB objects or no DQ_DB.DQ_SCHEMA tables: {sql}"
        )

    return sql


# def _generate_sql_from_nl(user_prompt: str) -> str:
#     """
#     Synchronous helper: convert natural language to SQL via OpenAI.
#     """
#     messages = _build_nl_sql_messages(user_prompt)

#     completion = openai_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         temperature=0.0,
#     )

#     raw_sql = (completion.choices[0].message.content or "").strip()

#     # In case the model ever wraps in ```sql fences
#     if raw_sql.startswith("```"):
#         raw_sql = raw_sql.strip("`")
#         if raw_sql.lower().startswith("sql"):
#             raw_sql = raw_sql.split("\n", 1)[1].strip()

#     if not raw_sql.rstrip().endswith(";"):
#         raw_sql = raw_sql.rstrip() + ";"

#     return raw_sql


def _run_snowflake_query(sql: str) -> List[Dict[str, Any]]:
    """
    Use existing Snowflake connection helper to run a query and return rows as list[dict].
    """
    conn = get_snowflake_connection()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    return _df_to_records(df)




def _ensure_default_ruleset(dataset: str) -> str:
    """
    Ensure a default active ruleset exists for the dataset and return its RULESET_ID.

    Convention:
      RULESET_ID = '<DATASET_NAME>__DEFAULT'
      VERSION    = 1
      IS_ACTIVE  = TRUE
    """
    ds = dataset.upper()
    ruleset_id = f"{ds}__DEFAULT"

    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            # Insert if not exists
            cur.execute(
                """
                INSERT INTO DQ_DB.DQ_SCHEMA.DQ_RULESETS (
                    RULESET_ID, DATASET_NAME, VERSION, IS_ACTIVE, DESCRIPTION
                )
                SELECT %s, %s, 1, TRUE,
                       CONCAT('Default ruleset for ', %s)
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM DQ_DB.DQ_SCHEMA.DQ_RULESETS
                    WHERE RULESET_ID = %s
                )
                """,
                (ruleset_id, ds, ds, ruleset_id),
            )
        conn.commit()
    finally:
        conn.close()

    return ruleset_id

def _persist_rules_to_dq_db(dataset: str, rules: list) -> int:
    """
    Persist inferred rules into DQ_DB.DQ_SCHEMA.DQ_RULES.

    Supports both shapes:
    - RuleSpec-style (NL Rules agent):
      { "dataset", "type", "column", "level", "threshold", "min", "max", "allowed_values", ... }

    - (future) DQ Brain-style:
      { "ruleType", "column", "level", "threshold", "minValue", "maxValue", "allowedValues", ... }
    """
    if not rules:
        return 0

    ds = dataset.upper()
    ruleset_id = _ensure_default_ruleset(ds)

    conn = get_snowflake_connection()
    inserted = 0
    try:
        with conn.cursor() as cur:
            for r in rules:
                # Handle both Pydantic objects and plain dicts
                if isinstance(r, dict):
                    get = r.get
                else:
                    get = lambda k, default=None: getattr(r, k, default)

                # -----------------------------
                # 1) Normalize rule type
                # -----------------------------
                raw_type = (
                    get("ruleType")
                    or get("RULE_TYPE")
                    or get("type")
                    or get("rule_type")
                    or get("type_")
                )

                if not raw_type:
                    # This is what was causing RULE_TYPE NULL issues before
                    raise ValueError(f"Rule has no type set: {r}")

                normalized = RULE_TYPE_ALIASES.get(str(raw_type), str(raw_type))
                rule_type_db = normalized.upper()

                # -----------------------------
                # 2) Other scalar fields
                # -----------------------------
                col = (
                    get("column")
                    or get("column_name")
                    or get("COLUMN_NAME")
                )
                level = get("level") or get("LEVEL") or "ERROR"

                threshold = get("threshold")

                # support min/min_value/minValue etc.
                minv = (
                    get("minValue")
                    or get("min_value")
                    or get("min")
                )
                maxv = (
                    get("maxValue")
                    or get("max_value")
                    or get("max")
                )

                # -----------------------------
                # 3) Allowed values
                # -----------------------------
                allowed_values_obj = (
                    get("allowedValues")
                    or get("allowed_values")
                    or get("ALLOWED_VALUES")
                )

                allowed_values_json = (
                    json.dumps(allowed_values_obj, allow_nan=False)
                    if allowed_values_obj is not None
                    else None
                )

                # -----------------------------
                # 4) Params JSON (optional extras)
                # -----------------------------
                params = get("params_json") or get("params") or {}
                if not isinstance(params, dict):
                    # be defensive – if a string was passed, don't blow up
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {"raw": params}

                params_json = (
                    json.dumps(params, allow_nan=False) if params else None
                )

                rule_id = str(uuid.uuid4())

                cur.execute(
                    """
                    INSERT INTO DQ_DB.DQ_SCHEMA.DQ_RULES (
                        RULE_ID,
                        RULESET_ID,
                        DATASET_NAME,
                        RULE_TYPE,
                        COLUMN_NAME,
                        LEVEL,
                        THRESHOLD,
                        MIN_VALUE,
                        MAX_VALUE,
                        ALLOWED_VALUES,
                        PARAMS_JSON
                    )
                    SELECT
                        %s AS RULE_ID,
                        %s AS RULESET_ID,
                        %s AS DATASET_NAME,
                        %s AS RULE_TYPE,
                        %s AS COLUMN_NAME,
                        %s AS LEVEL,
                        %s AS THRESHOLD,
                        %s AS MIN_VALUE,
                        %s AS MAX_VALUE,
                        CASE
                            WHEN %s IS NULL THEN NULL
                            ELSE PARSE_JSON(%s)
                        END AS ALLOWED_VALUES,
                        CASE
                            WHEN %s IS NULL THEN NULL
                            ELSE PARSE_JSON(%s)
                        END AS PARAMS_JSON
                    """,
                    (
                        rule_id,
                        ruleset_id,
                        ds,
                        rule_type_db,
                        col,
                        level,
                        threshold,
                        minv,
                        maxv,
                        allowed_values_json,  # for CASE WHEN
                        allowed_values_json,  # for PARSE_JSON
                        params_json,          # for CASE WHEN
                        params_json,          # for PARSE_JSON
                    ),
                )
                inserted += 1

        conn.commit()
    finally:
        conn.close()

    return inserted


# def _persist_rules_to_dq_db(dataset: str, rules: list) -> int:
#     """
#     Persist inferred rules into DQ_DB.DQ_SCHEMA.DQ_RULES.

#     - Uses a default ruleset: <DATASET>__DEFAULT.
#     - Generates RULE_ID via uuid4().
#     - Maps generic RuleSpec fields to DQ_RULES schema.
#     Returns: number of rules inserted.
#     """
#     if not rules:
#         return 0

#     ds = dataset.upper()
#     ruleset_id = _ensure_default_ruleset(ds)

#     conn = get_snowflake_connection()
#     inserted = 0
#     try:
#         with conn.cursor() as cur:
#             for r in rules:
#                 # Be defensive: handle both Pydantic objects and plain dicts
#                 if isinstance(r, dict):
#                     get = r.get
#                 else:
#                     get = lambda k, default=None: getattr(r, k, default)

#                 raw_type = get("type") or get("rule_type") or get("type_", "")
#                 col = get("column") or get("column_name") or get("COLUMN_NAME")
#                 level = get("level") or "ERROR"
#                 threshold = get("threshold")
#                 minv = get("min") or get("min_value")
#                 maxv = get("max") or get("max_value")
#                 allowed_values = get("allowed_values")
#                 params = get("params_json") or get("params")

#                 rule_id = str(uuid.uuid4())
                
#                 if not raw_type:
#                     # Fail fast with a clear error instead of sending NULL to Snowflake
#                     raise ValueError(f"Rule has no type set: {r}")
                
#                 normalized = RULE_TYPE_ALIASES.get(str(raw_type), str(raw_type))
#                 rule_type_db = normalized.upper()    # what goes into RULE_TYPE column

#                 cur.execute(
#                     """
#                     INSERT INTO DQ_DB.DQ_SCHEMA.DQ_RULES (
#                         RULE_ID,
#                         RULESET_ID,
#                         DATASET_NAME,
#                         RULE_TYPE,
#                         COLUMN_NAME,
#                         LEVEL,
#                         THRESHOLD,
#                         MIN_VALUE,
#                         MAX_VALUE,
#                         ALLOWED_VALUES,
#                         PARAMS_JSON
#                     )
#                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s))
#                     """,
#                     (
#                         rule_id,
#                         ruleset_id,
#                         ds,
#                         (rule_type_db or "").upper(),
#                         col,
#                         level,
#                         threshold,
#                         minv,
#                         maxv,
#                         json.dumps(allowed_values) if allowed_values is not None else None,
#                         json.dumps(params) if params is not None else None,
#                     ),
#                 )
#                 inserted += 1

#         conn.commit()
#     finally:
#         conn.close()

#     return inserted

# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

@mcp.tool()
async def dq_run_spark_pipeline(dataset: str, batch_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Kick off Spark-based DQ jobs for the given dataset via app_local.run_spark_jobs.
    Always returns a JSON dict, even on failure.
    """
    ds = dataset.upper()
    try:
        logs = run_spark_jobs(ds)
        return {
            "dataset": ds,
            "batch_date": batch_date,
            "logs": logs,
            "error": None,
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "dataset": ds,
            "batch_date": batch_date,
            "logs": "",
            "error": str(e),
            "traceback": tb,
        }


@mcp.tool()
async def dq_latest_profiling(dataset: str) -> Dict[str, Any]:
    """
    Return latest profiling metrics + a summarized view.
    """
    ds = dataset.upper()
    metrics_df = load_latest_profiling(ds)
    summary_df = summarize_profiling(metrics_df)

    return {
        "dataset": ds,
        "metrics": _df_to_records(metrics_df),
        "summary": _df_to_records(summary_df),
    }


@mcp.tool()
async def dq_latest_verification(dataset: str) -> Dict[str, Any]:
    """
    Return a JSON-safe summary of the latest verification run.
    """
    ds = dataset.upper()
    run_ts, run_id, verif_df = load_latest_verification_run(ds)

    # Normalize run_id to JSON-friendly type
    try:
        if run_id is None:
            norm_run_id: Any = None
        elif isinstance(run_id, (int, str)):
            norm_run_id = run_id
        else:
            norm_run_id = int(run_id)
    except Exception:
        norm_run_id = str(run_id)

    constraints = _df_to_records(verif_df)

    total_rules = len(constraints)
    failed_rules = sum(
        1
        for c in constraints
        if str(c.get("STATUS", "")).upper() in ("FAIL", "FAILED", "ERROR")
    )

    return {
        "dataset": ds,
        "run_id": norm_run_id,
        "run_ts": str(run_ts),
        "total_rules": total_rules,
        "failed_rules": failed_rules,
        "constraints": constraints,
    }


@mcp.tool()
async def dq_query_snowflake(sql: str) -> Dict[str, Any]:
    """
    Run an arbitrary SQL query against Snowflake and return rows.

    NOTE: This expects VALID SQL, not natural language. For NL -> SQL over
    DQ metadata, use dq_dqdb_nl_sql instead.
    """
    try:
        rows = _run_snowflake_query(sql)
        return {"sql": sql, "rows": rows, "error": None}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[dq_mcp_server] dq_query_snowflake ERROR: {e}\n{tb}", file=sys.stderr, flush=True)
        return {
            "sql": sql,
            "rows": [],
            "error": str(e),
            "traceback": tb,
        }


@mcp.tool()
async def dq_dqdb_nl_sql(prompt: str, execute: bool = True) -> Dict[str, Any]:
    print(f"[DEBUG] dq_dqdb_nl_sql TOOL CALLED with prompt: {prompt}", file=sys.stderr, flush=True)
    """
    Natural language -> SQL over DQ_DB.DQ_SCHEMA.*.

    Example prompt: "show me dq rules on sales table"

    - Generates SQL using OpenAI with strong grounding in DQ_DB.DQ_SCHEMA
    - If execute=True (default), runs the query and returns rows.
    """
    if not prompt:
        return {"error": "prompt is required"}

    try:
        sql = _generate_sql_from_nl(prompt)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[dq_mcp_server] dq_dqdb_nl_sql SQL-GEN ERROR: {e}\n{tb}", file=sys.stderr, flush=True)
        return {
            "error": f"failed to generate SQL from prompt",
            "prompt": prompt,
            "traceback": tb,
        }

    upper_sql = sql.upper()

    # Guardrails: only SELECT; prevent old INFORMATION_SCHEMA.TABLE_CONSTRAINTS bug for dq rules
    if not upper_sql.lstrip().startswith("SELECT"):
        return {
            "error": "Generated SQL is not a SELECT statement.",
            "prompt": prompt,
            "sql": sql,
        }

    if "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in upper_sql and "DQ_RULES" not in upper_sql:
        return {
            "error": "Generated SQL is incorrectly using INFORMATION_SCHEMA.TABLE_CONSTRAINTS.",
            "prompt": prompt,
            "sql": sql,
        }

    if not execute:
        return {
            "prompt": prompt,
            "sql": sql,
            "executed": False,
            "rows": [],
        }

    try:
        rows = _run_snowflake_query(sql)
        return {
            "prompt": prompt,
            "sql": sql,
            "executed": True,
            "row_count": len(rows),
            "rows": rows,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[dq_mcp_server] dq_dqdb_nl_sql EXEC ERROR: {e}\n{tb}", file=sys.stderr, flush=True)
        return {
            "prompt": prompt,
            "sql": sql,
            "executed": False,
            "error": str(e),
            "traceback": tb,
        }

@mcp.tool()
async def dq_sales_nl_sql(
    prompt: str,
    execute: bool = True,
) -> Dict[str, Any]:
    """
    Natural language -> SQL over SALES_DQ (data warehouse).

    Example prompt: "show top 10 products by sales amount in the last 30 days"

    - Generates SQL using OpenAI with grounding in SALES_DQ.PUBLIC tables.
    - If execute=True (default), runs the query and returns rows via Snowflake.
    """
    if not prompt:
        return {"error": "prompt is required"}

    try:
        sql = _generate_sales_sql_from_nl(prompt)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[dq_mcp_server] dq_sales_nl_sql SQL-GEN ERROR: {e}\n{tb}", file=sys.stderr, flush=True)
        return {
            "prompt": prompt,
            "error": f"failed to generate SQL from prompt",
            "traceback": tb,
        }

    if not execute:
        return {
            "prompt": prompt,
            "sql": sql,
            "executed": False,
            "rows": [],
        }

    try:
        rows = _run_snowflake_query(sql)
        return {
            "prompt": prompt,
            "sql": sql,
            "executed": True,
            "row_count": len(rows),
            "rows": rows,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[dq_mcp_server] dq_sales_nl_sql EXEC ERROR: {e}\n{tb}", file=sys.stderr, flush=True)
        return {
            "prompt": prompt,
            "sql": sql,
            "executed": False,
            "error": str(e),
            "traceback": tb,
        }


@mcp.tool(name="dq_nl_rules_agent")
async def dq_nl_rules_agent(
    dataset: str,
    prompt: str,
    apply: bool = False,
    feedback: Optional[str] = None,
    self_healing: bool = True,
) -> Dict[str, Any]:
    """
    MCP tool: given dataset + prompt (and optional apply flag),
    run the NL → Rules LangGraph workflow and return rules + messages.
    """
    ds = (dataset or "").upper()

    if not ds:
        return {"error": "dataset is required"}
    if not prompt:
        return {"error": "prompt is required", "dataset": ds}

    try:
        # 1) Prepare initial graph state
        columns = get_dataset_columns(ds)
        request = NLRequest(dataset=ds, prompt=prompt, apply=apply)

        init_state = GraphState(
            request=request,
            columns=columns,
            user_feedback=feedback,
            self_healing_enabled=True,
        )

        # 2) Build and invoke LangGraph app
        app = build_graph()
        raw_result = app.invoke(init_state)

        if isinstance(raw_result, GraphState):
            final_state = raw_result
        else:
            final_state = GraphState.model_validate(raw_result)
            
        inferred_rules = list(final_state.inferred_rules or [])
        
        # 3) Optionally apply/persist rules
        applied_count = 0
        if apply and inferred_rules:
            try:
                applied_count = _persist_rules_to_dq_db(ds, inferred_rules)
            except Exception as persist_err:
                # Do NOT blow up the tool call; just log and surface as message
                msg = f"Failed to persist rules to DQ_DB.DQ_SCHEMA.DQ_RULES: {persist_err}"
                (final_state.validation_messages or []).append(msg)

        # 4) Shape into JSON-safe dict for Streamlit
        return {
            "dataset": ds,
            "prompt": prompt,
            "apply": apply,
            "applied_rule_count": applied_count,
            "messages": final_state.validation_messages or [],
            "anomaly_messages": final_state.anomaly_messages or [],
            "rules": [r.dict() for r in inferred_rules],
            "original_rules": (
                [r.dict() for r in (final_state.rules or [])]
                if getattr(final_state, "rules", None)
                else []
            ),
        }

        # # 3) Shape into JSON-safe dict for Streamlit
        # return {
        #     "dataset": ds,
        #     "prompt": prompt,
        #     "apply": apply,
        #     "messages": final_state.validation_messages or [],
        #     "anomaly_messages": final_state.anomaly_messages or [],
        #     "rules": [r.dict() for r in (final_state.inferred_rules or [])],
        #     "original_rules": (
        #         [r.dict() for r in (final_state.rules or [])]
        #         if getattr(final_state, "rules", None)
        #         else []
        #     ),
        # }

    except Exception as e:
        tb = traceback.format_exc()
        print(
            f"[dq_mcp_server] dq_nl_rules_agent ERROR for dataset={ds}: {e}\n{tb}",
            file=sys.stderr,
            flush=True,
        )
        return {
            "error": str(e),
            "traceback": tb,
            "dataset": ds,
            "prompt": prompt,
        }


# ---------------------------------------------------------------------
# Entrypoint for stdio MCP
# ---------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")


# from __future__ import annotations

# import json
# import os
# import sys
# import traceback
# from typing import Any, Dict, List, Optional

# import pandas as pd
# from dotenv import load_dotenv
# from mcp.server.fastmcp import FastMCP

# from nl_constraints_graph.models import GraphState, NLRequest
# from nl_constraints_graph.graph_nl_to_yaml import build_graph
# from nl_constraints_graph.nodes_validate import get_dataset_columns

# # ---------------------------------------------------------------------
# # Path / env bootstrap
# # ---------------------------------------------------------------------

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# load_dotenv()

# from nl_constraints_graph.dq_brain import (
#     get_snowflake_connection,
#     load_latest_profiling,
#     summarize_profiling,
# )

# from nl_constraints_graph.streamlit_app import (
#     run_spark_jobs,
#     load_latest_verification_run,
#     load_bad_rows_for_run,
# )

# mcp = FastMCP("dq_mcp_server")


# # ---------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------


# def _df_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
#     """
#     Safely convert a DataFrame to a JSON-serializable list[dict].

#     Using to_json + json.loads ensures all pandas / numpy types are
#     converted to plain Python scalars (int, float, str, bool, None).
#     """
#     if df is None or df.empty:
#         return []
#     return json.loads(df.to_json(orient="records"))


# # ---------------------------------------------------------------------
# # Tools
# # ---------------------------------------------------------------------


# @mcp.tool()
# async def dq_run_spark_pipeline(dataset: str, batch_date: Optional[str] = None) -> Dict[str, Any]:
#     """
#     Kick off Spark-based DQ jobs for the given dataset via app_local.run_spark_jobs.
#     Always returns a JSON dict, even on failure.
#     """
#     ds = dataset.upper()
#     try:
#         logs = run_spark_jobs(ds)
#         return {
#             "dataset": ds,
#             "batch_date": batch_date,
#             "logs": logs,
#             "error": None,
#         }
#     except Exception as e:
#         # Never let the exception kill the MCP task group
#         tb = traceback.format_exc()
#         return {
#             "dataset": ds,
#             "batch_date": batch_date,
#             "logs": "",
#             "error": str(e),
#             "traceback": tb,
#         }
        
# @mcp.tool()
# async def dq_latest_profiling(dataset: str) -> Dict[str, Any]:
#     """
#     Return latest profiling metrics + a summarized view.
#     """
#     ds = dataset.upper()
#     metrics_df = load_latest_profiling(ds)
#     summary_df = summarize_profiling(metrics_df)

#     return {
#         "dataset": ds,
#         "metrics": _df_to_records(metrics_df),
#         "summary": _df_to_records(summary_df),
#     }


# @mcp.tool()
# async def dq_latest_verification(dataset: str) -> Dict[str, Any]:
#     """
#     Return a JSON-safe summary of the latest verification run.

#     NOTE: We deliberately do NOT include bad_rows here to avoid
#     potential heavy payload / serialization pitfalls. You can extend
#     this later once everything is stable.
#     """
#     ds = dataset.upper()
#     run_ts, run_id, verif_df = load_latest_verification_run(ds)

#     # Normalize run_id to JSON-friendly type
#     try:
#         if run_id is None:
#             norm_run_id: Any = None
#         elif isinstance(run_id, (int, str)):
#             norm_run_id = run_id
#         else:
#             # e.g. numpy scalar
#             norm_run_id = int(run_id)
#     except Exception:
#         norm_run_id = str(run_id)

#     constraints = _df_to_records(verif_df)

#     total_rules = len(constraints)
#     failed_rules = sum(
#         1
#         for c in constraints
#         if str(c.get("STATUS", "")).upper() in ("FAIL", "FAILED", "ERROR")
#     )

#     return {
#         "dataset": ds,
#         "run_id": norm_run_id,
#         "run_ts": str(run_ts),
#         "total_rules": total_rules,
#         "failed_rules": failed_rules,
#         "constraints": constraints,
#         # "bad_rows": ...  # intentionally omitted for now
#     }

# @mcp.tool()
# async def dq_query_snowflake(sql: str) -> Dict[str, Any]:
#     """
#     Run an arbitrary SQL query against Snowflake and return rows.
#     """
#     conn = get_snowflake_connection()
#     try:
#         df = pd.read_sql(sql, conn)
#     finally:
#         conn.close()
#     return {"sql": sql, "rows": _df_to_records(df)}


# @mcp.tool(name="dq_nl_rules_agent")
# async def dq_nl_rules_agent(
#     dataset: str,
#     prompt: str,
#     apply: bool = False,
#     feedback: Optional[str] = None,
#     self_healing: bool = True,
# ) -> Dict[str, Any]:
#     """
#     MCP tool: given dataset + prompt (and optional apply flag),
#     run the NL → Rules LangGraph workflow and return rules + messages.

#     This version is defensive: any internal error is caught and returned
#     as JSON instead of crashing the tool call.
#     """
#     ds = (dataset or "").upper()

#     if not ds:
#         return {"error": "dataset is required"}
#     if not prompt:
#         return {"error": "prompt is required", "dataset": ds}

#     try:
#         # 1) Prepare initial graph state
#         columns = get_dataset_columns(ds)
#         request = NLRequest(dataset=ds, prompt=prompt, apply=apply)

#         init_state = GraphState(
#             request=request,
#             columns=columns,
#             user_feedback=feedback,
#             self_healing_enabled=self_healing,
#         )

#         # 2) Build and invoke LangGraph app
#         app = build_graph()
#         raw_result = app.invoke(init_state)

#         if isinstance(raw_result, GraphState):
#             final_state = raw_result
#         else:
#             final_state = GraphState.model_validate(raw_result)

#         # 3) Shape into JSON-safe dict for Streamlit
#         return {
#             "dataset": ds,
#             "prompt": prompt,
#             "apply": apply,
#             "messages": final_state.validation_messages or [],
#             "anomaly_messages": final_state.anomaly_messages or [],
#             "rules": [r.dict() for r in (final_state.inferred_rules or [])],
#             "original_rules": (
#                 [r.dict() for r in (final_state.rules or [])]
#                 if getattr(final_state, "rules", None)
#                 else []
#             ),
#         }

#     except Exception as e:
#         tb = traceback.format_exc()
#         # Log to stderr so you can see it in journalctl
#         print(
#             f"[dq_mcp_server] dq_nl_rules_agent ERROR for dataset={ds}: {e}\n{tb}",
#             file=sys.stderr,
#             flush=True,
#         )
#         # Return a JSON error payload instead of raising
#         return {
#             "error": str(e),
#             "traceback": tb,
#             "dataset": ds,
#             "prompt": prompt,
#         }

# # ---------------------------------------------------------------------
# # Entrypoint for stdio MCP
# # ---------------------------------------------------------------------

# if __name__ == "__main__":
#     mcp.run(transport="stdio")