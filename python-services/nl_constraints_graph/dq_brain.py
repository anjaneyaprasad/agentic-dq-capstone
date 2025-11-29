import json
import os
import uuid
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

DQ_DB = "DQ_DB"
DQ_SCHEMA = "DQ_SCHEMA"

load_dotenv()

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

# =========================== Snowflake helpers ===========================

def get_snowflake_connection():
    """
    Single Snowflake connection factory.

    NOTE: We do NOT set database/schema here so we can freely query
    both DQ_DB.* (metadata) and SALES_DQ.* (actual data) with
    fully-qualified names in SQL.
    """
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        # No database/schema here on purpose
    )

def list_profiled_datasets() -> List[str]:
    """
    Return all dataset names that have profiling metrics in DQ_PROFILING_METRICS.
    """
    conn = get_snowflake_connection()
    try:
        df = pd.read_sql(
            """
            SELECT DISTINCT DATASET_NAME
            FROM DQ_DB.DQ_SCHEMA.DQ_PROFILING_METRICS
            ORDER BY DATASET_NAME
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return []
    return df["DATASET_NAME"].tolist()

def load_profiling_summary(dataset_name: str) -> pd.DataFrame:
    """
    Convenience wrapper:
      - load latest profiling snapshot for a dataset
      - convert to per-column summary using summarize_profiling
    """
    df = load_latest_profiling(dataset_name)
    if df.empty:
        return df
    return summarize_profiling(df)

def load_active_datasets() -> pd.DataFrame:
    """
    Return active datasets from DQ_DATASETS.
    """
    conn = get_snowflake_connection()
    try:
        sql = f"""
        SELECT DATASET_NAME
        FROM {DQ_DB}.{DQ_SCHEMA}.DQ_DATASETS
        WHERE IS_ACTIVE = TRUE
        ORDER BY DATASET_NAME
        """
        return pd.read_sql(sql, conn)
    finally:
        conn.close()

def load_latest_profiling(dataset_name: str) -> pd.DataFrame:
    """
    Load the latest profiling snapshot for a dataset from
    DQ_DB.DQ_SCHEMA.DQ_PROFILING_METRICS.
    Converts RUN_TS (epoch ms) to proper timestamp.
    """
    ds = dataset_name.upper()

    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(RUN_TS)
                FROM DQ_DB.DQ_SCHEMA.DQ_PROFILING_METRICS
                WHERE DATASET_NAME = %s
                """,
                (ds,),
            )
            row = cur.fetchone()
            if not row or row[0] is None:
                return pd.DataFrame()

            latest_ts = row[0]

            cur.execute(
                """
                SELECT
                    DATASET_NAME,
                    COLUMN_NAME,
                    METRIC_NAME,
                    METRIC_VALUE,
                    RUN_TS
                FROM DQ_DB.DQ_SCHEMA.DQ_PROFILING_METRICS
                WHERE DATASET_NAME = %s
                  AND RUN_TS      = %s
                ORDER BY COLUMN_NAME, METRIC_NAME
                """,
                (ds, latest_ts),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)

    # --- 🔧 FIX: Convert epoch ms → timestamp ---
    if "RUN_TS" in df.columns:
        try:
            df["RUN_TS"] = pd.to_datetime(df["RUN_TS"], unit="ms")
        except Exception:
            # fallback if Snowflake starts returning timestamp type later
            df["RUN_TS"] = pd.to_datetime(df["RUN_TS"], errors="ignore")

    return df

def _safe_float(x: Any) -> float:
    """Helper: safely cast metric to float, treating None/NaN as 0."""
    if x is None:
        return 0.0
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return f

# =========================== Core transforms ===========================

def metrics_to_json_per_column(metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Take DF with columns: COLUMN_NAME, METRIC_NAME, METRIC_VALUE
    and turn into JSON structure per column:

    [
      {
        "column": "COL1",
        "metrics": {
          "Completeness": 0.98,
          "ApproxCountDistinct": 1234,
          ...
        }
      },
      ...
    ]
    """
    grouped: Dict[Optional[str], Dict[str, Any]] = {}

    for _, row in metrics_df.iterrows():
        col = row["COLUMN_NAME"]
        metric = row["METRIC_NAME"]
        value = row["METRIC_VALUE"]
        if col not in grouped:
            grouped[col] = {}
        grouped[col][metric] = value

    result: List[Dict[str, Any]] = []
    for col, metrics in grouped.items():
        result.append(
            {
                "column": col,
                "metrics": metrics,
            }
        )
    return result

def build_dq_brain_payload(dataset_name: str, metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a compact payload (Python dict) that can be embedded into prompts or logs.
    """
    cols_json = metrics_to_json_per_column(metrics_df)
    payload = {
        "dataset": dataset_name,
        "columns": cols_json,
    }
    return payload

def summarize_profiling(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert low-level Deequ-style metrics into a human-friendly summary per column.

    Input: long-form profiling dataframe from load_latest_profiling:
      DATASET_NAME, COLUMN_NAME, METRIC_NAME, METRIC_VALUE, RUN_TS

    Output: one row per column with:
      - column
      - inferred_type
      - non_null_pct
      - approx_distinct
      - notes
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(
            columns=["column", "inferred_type", "non_null_pct", "approx_distinct", "notes"]
        )

    cols_json = metrics_to_json_per_column(metrics_df)

    rows: List[Dict[str, Any]] = []

    for col_entry in cols_json:
        col_name = col_entry["column"]
        metrics = col_entry.get("metrics") or {}

        compl = metrics.get("Completeness")
        approx_distinct = metrics.get("ApproxCountDistinct")

        # Histogram buckets for type inference
        str_abs = _safe_float(metrics.get("Histogram.abs.String"))
        int_abs = _safe_float(metrics.get("Histogram.abs.Integral"))
        frac_abs = _safe_float(metrics.get("Histogram.abs.Fractional"))
        bool_abs = _safe_float(metrics.get("Histogram.abs.Boolean"))

        type_label = "Unknown"
        buckets = {
            "String": str_abs,
            "Integral": int_abs,
            "Fractional": frac_abs,
            "Boolean": bool_abs,
        }
        if any(v > 0 for v in buckets.values()):
            type_label = max(buckets, key=buckets.get)

        # Non-null % and approx. distinct
        non_null_pct: Optional[float] = None
        if isinstance(compl, (int, float)):
            try:
                non_null_pct = round(float(compl) * 100, 2)
            except Exception:
                non_null_pct = None

        approx_distinct_int: Optional[int] = None
        if isinstance(approx_distinct, (int, float)):
            try:
                approx_distinct_int = int(approx_distinct)
            except Exception:
                approx_distinct_int = None

        # Human-readable notes
        notes_parts: List[str] = []

        if non_null_pct is not None:
            if non_null_pct >= 99:
                notes_parts.append("almost always present")
            elif non_null_pct >= 95:
                notes_parts.append("mostly present")
            elif non_null_pct >= 80:
                notes_parts.append("some missing values")
            else:
                notes_parts.append("many missing values")

        if approx_distinct_int is not None:
            if approx_distinct_int <= 10:
                notes_parts.append("small set of categories")
            elif approx_distinct_int <= 100:
                notes_parts.append("moderate variety")
            else:
                notes_parts.append("high cardinality")

        rows.append(
            {
                "column": col_name,
                "inferred_type": type_label,
                "non_null_pct": non_null_pct,
                "approx_distinct": approx_distinct_int,
                "notes": ", ".join(notes_parts),
            }
        )

    return pd.DataFrame(rows)

def build_dq_brain_prompt(dataset_name: str, profiling_df: pd.DataFrame) -> str:
    """
    Build the system/user prompt for DQ Brain.

    tests expect:
      - dataset name (e.g. 'FACT_SALES') to appear
      - the literal token 'PROFILING_JSON' to appear in the prompt
    """
    ds = dataset_name.upper()

    # Build structured payload first (dataset + per-column metrics)
    payload = build_dq_brain_payload(ds, profiling_df)

    # IMPORTANT: make it JSON-serializable (handles pandas.Timestamp etc.)
    profiling_json = json.dumps(payload, indent=2, default=str)

    return (
        "You are a data quality expert. You will receive profiling information "
        f"for a single dataset named '{ds}'.\n"
        "The profiling information is provided as JSON under the key 'PROFILING_JSON'.\n\n"
        "PROFILING_JSON:\n"
        f"{profiling_json}\n\n"
        "From this profiling, infer a set of candidate data quality rules.\n"
        "Return a single JSON object with a top-level field 'rules', which is an array.\n"
        "Each rule object should contain:\n"
        "- ruleType (e.g. COMPLETENESS, UNIQUENESS, RANGE, PATTERN)\n"
        "- column\n"
        "- level (e.g. ERROR, WARNING)\n"
        "- threshold (for completeness or other ratio-based rules, if applicable)\n"
        "- minValue / maxValue (for range rules, if applicable)\n"
        "- allowedValues (for domain rules, if any)\n"
        "- pattern (for regex-based rules, if any)\n"
        "- explanation (short natural language justification)\n\n"
        "Now produce the JSON object with a 'rules' array. "
        "Do not include any explanatory text outside the JSON."
    )

# =========================== LLM integration ===========================

def _strip_markdown_fences(raw_text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences and surrounding noise
    that LLMs often add around JSON.
    """
    text = raw_text.strip()

    # Handle ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()

        # drop first line (``` or ```json)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        # drop trailing ``` line(s)
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()

        text = "\n".join(lines).strip()

    # As an extra safety net, if there's still junk before/after JSON,
    # try to slice from first '{' to last '}'.
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return text

def parse_llm_rules_json(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM response (JSON string) into a list of rule dicts.

    Requirements:
    - top-level key "rules" must exist
    - "rules" must be a list
    """
    cleaned = _strip_markdown_fences(raw_text)

    data = json.loads(cleaned)

    if "rules" not in data:
        raise ValueError("LLM JSON must contain top-level 'rules' key")

    rules = data["rules"]

    if not isinstance(rules, list):
        raise ValueError("Expected 'rules' to be a list in LLM response JSON")

    return rules


# =========================== Persist rules to DQ_RULES ===========================

def save_rules_to_snowflake(
    dataset_name: str,
    rules: List[Dict[str, Any]],
    created_by: str = "DQ_BRAIN",
) -> None:
    """
    Save suggested rules into DQ_DB.DQ_SCHEMA.DQ_RULES.

    Supports TWO input shapes:
    1) DQ Brain JSON:
       {
         "ruleType", "column", "level",
         "threshold", "minValue", "maxValue",
         "allowedValues", "pattern", "explanation", ...
       }

    2) NL Rules JSON / RuleSpec-style:
       {
         "type", "column", "level",
         "threshold", "min", "max",
         "allowed_values", ...
       }
    """
    if not rules:
        return

    ds = dataset_name.upper()
    ruleset_id = f"{ds}__DEFAULT"

    def _to_optional_float(val: Any) -> Optional[float]:
        """Safe float → None for invalid/NaN/inf."""
        if val is None:
            return None
        try:
            f = float(val)
        except Exception:
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    def _safe_json_dump(obj: Any) -> Optional[str]:
        """Safe JSON string; disallow NaN; on failure → None."""
        if obj is None:
            return None
        try:
            return json.dumps(obj, allow_nan=False)
        except Exception:
            return None

    conn = get_snowflake_connection()
    try:
        cur = conn.cursor()  # not context manager (for tests)

        for r in rules:
            rule_id = str(uuid.uuid4())

            # ------------------------------------------------------------------
            # 1) Normalize rule type from multiple possible keys
            # ------------------------------------------------------------------
            rule_type_raw = (
                r.get("ruleType")
                or r.get("RULE_TYPE")
                or r.get("type")
                or r.get("rule_type")
                or r.get("type_")
                or r.get("ruleType")
            )

            if not rule_type_raw:
                # This is what was failing earlier
                raise ValueError(f"Rule has no type set: {r}")

            # Apply aliases (unique → uniqueness, etc.) and upper-case for DB
            normalized = RULE_TYPE_ALIASES.get(str(rule_type_raw), str(rule_type_raw))
            rule_type_db = normalized.upper()

            # ------------------------------------------------------------------
            # 2) Other scalar fields: column, level, thresholds, ranges
            # ------------------------------------------------------------------
            column_raw = (
                r.get("column")
                or r.get("COLUMN_NAME")
                or r.get("column_name")
            )
            level_raw = r.get("level") or r.get("LEVEL")

            column_name = None if column_raw is None else str(column_raw)
            level = None if level_raw is None else str(level_raw)

            threshold = _to_optional_float(r.get("threshold"))

            # Support both minValue/maxValue and min/max (NL agent)
            min_value = _to_optional_float(
                r.get("minValue") or r.get("min") or r.get("MIN_VALUE")
            )
            max_value = _to_optional_float(
                r.get("maxValue") or r.get("max") or r.get("MAX_VALUE")
            )

            # ------------------------------------------------------------------
            # 3) Allowed values → JSON for VARIANT
            # ------------------------------------------------------------------
            allowed_values_src = (
                r.get("allowedValues")
                or r.get("allowed_values")
                or r.get("ALLOWED_VALUES")
            )
            allowed_values_json: Optional[str] = _safe_json_dump(allowed_values_src)

            # ------------------------------------------------------------------
            # 4) Params JSON: pattern / explanation / created_by
            # ------------------------------------------------------------------
            params_payload: Dict[str, Any] = {}

            pattern_val = r.get("pattern") or r.get("PATTERN")
            if pattern_val is not None:
                params_payload["pattern"] = pattern_val

            explanation_val = r.get("explanation") or r.get("EXPLANATION")
            if explanation_val is not None:
                params_payload["explanation"] = explanation_val

            if created_by:
                params_payload["created_by"] = created_by

            params_json: Optional[str] = _safe_json_dump(
                params_payload if params_payload else None
            )

            # ------------------------------------------------------------------
            # 5) Insert into DQ_RULES
            # ------------------------------------------------------------------
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
                    column_name,
                    level,
                    threshold,
                    min_value,
                    max_value,
                    allowed_values_json,  # for CASE WHEN IS NULL
                    allowed_values_json,  # for PARSE_JSON(...)
                    params_json,          # for CASE WHEN IS NULL
                    params_json,          # for PARSE_JSON(...)
                ),
            )

        conn.commit()
    finally:
        conn.close()




# def save_rules_to_snowflake(
#     dataset_name: str,
#     rules: List[Dict[str, Any]],
#     created_by: str = "DQ_BRAIN",
# ) -> None:
#     """
#     Save suggested rules into DQ_DB.DQ_SCHEMA.DQ_RULES.

#     Table:

#       DQ_RULES(
#         RULE_ID        VARCHAR NOT NULL,
#         RULESET_ID     VARCHAR NOT NULL,
#         DATASET_NAME   VARCHAR NOT NULL,
#         RULE_TYPE      VARCHAR NOT NULL,
#         COLUMN_NAME    VARCHAR,
#         LEVEL          VARCHAR,
#         THRESHOLD      FLOAT,
#         MIN_VALUE      FLOAT,
#         MAX_VALUE      FLOAT,
#         ALLOWED_VALUES VARIANT,
#         PARAMS_JSON    VARIANT,
#         CREATED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
#       )

#     RULESET_ID is set to f"{DATASET_NAME.upper()}__DEFAULT".
#     'created_by' is stored inside PARAMS_JSON["created_by"].
#     """
#     if not rules:
#         return

#     ds = dataset_name.upper()
#     ruleset_id = f"{ds}__DEFAULT"

#     def _to_optional_float(val: Any) -> Optional[float]:
#         """Safe float → None for invalid/NaN/inf."""
#         if val is None:
#             return None
#         try:
#             f = float(val)
#         except Exception:
#             return None
#         if math.isnan(f) or math.isinf(f):
#             return None
#         return f

#     def _safe_json_dump(obj: Any) -> Optional[str]:
#         """Safe JSON string; disallow NaN; on failure → None."""
#         if obj is None:
#             return None
#         try:
#             return json.dumps(obj, allow_nan=False)
#         except Exception:
#             return None

#     conn = get_snowflake_connection()
#     try:
#         cur = conn.cursor()  # not context manager (for tests)

#         for r in rules:
#             rule_id = str(uuid.uuid4())

#             # ---- Normalize scalar fields ----
#             rule_type_raw = r.get("ruleType")
#             column_raw = r.get("column")
#             level_raw = r.get("level")

#             rule_type = None if rule_type_raw is None else str(rule_type_raw)
#             column_name = None if column_raw is None else str(column_raw)
#             level = None if level_raw is None else str(level_raw)

#             threshold = _to_optional_float(r.get("threshold"))
#             min_value = _to_optional_float(r.get("minValue"))
#             max_value = _to_optional_float(r.get("maxValue"))
            
#             if not rule_type:
#                 # Fail fast with a clear error instead of sending NULL to Snowflake
#                 raise ValueError(f"Rule has no type set: {r}")

#             normalized = RULE_TYPE_ALIASES.get(str(rule_type), str(rule_type))
#             rule_type_db = normalized.upper()    # what goes into RULE_TYPE column

#             # ---- Allowed values → JSON string (for VARIANT) ----
#             allowed_values_json: Optional[str] = _safe_json_dump(
#                 r.get("allowedValues")
#             )

#             # ---- Params JSON: pattern / explanation / created_by ----
#             params_payload: Dict[str, Any] = {}
#             if r.get("pattern") is not None:
#                 params_payload["pattern"] = r["pattern"]
#             if r.get("explanation") is not None:
#                 params_payload["explanation"] = r["explanation"]
#             if created_by:
#                 params_payload["created_by"] = created_by

#             params_json: Optional[str] = _safe_json_dump(
#                 params_payload if params_payload else None
#             )

#             cur.execute(
#                 """
#                 INSERT INTO DQ_DB.DQ_SCHEMA.DQ_RULES (
#                     RULE_ID,
#                     RULESET_ID,
#                     DATASET_NAME,
#                     RULE_TYPE,
#                     COLUMN_NAME,
#                     LEVEL,
#                     THRESHOLD,
#                     MIN_VALUE,
#                     MAX_VALUE,
#                     ALLOWED_VALUES,
#                     PARAMS_JSON
#                 )
#                 SELECT
#                     %s AS RULE_ID,
#                     %s AS RULESET_ID,
#                     %s AS DATASET_NAME,
#                     %s AS RULE_TYPE,
#                     %s AS COLUMN_NAME,
#                     %s AS LEVEL,
#                     %s AS THRESHOLD,
#                     %s AS MIN_VALUE,
#                     %s AS MAX_VALUE,
#                     CASE
#                         WHEN %s IS NULL THEN NULL
#                         ELSE PARSE_JSON(%s)
#                     END AS ALLOWED_VALUES,
#                     CASE
#                         WHEN %s IS NULL THEN NULL
#                         ELSE PARSE_JSON(%s)
#                     END AS PARAMS_JSON
#                 """,
#                 (
#                     rule_id,
#                     ruleset_id,
#                     ds,
#                     rule_type_db,
#                     column_name,
#                     level,
#                     threshold,
#                     min_value,
#                     max_value,
#                     allowed_values_json,  # for CASE WHEN IS NULL
#                     allowed_values_json,  # for PARSE_JSON(...)
#                     params_json,          # for CASE WHEN IS NULL
#                     params_json,          # for PARSE_JSON(...)
#                 ),
#             )

#         conn.commit()
#     finally:
#         conn.close()