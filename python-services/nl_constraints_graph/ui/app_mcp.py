from __future__ import annotations

import os
import sys
import json
from typing import List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---- PATH FIX ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Internal imports ----
from nl_constraints_graph.mcp_client import run_dq_tool
from nl_constraints_graph.llm.llm_router import call_llm

from nl_constraints_graph.dq_brain import (
    build_dq_brain_prompt,
    parse_llm_rules_json,
    save_rules_to_snowflake,
    summarize_profiling,
)

# ---------------------------------------------------------------------
# Helper functions (UI-side only)
# ---------------------------------------------------------------------
def _convert_epoch_ms_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    If any of the given columns are integer-like epoch millis, convert them
    to pandas datetime (UTC, naive in UI).
    """
    if df is None:
        return df
    for col in cols:
        if col in df.columns:
            try:
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], unit="ms")
            except Exception:
                # Be defensive: don't break UI if conversion fails
                pass
    return df


def _auto_convert_epoch_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort conversion of numeric epoch-like columns to pandas datetime.

    - Handles integer/float columns.
    - Detects plausible epoch-in-ms (1973–2033) and epoch-in-seconds ranges.
    - Leaves everything else as-is.
    """
    if df is None:
        return df

    for col in df.columns:
        s = df[col]

        # Only look at numeric-ish columns
        if not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)):
            continue

        sample = s.dropna()
        if sample.empty:
            continue

        try:
            mn = float(sample.min())
            mx = float(sample.max())
        except Exception:
            continue

        # Epoch milliseconds ~ 1973–2033
        if 1e11 <= mn <= 2e12 and 1e11 <= mx <= 2e12:
            try:
                df[col] = pd.to_datetime(s, unit="ms")
            except Exception:
                pass
            continue

        # Epoch seconds ~ 2001–2038
        if 1e9 <= mn <= 2e9 and 1e9 <= mx <= 2e9:
            try:
                df[col] = pd.to_datetime(s, unit="s")
            except Exception:
                pass
            continue

    return df


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "exec_logs" not in st.session_state:
        st.session_state.exec_logs = ""
    if "last_dataset" not in st.session_state:
        st.session_state.last_dataset = None
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None
    if "last_apply_flag" not in st.session_state:
        st.session_state.last_apply_flag = False

def call_dq_brain_llm(prompt: str) -> str:
    """
    Wraps llm_router.call_llm so DQ Brain always gets back a JSON string.

    Handles:
    - plain string
    - dict with 'output_text' / 'text' / 'content'
    - OpenAI-style 'choices'
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        result = call_llm(messages)
    except Exception as e:
        raise RuntimeError(f"call_llm failed: {e}") from e

    # Already a string -> assume it's JSON
    if isinstance(result, str):
        return result

    # Dict -> try common keys
    if isinstance(result, dict):
        for key in ("output_text", "text", "content"):
            if key in result and isinstance(result[key], str):
                return result[key]

        # OpenAI style: choices[0].message.content
        if "choices" in result and isinstance(result["choices"], list) and result["choices"]:
            choice0 = result["choices"][0]
            if isinstance(choice0, dict):
                msg = choice0.get("message") or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content

    # If we reach here, we don't know how to extract text
    raise ValueError(f"Unsupported llm_router.call_llm result format: {type(result)} -> {result}")


# ---------------------------------------------------------------------
# Tabs / Flows
# ---------------------------------------------------------------------

def tab_nl_rules_mcp(dataset: str, self_healing: bool, apply_changes: bool):
    """
    Workflow 1: NL → Rules (dq_nl_rules_agent).
    """
    ds = dataset.upper()

    col_left, col_right = st.columns([1.1, 1])

    # ---------- LEFT: Prompt + History ----------
    with col_left:
        st.subheader("Design and refine data quality rules")

        prompt = st.text_area(
            "Instruction",
            height=120,
            placeholder=(
                "Example: Ensure CUSTOMER_ID is unique and at least 99% complete.\n"
                "Example: Check SALE_TS is not null and always within the last 5 years."
            ),
            key="mcp_agent_main_prompt",
        )

        feedback = st.text_area(
            "Optional feedback / refinement (sent along with this run)",
            height=80,
            placeholder=(
                "Example: Use (STORE_ID, CUSTOMER_ID) for uniqueness; "
                "lower completeness to 0.95; drop domain rules, etc."
            ),
            key="mcp_agent_feedback",
        )

        run_button = st.button(
            "Run NL → Rules",
            type="primary",
            key="mcp_agent_run_button",
        )

        st.markdown("### 2. Previous runs (summary)")

        if st.session_state.chat_history:
            for idx, item in enumerate(reversed(st.session_state.chat_history[-5:])):
                run_no = len(st.session_state.chat_history) - idx
                st.markdown(f"**Run #{run_no}**")
                st.markdown(f"- **Instruction:** {item['prompt']}")
                if item.get("feedback"):
                    st.markdown(f"- **Feedback:** {item['feedback']}")
                if item.get("messages"):
                    st.markdown("  • " + " | ".join(item["messages"][:3]))
                st.markdown("---")
        else:
            st.info("No previous runs yet. Submit an instruction to see history.")

    # ---------- RIGHT: Latest Run ----------
    with col_right:
        st.subheader("3. Latest result – Inferred Rules")

        if run_button:
            if not prompt.strip():
                st.warning("Please enter an instruction.")
                return

            with st.spinner("Calling dq_nl_rules_agent tool..."):
                try:
                    nl_result = run_dq_tool(
                        "dq_nl_rules_agent",
                        {
                            "dataset": ds,
                            "prompt": prompt,
                            "apply": apply_changes,
                            "feedback": feedback or None,
                            "self_healing": self_healing,
                        },
                    )
                except Exception as e:
                    st.error(f"Error calling dq_nl_rules_agent: {e}")
                    return

            # Raw payload (collapsible)
            with st.expander("Advanced: Raw NL → Rules payload", expanded=False):
                st.json(nl_result)

            # If tool itself reported an error payload, show and bail
            if isinstance(nl_result, dict) and nl_result.get("error"):
                st.error(f"NL → Rules tool error: {nl_result['error']}")
                if nl_result.get("traceback"):
                    with st.expander("Traceback from tool"):
                        st.code(nl_result["traceback"])
                return

            rules = []
            messages = []
            anomaly_messages = []
            original_rules = []

            if isinstance(nl_result, dict):
                rules = nl_result.get("rules") or nl_result.get("inferred_rules") or []
                messages = nl_result.get("messages") or nl_result.get("validation_messages") or []
                anomaly_messages = nl_result.get("anomaly_messages") or []
                original_rules = nl_result.get("original_rules") or nl_result.get("rules_before") or []

            # Update last run context + history
            st.session_state.last_dataset = ds
            st.session_state.last_prompt = prompt
            st.session_state.last_apply_flag = apply_changes

            st.session_state.chat_history.append(
                {
                    "prompt": prompt,
                    "feedback": feedback,
                    "messages": messages or ["(no messages)"],
                }
            )

            st.success("NL → Rules run completed.")

            st.markdown("#### 3.1 Messages")
            if messages:
                for msg in messages:
                    st.write(f"- {msg}")
            else:
                st.write("(none)")

            st.markdown("#### 3.2 Inferred / Refined Rules")
            if rules:
                df_rules = pd.DataFrame(rules)
                st.dataframe(df_rules, width="stretch")
            else:
                st.write("No rules returned by tool.")

            # Advanced: anomaly + original rules + JSON preview
            with st.expander("Advanced: Anomaly messages", expanded=False):
                if anomaly_messages:
                    for msg in anomaly_messages:
                        st.warning(msg)
                else:
                    st.write("(none)")

            with st.expander("Advanced: Original rules (before change)", expanded=False):
                if original_rules:
                    df_rules_orig = pd.DataFrame(original_rules)
                    st.dataframe(df_rules_orig, width="stretch")
                else:
                    st.write("No original rules returned in response.")

            with st.expander("Advanced: Rules payload preview (JSON)", expanded=False):
                if rules:
                    st.code(json.dumps(rules, indent=2), language="json")
                else:
                    st.write("No rules to show.")

            if apply_changes:
                st.info(
                    "Apply flag was sent to dq_nl_rules_agent. "
                    "Whether rules are persisted to Snowflake depends on the server implementation."
                )
            else:
                st.info(
                    "Dry-run only: the server was asked not to apply changes. "
                    "Enable 'Apply changes' in the sidebar to let the server persist rules (if implemented)."
                )


def tab_pipeline_mcp(dataset: str):
    """
    Workflow 2: DQ Spark pipeline (dq_run_spark_pipeline).
    """
    ds = dataset.upper()

    st.subheader("Run data quality pipeline")
    st.caption(f"Dataset: `{ds}`")

    run_spark_mcp_btn = st.button("Run Pipeline",
        key="mcp_run_spark_pipeline_btn",
    )

    if run_spark_mcp_btn:
        with st.spinner("Running Spark DQ jobs via dq_run_spark_pipeline..."):
            try:
                result = run_dq_tool(
                    "dq_run_spark_pipeline",
                    {"dataset": ds, "batch_date": None},
                )
            except Exception as e:
                st.session_state.exec_logs = f"Error calling dq_run_spark_pipeline: {e}"
                st.error("Failed to run Spark pipeline. Check logs below.")
            else:
                if isinstance(result, dict) and result.get("error"):
                    st.session_state.exec_logs = (result.get("logs") or "") + \
                        f"\n\n[ERROR] {result['error']}\n{result.get('traceback', '')}"
                    st.error("Spark pipeline failed inside tool. See logs below.")
                else:
                    logs = (result or {}).get("logs") or str(result)
                    st.session_state.exec_logs = logs
                    st.success("Spark DQ pipeline completed.")

    st.markdown("#### DQ pipeline logs")
    if st.session_state.exec_logs:
        with st.expander("Execution Logs", expanded=False):
            st.text_area("Logs", value=st.session_state.exec_logs, height=250)
    else:
        st.write("No logs yet. Run the pipeline to see output.")


def tab_results_mcp(dataset: str):
    """
    Explore latest DQ results (auto-loaded):
      - Profiling
      - Constraint results
      - Bad rows for failed rules
    """
    ds = dataset.upper()

    st.subheader("Latest DQ Results")
    st.caption(f"Dataset: `{ds}`")

    tab_prof, tab_verif = st.tabs(["Profiling", "Validation & Bad rows"])

    # ================== PROFILING ==================
    with tab_prof:
        with st.spinner("Loading latest profiling snapshot..."):
            try:
                prof_result = run_dq_tool("dq_latest_profiling", {"dataset": ds})
            except Exception as e:
                st.error(f"Error calling dq_latest_profiling: {e}")
                prof_result = None

        if prof_result is None:
            st.info("No profiling result returned.")
        else:
            with st.expander("Raw dq_latest_profiling payload", expanded=False):
                st.json(prof_result)

            metrics = (prof_result or {}).get("metrics") or []
            summary = (prof_result or {}).get("summary") or []

            run_ts = None
            if metrics:
                # Try to derive Run TS from first metric row
                first = metrics[0]
                run_ts = (
                    first.get("RUN_TS")
                    or first.get("RUN_TS_STR")
                    or first.get("RUN_TS_UTC")
                )

            if run_ts:
                st.markdown(f"**Latest profiling run TS:** `{run_ts}`")

            if metrics:
                df_metrics = pd.DataFrame(metrics)
                df_metrics = _convert_epoch_ms_columns(
                    df_metrics,
                    ["RUN_TS", "STARTED_AT", "FINISHED_AT"],
                )
                st.markdown("**Profiling metrics**")
                st.dataframe(df_metrics, use_container_width=True)
            else:
                st.info("No metrics returned by tool.")

            if summary:
                df_summary = pd.DataFrame(summary)
                df_summary = _convert_epoch_ms_columns(
                    df_summary,
                    ["RUN_TS", "STARTED_AT", "FINISHED_AT"],
                )
                st.markdown("**Summary view**")
                st.dataframe(df_summary, use_container_width=True)
            else:
                st.info("No summary table returned by tool.")

    # ================== VALIDATION + BAD ROWS ==================
    with tab_verif:
        with st.spinner("Loading latest validation results..."):
            try:
                verif_result = run_dq_tool("dq_latest_verification", {"dataset": ds})
            except Exception as e:
                st.error(f"Error calling dq_latest_verification: {e}")
                verif_result = None

        if verif_result is None:
            st.info("No verification result returned.")
            return

        with st.expander("Raw dq_latest_verification payload", expanded=False):
            st.json(verif_result)

        # ---- Prefer true constraint-level rows ----
        constraints = verif_result.get("constraints")
        if constraints is None:
            constraints = []  # don't silently treat 'rows' as constraints

        runs_rows = verif_result.get("rows") or []

        total_rules = verif_result.get("total_rules") or verif_result.get("totalRules")
        failed_rules = verif_result.get("failed_rules") or verif_result.get("failedRules")
        run_id = verif_result.get("run_id") or verif_result.get("runId")
        run_ts = verif_result.get("run_ts") or verif_result.get("runTs")

        # If run_id / run_ts missing at top, try to infer from constraints / runs_rows
        if not run_id:
            if constraints:
                run_id = constraints[0].get("RUN_ID") or constraints[0].get("run_id")
            elif runs_rows:
                run_id = runs_rows[0].get("RUN_ID") or runs_rows[0].get("run_id")

        if not run_ts:
            if constraints:
                run_ts = constraints[0].get("RUN_TS") or constraints[0].get("run_ts")
            elif runs_rows:
                run_ts = runs_rows[0].get("RUN_TS") or runs_rows[0].get("run_ts")

        st.markdown(
            f"**Run ID:** `{run_id or 'UNKNOWN'}` &nbsp;&nbsp; "
            f"**Run TS:** `{run_ts or 'UNKNOWN'}`"
        )

        # Top-level counters if present
        passed_rules = None
        if total_rules is not None and failed_rules is not None:
            passed_rules = total_rules - failed_rules

        c1, c2, c3 = st.columns(3)
        if total_rules is not None:
            c1.metric("Total rules", int(total_rules))
        if passed_rules is not None:
            c2.metric("Passed rules", int(passed_rules))
        if failed_rules is not None:
            c3.metric("Failed rules", int(failed_rules))

        # ---------- CASE 1: We have real constraint rows ----------
        if constraints:
            df_constraints = pd.DataFrame(constraints)
            df_constraints = _convert_epoch_ms_columns(
                df_constraints,
                ["STARTED_AT", "FINISHED_AT"],
            )

            st.markdown("**All constraint results**")
            st.dataframe(df_constraints, use_container_width=True)

            # Failed constraints (completed ones)
            if "STATUS" in df_constraints.columns:
                failed_df = df_constraints[
                    df_constraints["STATUS"].astype(str).str.upper().isin(
                        ["FAIL", "FAILED", "ERROR"]
                    )
                ]
            elif "CONSTRAINT_STATUS" in df_constraints.columns:
                failed_df = df_constraints[
                    df_constraints["CONSTRAINT_STATUS"]
                    .astype(str)
                    .str.upper()
                    .isin(["FAIL", "FAILED", "ERROR", "FAILED_CONSTRAINT"])
                ]
            else:
                failed_df = pd.DataFrame([])

            st.markdown("**Failed constraints**")
            if failed_df.empty:
                st.success("No failed constraints in latest run.")
            else:
                st.dataframe(failed_df, use_container_width=True)

                # ---- Bad rows viewer ----
                if {"RUN_ID", "RULE_ID"}.issubset(failed_df.columns):
                    st.markdown("**Bad rows**")

                    idx_options = list(range(len(failed_df)))
                    labels = [
                        f"{failed_df.iloc[i]['RULE_ID']} – "
                        f"{failed_df.iloc[i].get('COLUMN_NAME', '') or ''} "
                        f"({failed_df.iloc[i].get('CONSTRAINT_STATUS', failed_df.iloc[i].get('STATUS', ''))})"
                        for i in idx_options
                    ]

                    selected_idx = st.selectbox(
                        "Pick a failed rule",
                        options=idx_options,
                        format_func=lambda i: labels[i],
                        key="bad_rows_rule_selector",
                    )

                    if st.button("Show bad rows", key="fetch_bad_rows_btn"):
                        sel = failed_df.iloc[selected_idx]
                        sel_run_id = str(sel["RUN_ID"])
                        sel_rule_id = str(sel["RULE_ID"])

                        bad_rows_sql = f"""
SELECT *
FROM DQ_DB.DQ_SCHEMA.DQ_BAD_ROWS
WHERE RUN_ID = '{sel_run_id}'
  AND RULE_ID = '{sel_rule_id}'
LIMIT 100;
""".strip()

                        with st.spinner("Fetching bad rows from DQ_BAD_ROWS..."):
                            try:
                                bad_rows_result = run_dq_tool(
                                    "dq_query_snowflake", {"sql": bad_rows_sql}
                                )
                            except Exception as e:
                                st.error(f"Error fetching bad rows: {e}")
                            else:
                                rows = (bad_rows_result or {}).get("rows") or []
                                if rows:
                                    df_bad = pd.DataFrame(rows)
                                    df_bad = _auto_convert_epoch_like(df_bad)
                                    st.dataframe(df_bad, use_container_width=True)
                                else:
                                    st.info("No bad rows returned for this rule.")

                        with st.expander("SQL used to fetch bad rows", expanded=False):
                            st.code(bad_rows_sql, language="sql")

        # ---------- CASE 2: No constraints, but we DO have run-level rows ----------
        elif runs_rows:
            df_runs = pd.DataFrame(runs_rows)
            df_runs = _convert_epoch_ms_columns(
                df_runs,
                ["RUN_TS", "STARTED_AT", "FINISHED_AT"],
            )

            st.markdown("**Latest validation runs (no constraint rows yet)**")
            st.dataframe(df_runs, use_container_width=True)

            # Try to interpret run status
            status_col = None
            for cand in ["STATUS", "RUN_STATUS", "RUN_STATE"]:
                if cand in df_runs.columns:
                    status_col = cand
                    break

            if status_col:
                statuses = (
                    df_runs[status_col]
                    .astype(str)
                    .str.upper()
                    .value_counts()
                    .to_dict()
                )
                status_text = ", ".join(f"{k}: {v}" for k, v in statuses.items())
                st.caption(f"Run status summary → {status_text}")

                if all(s == "RUNNING" for s in statuses.keys()):
                    st.warning(
                        "Latest validation run is still RUNNING. "
                        "Constraint-level results and bad rows are not available yet."
                    )
                else:
                    st.info(
                        "No constraint-level rows were returned by the tool. "
                        "Once the server populates constraint results, they will appear here."
                    )
            else:
                st.info(
                    "Could not detect a status column on run rows. "
                    "Check dq_latest_verification server implementation."
                )

        # ---------- CASE 3: Nothing at all ----------
        else:
            st.info("No constraints or validation runs returned by the tool.")


def tab_insights(dataset: str):
    """
    Combined Insights view:
    - Latest results dashboard (profiling + validation + bad rows)
    - AI suggestions (DQ Brain)
    """
    st.subheader("Insights")
    st.caption(f"Dataset: `{dataset.upper()}`")

    results_tab, brain_tab = st.tabs(["Results dashboard", "AI suggestions"])

    with results_tab:
        tab_results_mcp(dataset)

    with brain_tab:
        tab_dq_brain_mcp(dataset)


def tab_nl_sql_mcp(dataset: str):
    # ------------------------------------------------------------------
    # 5. Natural language → SQL
    # ------------------------------------------------------------------
    st.subheader("Ask questions in natural language")

    mode = st.radio(
        "Target",
        ["DQ metadata (DQ_DB.DQ_SCHEMA)", "Sales data (SALES_DQ.PUBLIC)"],
        index=0,
        horizontal=True,
        key="nl_sql_mode",
    )

    st.markdown(
        """
Use natural language to generate SQL, then optionally run it via `dq_query_snowflake`.

- **DQ metadata** → queries `DQ_DB.DQ_SCHEMA.*` (rules, runs, bad rows, profiling).
- **Sales data** → queries `SALES_DQ.PUBLIC.*` (FACT_SALES + DIM_* tables).
"""
    )

    nl_prompt = st.text_area(
        "Describe your question in natural language",
        value="give me query to see uniqueness of sales_id on sales table",
        key="nl_sql_prompt",
        height=80,
    )

    # Keep latest generated SQL in session so we can edit/run it
    if "nl_sql_latest" not in st.session_state:
        st.session_state["nl_sql_latest"] = ""
    if "nl_sql_sql_to_run" not in st.session_state:
        st.session_state["nl_sql_sql_to_run"] = ""

    col_gen, col_run = st.columns([1, 1])

    # 5a. Generate SQL (different tool per mode)
    with col_gen:
        if st.button("Generate SQL", type="primary"):
            if not nl_prompt.strip():
                st.warning("Please enter a natural language prompt first.")
            else:
                tool_name = (
                    "dq_dqdb_nl_sql"
                    if "DQ metadata" in mode
                    else "dq_sales_nl_sql"
                )
                with st.spinner(f"Calling {tool_name}..."):
                    try:
                        result = run_dq_tool(
                            tool_name,
                            {"prompt": nl_prompt, "execute": False},
                        )
                    except Exception as e:
                        st.error(f"Error calling {tool_name}: {e}")
                    else:
                        if isinstance(result, dict) and result.get("error"):
                            st.error(
                                f"{tool_name} error: {result.get('error')}\n\n"
                                f"{result.get('traceback', '')}"
                            )
                        else:
                            sql_generated = (result or {}).get("sql", "")
                            if not sql_generated:
                                st.warning(
                                    f"{tool_name} did not return any SQL. "
                                    "Check server logs for details."
                                )
                            else:
                                st.session_state["nl_sql_latest"] = sql_generated
                                st.session_state["nl_sql_sql_to_run"] = sql_generated
                                st.success(f"SQL generated.")

    # 5b. Run the SQL
    with col_run:
        if st.button("Run SQL (dq_query_snowflake)"):
            sql_to_run = st.session_state.get("nl_sql_sql_to_run", "").strip()
            if not sql_to_run:
                st.warning("No SQL to run. Generate or paste SQL below.")
            else:
                with st.spinner("Running SQL via dq_query_snowflake..."):
                    try:
                        result = run_dq_tool(
                            "dq_query_snowflake", {"sql": sql_to_run}
                        )
                    except Exception as e:
                        st.error(f"Error calling dq_query_snowflake: {e}")
                    else:
                        if isinstance(result, dict) and result.get("error"):
                            st.error(
                                f"dq_query_snowflake error: {result.get('error')}\n\n"
                                f"{result.get('traceback', '')}"
                            )
                        else:
                            rows = (result or {}).get("rows", [])
                            st.success(
                                f"dq_query_snowflake returned {len(rows)} rows."
                            )
                            if rows:
                                df = pd.DataFrame(rows)
                                df = _auto_convert_epoch_like(df)
                                st.dataframe(df, width="stretch")
                            else:
                                st.info("Query returned 0 rows.")

    # 5c. Always show editable SQL editor (bound to session)
    st.markdown("#### SQL to run (you can edit before execution)")
    _ = st.text_area(
        "SQL to run via dq_query_snowflake",
        value=st.session_state.get("nl_sql_sql_to_run", ""),
        height=220,
        key="nl_sql_sql_to_run",
    )

    # Optional: raw generated SQL preview
    with st.expander("Raw generated SQL", expanded=False):
        st.code(st.session_state.get("nl_sql_latest", ""), language="sql")

def tab_raw_sql_mcp():
    """
    Workflow 6: Raw Snowflake SQL (dq_query_snowflake).
    """
    st.subheader("Query Snowflake")

    sql = st.text_area(
        "SQL to run via dq_query_snowflake",
        height=150,
        placeholder=(
            "Example: SELECT * FROM DQ_DB.DQ_SCHEMA.DQ_RUNS ORDER BY STARTED_AT DESC LIMIT 50"
        ),
        key="mcp_raw_sql",
    )

    if st.button("Run SQL (raw workbench)", key="mcp_run_sql_btn"):
        if not sql.strip():
            st.warning("Please enter some SQL.")
            return

        with st.spinner("Running dq_query_snowflake..."):
            try:
                result = run_dq_tool("dq_query_snowflake", {"sql": sql})
            except Exception as e:
                st.error(f"Error calling dq_query_snowflake: {e}")
                return

        with st.expander("Raw dq_query_snowflake payload", expanded=False):
            st.json(result)

        rows = (result or {}).get("rows") or []
        if rows:
            df = pd.DataFrame(rows)
            df = _auto_convert_epoch_like(df)
            st.dataframe(df, width="stretch")
        else:
            st.info("No rows returned.")

def tab_dq_brain_mcp(dataset: str):
    """
    DQ Brain (lean UI):

    - Uses latest profiling + verification internally
    - Builds a prompt for the LLM
    - Shows suggested rules and lets user approve/save
    """
    ds = dataset.upper()
    dq_dataset = ds

    st.markdown("#### AI Suggestions from latest run")

    if st.button(
        "Generate rule suggestions from latest profiling & validation",
        key="dq_brain_mcp_btn",
    ):
        # --- 1) Load profiling (no big tables shown here) ---
        try:
            with st.spinner(f"Loading latest profiling for {dq_dataset}..."):
                prof_result = run_dq_tool("dq_latest_profiling", {"dataset": dq_dataset})
        except Exception as e:
            st.error(f"Error calling dq_latest_profiling: {e}")
            return

        metrics = (prof_result or {}).get("metrics") or []
        if not metrics:
            st.warning(
                f"No profiling metrics found for dataset '{dq_dataset}'. "
                "Run the Spark profiling job first."
            )
            return

        prof_df = pd.DataFrame(metrics)
        prof_df = _convert_epoch_ms_columns(prof_df, ["RUN_TS", "STARTED_AT", "FINISHED_AT"])

        # --- 2) Load verification (only to guide suggestions) ---
        failed_df = None
        try:
            with st.spinner(f"Loading latest verification for {dq_dataset}..."):
                verif_result = run_dq_tool("dq_latest_verification", {"dataset": dq_dataset})
        except Exception as e:
            st.warning(f"Could not load verification data for '{dq_dataset}': {e}")
            verif_result = None

        if verif_result:
            constraints = verif_result.get("constraints") or verif_result.get("rows") or []
            if constraints:
                df_constraints = pd.DataFrame(constraints)
                df_constraints = _convert_epoch_ms_columns(df_constraints, ["STARTED_AT", "FINISHED_AT"])

                if "STATUS" in df_constraints.columns:
                    failed_df = df_constraints[
                        df_constraints["STATUS"].str.upper().isin(["FAIL", "FAILED", "ERROR"])
                    ].copy()
                elif "CONSTRAINT_STATUS" in df_constraints.columns:
                    failed_df = df_constraints[
                        df_constraints["CONSTRAINT_STATUS"]
                        .str.upper()
                        .isin(["FAIL", "FAILED", "ERROR", "FAILED_CONSTRAINT"])
                    ].copy()
                else:
                    failed_df = df_constraints.copy()
        # We *don't* show these tables again – only use as signal

        # Quick status summary instead of full duplication
        n_metrics = len(prof_df)
        n_failed = 0 if failed_df is None else len(failed_df)
        st.info(
            f"Using {n_metrics} profiling metric rows and {n_failed} failed constraint rows "
            f"from the latest run for AI suggestions."
        )

        # --- 3) Build prompt for LLM ---
        with st.spinner("Building prompt for DQ Brain..."):
            base_prompt = build_dq_brain_prompt(dq_dataset, prof_df)

            extra = ""
            if failed_df is not None and not failed_df.empty:
                lines = []
                for _, row in failed_df.head(30).iterrows():
                    rule_type = str(row.get("RULE_TYPE", "") or "")
                    col_name = str(row.get("COLUMN_NAME", "") or "")
                    status = str(row.get("CONSTRAINT_STATUS", row.get("STATUS", "")) or "")
                    metric_name = str(row.get("METRIC_NAME", "") or "")
                    metric_value = row.get("METRIC_VALUE", "")
                    msg = str(row.get("CONSTRAINT_MESSAGE", row.get("MESSAGE", "")) or "")
                    threshold = row.get("THRESHOLD", "")

                    line = (
                        f"- RULE_TYPE={rule_type}, COLUMN={col_name}, STATUS={status}, "
                        f"METRIC={metric_name}={metric_value}, THRESHOLD={threshold}, "
                        f"MESSAGE={msg}"
                    )
                    lines.append(line)

                failed_text = "\n".join(lines) if lines else "(no failed constraints listed)"

                extra = f"""
Additional context: here are recent FAILED constraints for dataset {dq_dataset}.
Each bullet shows the failing rule and its message/metric.

{failed_text}

Using BOTH:
- the profiling metrics described above, and
- these failed constraints,

propose improved or additional data quality rules that will reduce or eliminate these failures.

Follow the SAME JSON output format you normally use for DQ Brain rules.
Do NOT include explanations or markdown, only the JSON array of rule objects.
""".strip()

            dq_prompt = base_prompt + ("\n\n" + extra if extra else "")

        with st.expander("Prompt sent to LLM", expanded=False):
            st.code(dq_prompt)

        # --- 4) Call LLM ---
        try:
            with st.spinner("Calling LLM to suggest rules..."):
                llm_json_text = call_dq_brain_llm(dq_prompt)
        except Exception as e:
            st.error(
                "DQ Brain LLM call failed. "
                "Please check OPENAI_API_KEY / PRIMARY_MODEL / FALLBACK_MODEL in your .env."
            )
            st.code(str(e))
            return

        st.markdown("#### Raw LLM JSON (editable)")
        llm_json_text = st.text_area(
            "LLM JSON",
            value=llm_json_text,
            height=220,
            key=f"dq_brain_llm_text_mcp_{dq_dataset}",
        )

        rules = []
        if not llm_json_text.strip():
            st.info("LLM JSON is empty. Provide valid JSON above to see rule suggestions.")
        else:
            try:
                rules = parse_llm_rules_json(llm_json_text)
            except Exception as e:
                st.error(f"Could not parse LLM JSON into rules: {e}")
                rules = []

        if rules:
            st.session_state["dq_brain_rules_mcp"] = rules
            st.success(
                f"Loaded {len(rules)} suggested rules for dataset '{dq_dataset}'. "
                "Review and save below."
            )
        else:
            st.info("No rules parsed from LLM response.")

    # --- 5) Approval + save section (always visible once rules are in session) ---
    st.markdown("---")
    st.markdown("#### Suggested rules (toggle APPROVE before saving)")

    rules_key = "dq_brain_rules_mcp"
    if rules_key in st.session_state and st.session_state[rules_key]:
        rules_df = pd.DataFrame(st.session_state[rules_key])
        rules_df["APPROVE"] = True

        edited_rules_df = st.data_editor(
            rules_df,
            use_container_width=True,
            num_rows="dynamic",
            key="dq_brain_rules_editor_mcp",
        )

        if st.button("Save approved rules", key="dq_brain_save_mcp_btn"):
            approved_df = edited_rules_df[edited_rules_df["APPROVE"] == True].copy()
            if approved_df.empty:
                st.info("No rules approved; nothing to save.")
            else:
                rules_to_save = approved_df.drop(columns=["APPROVE"]).to_dict(orient="records")
                try:
                    save_rules_to_snowflake(
                        dq_dataset,
                        rules_to_save,
                        created_by="DQ_BRAIN_UI",
                    )
                    st.success(
                        f"Saved {len(rules_to_save)} rules to DQ_RULES "
                        f"for dataset '{dq_dataset}'."
                    )
                except Exception as e:
                    st.error(f"Failed to save rules to Snowflake: {e}")
    else:
        st.info("No suggested rules yet. Click the button above to generate them.")


def main():
    st.set_page_config(
        page_title="DQ NL → Rules Agent Console",
        layout="wide",
    )
    init_session_state()
    load_dotenv(override=True)

    st.title("Data Quality NL → Rules Console")
    st.caption("All data-plane calls (Spark, Snowflake) are routed through mcp.")

    # ===================== SIDEBAR =====================
    # st.sidebar.header("Configuration")
    st.sidebar.markdown("### Settings")

    dataset = st.sidebar.text_input("Dataset", value="fact_sales")
    self_healing = True
    # st.sidebar.checkbox("Enable self-healing suggestions", value=True)

    apply_changes = st.sidebar.checkbox(
        "Apply rule changes",
        value=False,
        help=(
            "If enabled, rules will be persisted."
        ),
    )

    st.sidebar.markdown("---")
    
    st.sidebar.caption("Powered by OpenAI | Snowflake | Spark | MCP | Deequ")
    
    (
        tab_rules_view,
        tab_pipeline_view,
        tab_insights_view,
        tab_nl_sql_view,
        tab_sql_workbench_view,
    ) = st.tabs(
        [
            "Rules",
            "Run Pipeline",
            "Insights",
            "Ask SQL",
            "SQL Workbench",
        ]
    )

    with tab_rules_view:
        tab_nl_rules_mcp(dataset, self_healing, apply_changes)

    with tab_pipeline_view:
        tab_pipeline_mcp(dataset)

    with tab_insights_view:
        tab_insights(dataset)

    with tab_nl_sql_view:
        tab_nl_sql_mcp(dataset)

    with tab_sql_workbench_view:
        tab_raw_sql_mcp()



if __name__ == "__main__":
    main()