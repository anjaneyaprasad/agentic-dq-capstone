from __future__ import annotations

import os
import sys
import subprocess
import yaml
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import traceback

# ---- PATH FIX ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Internal imports ----
from nl_constraints_graph.llm.llm_router import call_llm
from nl_constraints_graph.models import GraphState, NLRequest
from nl_constraints_graph.graph_nl_to_yaml import build_graph
from nl_constraints_graph.core.nodes_validate import get_dataset_columns

from nl_constraints_graph.rules_memory import save_interaction
from nl_constraints_graph.dq_brain import (
    load_latest_profiling,
    build_dq_brain_prompt,
    parse_llm_rules_json,
    save_rules_to_snowflake,
    summarize_profiling,
    list_profiled_datasets,
    load_profiling_summary,
    get_snowflake_connection,
)

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
SPARK_PROJECT_DIR = os.path.join(PROJECT_ROOT, "dq-spark-project")
HTML_BUILDER_DIR = os.path.join(PROJECT_ROOT, "python-services", "html_report_builder")

METRICS_DIR = os.path.join(PROJECT_ROOT, "output", "dq_metrics_all")
VERIF_DIR = os.path.join(PROJECT_ROOT, "output", "dq_verification_all")

# ------------------ Global DQ Config ------------------ #
DQ_DB = "DQ_DB"
DQ_SCHEMA = "DQ_SCHEMA"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {prompt, feedback, messages}
    if "exec_logs" not in st.session_state:
        st.session_state.exec_logs = ""
    if "last_dataset" not in st.session_state:
        st.session_state.last_dataset = None
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None
    if "last_apply_flag" not in st.session_state:
        st.session_state.last_apply_flag = False


def run_graph(dataset: str, prompt: str, apply: bool, feedback: str | None, self_healing: bool):
    """Invoke the LangGraph NL → YAML workflow and return a GraphState."""
    columns = get_dataset_columns(dataset)
    request = NLRequest(dataset=dataset, prompt=prompt, apply=apply)
    init_state = GraphState(
        request=request,
        columns=columns,
        user_feedback=feedback,
        self_healing_enabled=self_healing,
    )
    app = build_graph()
    raw_result = app.invoke(init_state)

    if isinstance(raw_result, GraphState):
        return raw_result
    return GraphState.model_validate(raw_result)


def run_cmd(cmd: list[str], cwd: str | None = None) -> str:
    """Run a shell command and capture stdout+stderr as text."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout

def load_latest_verification_run(dataset: str) -> tuple[pd.Timestamp, str, pd.DataFrame]:
    """
    Load the latest run for a dataset that has at least one row
    in DQ_CONSTRAINT_RESULTS.

    Returns:
        (run_ts, run_id, df)
    """
    ds = dataset.upper()
    conn = get_snowflake_connection()
    try:
        # 1) Find latest run that actually has constraint results
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

        # 2) All constraint results for that run
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


def run_spark_jobs(dataset: str) -> str:
    """
    Run ProfilingJob + ValidationJob for a dataset using the same pattern
    as manual spark-submit.

    Steps:
      1) sbt assembly  -> build fat JAR
      2) spark-submit  -> ProfilingJob
      3) spark-submit  -> ValidationJob
    """
    logs: list[str] = []

    jar_path = "target/scala-2.12/dq-spark-project-assembly-0.1.0-SNAPSHOT.jar"

    # 1) Build fat JAR
    logs.append("Building Spark fat JAR with sbt assembly...\n")
    logs.append(
        run_cmd(
            ["sbt", "clean", "assembly"],
            cwd=SPARK_PROJECT_DIR,
        )
    )

    # 2) ProfilingJob
    logs.append(f"\nRunning ProfilingJob for {dataset} via spark-submit...\n")
    logs.append(
        run_cmd(
            [
                "spark-submit",
                "--master",
                "local[4]",
                "--class",
                "com.anjaneya.dq.DqProfilingJob",
                "--packages",
                "net.snowflake:spark-snowflake_2.12:2.16.0-spark_3.4,net.snowflake:snowflake-jdbc:3.16.1",
                jar_path,
                dataset,
            ],
            cwd=SPARK_PROJECT_DIR,
        )
    )

    # 3) Validation/Verification job
    logs.append(f"\nRunning ValidationJob for {dataset} via spark-submit...\n")
    logs.append(
        run_cmd(
            [
                "spark-submit",
                "--master",
                "local[4]",
                "--class",
                "com.anjaneya.dq.DqValidationJob",
                "--packages",
                "net.snowflake:spark-snowflake_2.12:2.16.0-spark_3.4,net.snowflake:snowflake-jdbc:3.16.1",
                jar_path,
                dataset,
            ],
            cwd=SPARK_PROJECT_DIR,
        )
    )

    return "\n".join(logs)

def run_html_report(dataset: str) -> str:
    """Run Python HTML report builder for a dataset."""
    logs = [f"Building HTML report for {dataset}...\n"]
    logs.append(
        run_cmd(
            ["python", "build_html_report.py", "--dataset", dataset],
            cwd=HTML_BUILDER_DIR,
        )
    )

    report_dir = os.path.join(PROJECT_ROOT, "reports", dataset)
    latest_path = os.path.join(report_dir, "latest.html")
    logs.append(f"\nExpected latest report path: {latest_path}\n")

    return "\n".join(logs)


def get_latest_report_path(dataset: str) -> str:
    report_dir = os.path.join(PROJECT_ROOT, "reports", dataset)
    return os.path.join(report_dir, "latest.html")


def load_latest_report_html(dataset: str) -> str | None:
    path = get_latest_report_path(dataset)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def load_verification_data() -> pd.DataFrame | None:
    """Load full verification results from Parquet."""
    if not os.path.exists(VERIF_DIR):
        return None
    return pd.read_parquet(VERIF_DIR)


def get_datasets_from_verification() -> list[str]:
    """Get distinct dataset names from verification parquet."""
    df = load_verification_data()
    if df is None or "dataset_name" not in df.columns:
        return []
    return sorted(df["dataset_name"].unique().tolist())


def load_compact_summary(dataset: str) -> dict:
    """
    Return a compact summary for a dataset:
      - overview (counts)
      - failed constraints df
    """
    verif_df = load_verification_data()
    if verif_df is None:
        return {"overview": None, "failed": None}

    df_ds = verif_df[verif_df["dataset_name"] == dataset].copy()
    if df_ds.empty:
        return {"overview": None, "failed": None}

    total_checks = len(df_ds)
    failed_df = df_ds[df_ds["constraint_status"] != "Success"].copy()
    failed_checks = len(failed_df)
    pass_rate = 0.0 if total_checks == 0 else round(
        (total_checks - failed_checks) / total_checks * 100, 2
    )

    cols = [c for c in df_ds.columns if c in ["check", "constraint", "constraint_status", "message"]]
    if cols:
        failed_df = failed_df[cols]

    overview = {
        "total_checks": total_checks,
        "failed_checks": failed_checks,
        "pass_rate": pass_rate,
    }

    return {"overview": overview, "failed": failed_df}

def call_dq_brain_llm(prompt: str) -> str:
    """
    Wraps llm_router.call_llm so DQ Brain always gets back a JSON string.

    We try to be defensive about what call_llm returns:
    - plain string (already JSON)
    - dict with 'output_text', 'text', or 'content'
    - dict with 'choices' like OpenAI
    """
    try:
        # Common pattern: call_llm takes a single string or list of messages.
        # We treat our prompt as a single user message.
        messages = [
            {"role": "user", "content": prompt}
        ]
        result = call_llm(messages)  # adapt if your router expects something else
    except Exception as e:
        raise RuntimeError(f"call_llm failed: {e}") from e

    # Already a string -> assume it's JSON
    if isinstance(result, str):
        return result

    # Dict -> try common keys
    if isinstance(result, dict):
        # 1) Some routers put text here
        for key in ("output_text", "text", "content"):
            if key in result and isinstance(result[key], str):
                return result[key]

        # 2) OpenAI style
        if "choices" in result and isinstance(result["choices"], list):
            choice0 = result["choices"][0]
            # Chat completion style
            if isinstance(choice0, dict):
                msg = choice0.get("message") or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content

    # If we reach here, we don't know how to extract text
    raise ValueError(f"Unsupported llm_router.call_llm result format: {type(result)} -> {result}")


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="DQ NL → YAML Agent Console",
        layout="wide",
    )
    init_session_state()

    st.title("🧠 Data Quality NL → YAML Agent Console")
    st.caption("LangChain + LangGraph + Deequ + Streamlit")

    # ===================== SIDEBAR =====================
    st.sidebar.header("Configuration")

    dataset = st.sidebar.text_input("Dataset", value="fact_sales")
    self_healing = st.sidebar.checkbox("Enable self-healing suggestions", value=True)

    apply_changes = st.sidebar.checkbox(
        "Apply changes to YAML (not just dry-run)",
        value=False,
    )

    # --- .env reload + configured models ---
    if "config" not in st.session_state:
        st.session_state.config = {
            "PRIMARY_MODEL": os.getenv("PRIMARY_MODEL", "gpt-4o-mini"),
            "FALLBACK_MODEL": os.getenv("FALLBACK_MODEL", "gemini-2.0-flash"),
        }

    if st.sidebar.button("🔄 Reload .env"):
        load_dotenv(override=True)
        st.session_state.config["PRIMARY_MODEL"] = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
        st.session_state.config["FALLBACK_MODEL"] = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")
        st.sidebar.success("Reloaded .env")

    st.sidebar.markdown("**Configured models (from .env):**")
    st.sidebar.markdown(
        f"- Primary: `{st.session_state.config['PRIMARY_MODEL']}`  \n"
        f"- Fallback: `{st.session_state.config['FALLBACK_MODEL']}`"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tip:** Keep 'Apply changes' off while experimenting. "
        "Turn it on only when you're happy with the YAML."
    )

    # ===================== MAIN TABS =====================
    (
        tab_agent,
        tab_dq_brain,
        tab_pipeline,
        tab_report,
        tab_summary,
        tab_graph,
    ) = st.tabs([
        "NL → YAML Agent",
        "DQ Brain (Profiling → Rules)",
        "DQ Pipeline & Logs",
        "HTML Report Preview",
        "Multi-dataset Summary",
        "LangGraph Structure",
    ])

    # ----------------------------------------------------
    # TAB 1: NL → YAML Agent (prompt, results, refinement)
    # ----------------------------------------------------
    with tab_agent:
        col_left, col_right = st.columns([1.1, 1])

        # ---------- LEFT: Prompt + History ----------
        with col_left:
            st.subheader("1. Describe your rule in natural language")

            prompt = st.text_area(
                "Instruction",
                height=120,
                placeholder=(
                    "Example: Ensure customer_id is unique and at least 99% complete. "
                    "Store_id should only be 101, 102 and 103."
                ),
                key="main_prompt",
            )

            feedback_sidebar = st.text_area(
                "Optional inline feedback (used in same run)",
                height=80,
                placeholder=(
                    "Example: Use (store_id, customer_id) for uniqueness; "
                    "lower completeness to 0.95; drop domain rules, etc."
                ),
                key="inline_feedback",
            )

            run_button = st.button("Run Agent", type="primary", key="run_button")

            st.markdown("### 2. Conversation / Runs")
            if st.session_state.chat_history:
                for idx, item in enumerate(reversed(st.session_state.chat_history)):
                    st.markdown(f"**Run #{len(st.session_state.chat_history) - idx}**")
                    st.markdown(f"- **Prompt:** {item['prompt']}")
                    if item.get("feedback"):
                        st.markdown(f"- **Feedback:** {item['feedback']}")
                    st.markdown("**Messages:**")
                    for m in item["messages"]:
                        st.markdown(f"• {m}")
                    st.markdown("---")
            else:
                st.info("No previous runs yet. Submit an instruction to see history.")

        # ---------- RIGHT: Latest Run, Rules, YAML ----------
        with col_right:
            st.subheader("3. Latest Run – Rules & YAML")

            if run_button:
                if not prompt.strip():
                    st.warning("Please enter an instruction.")
                else:
                    with st.spinner("Running LangGraph NL → YAML workflow..."):
                        try:
                            final_state = run_graph(
                                dataset=dataset,
                                prompt=prompt,
                                apply=apply_changes,
                                feedback=feedback_sidebar or None,
                                self_healing=self_healing,
                            )
                        except Exception as e:
                            st.error(f"Error running graph: {e}")
                            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                            st.stop()

                    # Save interaction to memory
                    try:
                        save_interaction(
                            dataset=dataset,
                            prompt=prompt,
                            rules=[r.dict() for r in final_state.inferred_rules],
                            messages=final_state.validation_messages,
                        )
                    except Exception as e:
                        st.warning(f"Could not save interaction to memory: {e}")

                    # Update last run context
                    st.session_state.last_dataset = dataset
                    st.session_state.last_prompt = prompt
                    st.session_state.last_apply_flag = apply_changes

                    # Add to chat history
                    history_entry = {
                        "prompt": prompt,
                        "feedback": feedback_sidebar,
                        "messages": final_state.validation_messages or ["(no messages)"],
                    }
                    st.session_state.chat_history.append(history_entry)

                    st.success("Agent run completed.")

                    st.markdown("#### Validation & Refinement Messages")
                    if final_state.validation_messages:
                        for msg in final_state.validation_messages:
                            st.write(f"- {msg}")
                    else:
                        st.write("(none)")

                    st.markdown("#### Anomaly Messages")
                    if final_state.anomaly_messages:
                        for msg in final_state.anomaly_messages:
                            st.warning(msg)
                    else:
                        st.write("(none)")

                    st.markdown("#### Inferred / Refined Rules")
                    if final_state.inferred_rules:
                        df_rules = pd.DataFrame([r.dict() for r in final_state.inferred_rules])
                        st.dataframe(df_rules, width="stretch") # use_container_width=True)
                    else:
                        st.write("No rules inferred.")

                    st.markdown("#### Original Rules (from YAML / previous state)")
                    if getattr(final_state, "rules", None):
                        df_rules_orig = pd.DataFrame([r.dict() for r in final_state.rules])
                        st.dataframe(df_rules_orig, width="stretch") # use_container_width=True)
                    else:
                        st.write("No original rules in state.")

                    with st.expander("🔍 Debug: Raw GraphState"):
                        try:
                            st.json(final_state.model_dump())
                        except Exception:
                            st.write(final_state)

                    st.markdown("#### YAML Preview")
                    if final_state.merged_yaml:
                        st.code(
                            yaml.safe_dump(final_state.merged_yaml, sort_keys=False),
                            language="yaml",
                        )
                    else:
                        st.write("No YAML generated (validation failed or no rules).")

                    if apply_changes:
                        if final_state.yaml_path:
                            st.success(f"YAML file updated at: {final_state.yaml_path}")
                        else:
                            st.warning(
                                "Apply was requested, but YAML path is empty "
                                "(likely validation failed)."
                            )
                    else:
                        st.info(
                            "Dry-run only: YAML file was not written "
                            "(use 'Apply changes' to persist)."
                        )
            else:
                st.info("Enter an instruction on the left and click **Run Agent** to see results here.")

        # ---------- Refinement (still part of Agent tab) ----------
        st.markdown("---")
        st.subheader("4. Refine Current Rules with Feedback")

        refine_feedback = st.text_area(
            "Feedback for refinement (applies to last prompt)",
            height=100,
            placeholder=(
                "Example: Use (store_id, customer_id) for uniqueness; "
                "change completeness to 0.95; remove domain on store_id, etc."
            ),
            key="refine_feedback",
        )

        refine_button = st.button("Run Refinement", type="secondary", key="refine_button")

        if refine_button:
            if not st.session_state.last_prompt or not st.session_state.last_dataset:
                st.warning("No previous run to refine. Run the agent first.")
            elif not refine_feedback.strip():
                st.warning("Please enter some feedback to apply.")
            else:
                with st.spinner("Running refinement with feedback..."):
                    try:
                        refined_state = run_graph(
                            dataset=st.session_state.last_dataset,
                            prompt=st.session_state.last_prompt,
                            apply=apply_changes,
                            feedback=refine_feedback,
                            self_healing=self_healing,
                        )
                    except Exception as e:
                        st.error(f"Error running refinement: {e}")
                    else:
                        try:
                            save_interaction(
                                dataset=st.session_state.last_dataset,
                                prompt=st.session_state.last_prompt,
                                rules=[r.dict() for r in refined_state.inferred_rules],
                                messages=refined_state.validation_messages,
                            )
                        except Exception as e:
                            st.warning(f"Could not save refinement interaction to memory: {e}")

                        history_entry = {
                            "prompt": st.session_state.last_prompt,
                            "feedback": refine_feedback,
                            "messages": refined_state.validation_messages or ["(no messages)"],
                        }
                        st.session_state.chat_history.append(history_entry)

                        st.success("Refinement completed.")

                        st.markdown("#### Refinement Messages")
                        if refined_state.validation_messages:
                            for msg in refined_state.validation_messages:
                                st.write(f"- {msg}")
                        else:
                            st.write("(none)")

                        st.markdown("#### Refined Rules")
                        if refined_state.inferred_rules:
                            df_refined = pd.DataFrame(
                                [r.dict() for r in refined_state.inferred_rules]
                            )
                            st.dataframe(df_refined, width="stretch") # use_container_width=True)
                        else:
                            st.write("No rules inferred after refinement.")

                        st.markdown("#### Refined YAML Preview")
                        if refined_state.merged_yaml:
                            st.code(
                                yaml.safe_dump(refined_state.merged_yaml, sort_keys=False),
                                language="yaml",
                            )
                        else:
                            st.write("No YAML generated (refinement validation failed).")

                        if apply_changes:
                            if refined_state.yaml_path:
                                st.success(f"YAML file updated at: {refined_state.yaml_path}")
                            else:
                                st.warning(
                                    "Apply was requested, but YAML path is empty "
                                    "(likely refinement validation failed)."
                                )
                        else:
                            st.info("Refinement run was dry-run only; YAML not written.")

    # ----------------------------------------------------
    # TAB 2: DQ Brain – Profiling → Rule suggestions
    # ----------------------------------------------------
    with tab_dq_brain:
        st.subheader("🧠 DQ Brain – Suggest Rules from Profiling")

        dq_dataset = dataset.upper()

        if st.button("Load profiling & generate DQ rule suggestions", key="dq_brain_btn"):
            try:
                with st.spinner(f"Loading latest profiling metrics for {dq_dataset} from Snowflake..."):
                    prof_df = load_latest_profiling(dq_dataset)

                if prof_df.empty:
                    st.warning(f"No profiling metrics found for dataset '{dq_dataset}'. Run ProfilingJob first.")
                    st.stop()

                st.markdown("**Latest profiling snapshot**")
                st.dataframe(prof_df, width="stretch")

                pretty = summarize_profiling(prof_df)
                st.markdown("**Human-readable profiling summary**")
                st.dataframe(pretty, width="stretch")


                with st.spinner("Building DQ Brain prompt..."):
                    dq_prompt = build_dq_brain_prompt(dq_dataset, prof_df)

                with st.expander("🔍 Prompt sent to DQ Brain (LLM)"):
                    st.code(dq_prompt)

                # ONLINE ONLY: require LLM to succeed, no dummy fallback
                try:
                    with st.spinner("Calling LLM to suggest rules..."):
                        llm_json_text = call_dq_brain_llm(dq_prompt)
                except Exception as e:
                    st.error(
                        "DQ Brain LLM call failed. "
                        "Please check OPENAI_API_KEY / PRIMARY_MODEL / FALLBACK_MODEL in your .env."
                    )
                    st.code(str(e))
                    st.stop()

                st.markdown("#### Raw LLM JSON (you can edit before parsing)")
                llm_json_text = st.text_area(
                    "LLM JSON",
                    value=llm_json_text,
                    height=220,
                    key=f"dq_brain_llm_text_{dq_dataset}",
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
                    rules_df = pd.DataFrame(rules)
                    rules_df["APPROVE"] = True

                    st.markdown("#### Suggested Rules (toggle APPROVE before saving)")
                    edited_rules_df = st.data_editor(
                        rules_df,
                        width="stretch", # use_container_width=True,
                        num_rows="dynamic",
                        key="dq_brain_rules_editor",
                    )

                    if st.button("💾 Save APPROVED rules to DQ_RULES", key="dq_brain_save_btn"):
                        approved_df = edited_rules_df[edited_rules_df["APPROVE"] == True].copy()
                        if approved_df.empty:
                            st.info("No rules approved; nothing to save.")
                        else:
                            rules_to_save = approved_df.drop(columns=["APPROVE"]).to_dict(orient="records")
                            try:
                                save_rules_to_snowflake(dq_dataset, rules_to_save, created_by="DQ_BRAIN_STREAMLIT")
                                st.success(f"Saved {len(rules_to_save)} rules to DQ_RULES for dataset '{dq_dataset}'.")
                            except Exception as e:
                                st.error(f"Failed to save rules to Snowflake: {e}")
                else:
                    st.info("No rules parsed from LLM response.")
            except Exception as e:
                st.error(f"Error in DQ Brain flow: {e}")
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

    # ----------------------------------------------------
    # TAB 3: DQ Pipeline (Spark + HTML builder) & Logs
    # ----------------------------------------------------
    with tab_pipeline:
        st.subheader("5. Execute DQ Pipeline for This Dataset")

        run_spark_btn = st.button("Run Spark Profiling + Verification", key="run_spark")
        run_report_btn = st.button("Build HTML Report", key="run_report")

        if run_spark_btn:
            with st.spinner("Running Spark DQ jobs via sbt..."):
                try:
                    logs = run_spark_jobs(dataset)
                    st.session_state.exec_logs = logs
                    st.success("Spark jobs completed.")
                except Exception as e:
                    st.session_state.exec_logs = f"Error running Spark jobs: {e}"
                    st.error("Failed to run Spark jobs. Check logs below.")

        if run_report_btn:
            with st.spinner("Running HTML report builder..."):
                try:
                    logs = run_html_report(dataset)
                    st.session_state.exec_logs = logs
                    st.success("HTML report built.")
                except Exception as e:
                    st.session_state.exec_logs = f"Error running HTML report builder: {e}"
                    st.error("Failed to build HTML report. Check logs below.")

        st.markdown("#### Execution Logs")
        if st.session_state.exec_logs:
            st.text_area(
                "Logs",
                value=st.session_state.exec_logs,
                height=250,
            )
        else:
            st.write("No execution logs yet. Run Spark or HTML jobs to see output.")

    # ----------------------------------------------------
    # TAB 4: Latest HTML Report preview
    # ----------------------------------------------------
    with tab_report:
        st.subheader("6. DQ Results Explorer")

        sub_tab_profiling, sub_tab_results, sub_tab_bad_rows, sub_tab_html = st.tabs(
            ["Profiling", "Constraint Results", "Bad Rows", "HTML Report"]
        )

        dq_dataset = dataset.upper()

        # ---------- Profiling ----------
        with sub_tab_profiling:
            st.markdown("### Latest Profiling Snapshot")
            try:
                prof_df = load_latest_profiling(dq_dataset)
                if prof_df.empty:
                    st.info(
                        f"No profiling metrics found for dataset '{dq_dataset}'. "
                        "Run the Spark profiling job first."
                    )
                else:
                    st.markdown("**Raw profiling metrics (long form)**")
                    st.dataframe(prof_df, use_container_width=True)

                    pretty = summarize_profiling(prof_df)
                    st.markdown("**Human-readable profiling summary**")
                    st.dataframe(pretty, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading profiling metrics: {e}")
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

        # ---------- Constraint Results ----------
        with sub_tab_results:
            st.markdown("### Latest Constraint Results")
            if st.button("Reload latest run", key="reload_verif"):
                st.experimental_rerun()

            try:
                run_ts, run_id, verif_df = load_latest_verification_run(dataset)
                st.markdown(
                    f"**Latest run:** `{run_id}` &nbsp;&nbsp; "
                    f"**Dataset:** `{dq_dataset}` &nbsp;&nbsp; "
                    f"**Started at:** `{run_ts}`"
                )

                total_checks = len(verif_df)
                failed_checks = (verif_df["CONSTRAINT_STATUS"] != "Success").sum()
                passed_checks = total_checks - failed_checks

                total_rows = verif_df.get("TOTAL_ROWS").iloc[0] if "TOTAL_ROWS" in verif_df.columns else None
                failed_rows = verif_df.get("FAILED_ROWS").iloc[0] if "FAILED_ROWS" in verif_df.columns else None

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total checks", total_checks)
                c2.metric("Passed checks", int(passed_checks))
                c3.metric("Failed checks", int(failed_checks))
                if total_rows is not None:
                    c4.metric("Rows scanned", int(total_rows))

                st.markdown("**All constraint results**")
                st.dataframe(verif_df, use_container_width=True)

                st.markdown("**Only failed constraints**")
                failed_df = verif_df[verif_df["CONSTRAINT_STATUS"] != "Success"].copy()
                if failed_df.empty:
                    st.success("No failed constraints 🎉")
                else:
                    cols = [
                        "RULE_TYPE",
                        "COLUMN_NAME",
                        "CONSTRAINT_STATUS",
                        "CONSTRAINT_MESSAGE",
                        "METRIC_NAME",
                        "METRIC_VALUE",
                    ]
                    cols = [c for c in cols if c in failed_df.columns]
                    if cols:
                        st.dataframe(failed_df[cols], use_container_width=True)
                    else:
                        st.dataframe(failed_df, use_container_width=True)

                # Keep latest run info in session for Bad Rows tab
                st.session_state["last_verif_run_id"] = run_id
                st.session_state["last_verif_dataset"] = dq_dataset

            except Exception as e:
                st.error(f"Error loading verification results: {e}")
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

        # ---------- Bad Rows ----------
        with sub_tab_bad_rows:
            st.markdown("### Sample Bad Rows (DQ_BAD_ROWS)")
            run_id = st.session_state.get("last_verif_run_id")
            ds_for_run = st.session_state.get("last_verif_dataset", dq_dataset)

            if not run_id:
                st.info(
                    "No run selected yet. Load constraint results first from the "
                    "**Constraint Results** tab."
                )
            else:
                try:
                    bad_df = load_bad_rows_for_run(ds_for_run, run_id)
                    if bad_df.empty:
                        st.success("No bad rows found for the latest run 🎉")
                    else:
                        # Show a limited sample for usability
                        st.dataframe(bad_df.head(200), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading bad rows: {e}")
                    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

        # ---------- HTML Report ----------
        with sub_tab_html:
            st.markdown("### HTML Report Preview")
            if st.button("Build & Load Latest HTML Report", key="load_report"):
                # Trigger HTML builder script via the same helper used in the Pipeline tab
                try:
                    logs = run_html_report(dataset)
                    st.text_area("HTML Builder Logs", value=logs, height=180)
                except Exception as e:
                    st.error(f"Failed to build HTML report: {e}")

                html_content = load_latest_report_html(dataset)
                if html_content:
                    st.success(f"Loaded report for dataset '{dataset}'.")
                    components.html(html_content, height=800, scrolling=True)
                else:
                    st.warning(
                        f"No latest.html found for dataset '{dataset}'. "
                        "Run 'Build HTML Report' from the Pipeline tab first."
                    )

    # ----------------------------------------------------
    # TAB 5: Compact Multi-dataset Summary (Profiling-based)
    # ----------------------------------------------------
    with tab_summary:
        st.subheader("7. Compact Multi-Dataset View (Profiling Metrics)")

        all_ds = list_profiled_datasets()
        if not all_ds:
            st.info("No profiling data found yet. Run ProfilingJob first.")
        else:
            tabs = st.tabs(all_ds)

            for ds_name, inner_tab in zip(all_ds, tabs):
                with inner_tab:
                    st.markdown(f"##### Dataset: `{ds_name}`")

                    try:
                        summary_df = load_profiling_summary(ds_name)
                    except Exception as e:
                        st.error(f"Failed to load profiling summary for {ds_name}: {e}")
                        continue

                    if summary_df is None or summary_df.empty:
                        st.write("No profiling metrics available for this dataset.")
                        continue

                    # Show per-column metrics
                    st.markdown("**Per-column profiling metrics**")
                    st.dataframe(summary_df, use_container_width=True)

    # ----------------------------------------------------
    # TAB 6: NL → YAML Agent Graph (LangGraph)
    # ----------------------------------------------------
    with tab_graph:
        st.subheader("8. NL → YAML Agent Graph")

        if st.button("Render LangGraph", key="render_graph"):
            with st.spinner("Rendering LangGraph structure..."):
                try:
                    app = build_graph()
                    g = app.get_graph()

                    raw_mermaid = g.draw_mermaid()
                    cleaned = raw_mermaid
                    idx = cleaned.find("graph ")
                    if idx != -1:
                        cleaned = cleaned[idx:]

                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                    <meta charset="UTF-8" />
                    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                    <script>
                        mermaid.initialize({{ startOnLoad: true, securityLevel: "loose" }});
                    </script>
                    </head>
                    <body>
                    <div class="mermaid">
                    {cleaned}
                    </div>
                    </body>
                    </html>
                    """

                    components.html(html, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to render graph: {e}")

if __name__ == "__main__":
    main()