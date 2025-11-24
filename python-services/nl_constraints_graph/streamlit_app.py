import os
import sys
import subprocess
import yaml
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from llm_router import call_llm
import traceback

# ---------------------------------------------------------------------
# Path + imports setup
# ---------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

load_dotenv()
# (override=True)

from nl_constraints_graph.models import GraphState, NLRequest  # type: ignore
from nl_constraints_graph.graph_nl_to_yaml import build_graph  # type: ignore
from nl_constraints_graph.nodes_validate import get_dataset_columns  # type: ignore
from nl_constraints_graph.rules_memory import save_interaction  # type: ignore
# from nl_constraints_graph.graph_nl_to_yaml import export_graph_png  # type: ignore

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SPARK_PROJECT_DIR = os.path.join(PROJECT_ROOT, "dq-spark-project")
HTML_BUILDER_DIR = os.path.join(PROJECT_ROOT, "python-services", "html_report_builder")

METRICS_DIR = os.path.join(PROJECT_ROOT, "output", "dq_metrics_all")
VERIF_DIR = os.path.join(PROJECT_ROOT, "output", "dq_verification_all")


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


def run_spark_jobs(dataset: str) -> str:
    """Run ProfilingJob + VerificationJob for a dataset via sbt."""
    logs = []

    logs.append(f"Running ProfilingJob for {dataset}...\n")
    logs.append(
        run_cmd(
            ["sbt", f"""runMain com.anjaneya.dq.jobs.ProfilingJob {dataset}"""],
            cwd=SPARK_PROJECT_DIR,
        )
    )

    logs.append(f"\nRunning VerificationJob for {dataset}...\n")
    logs.append(
        run_cmd(
            ["sbt", f"""runMain com.anjaneya.dq.jobs.VerificationJob {dataset}"""],
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
    
    # self healing checkbox (not used in code yet)
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
        # reload .env and override existing env vars
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

    
    
    
    
    

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tip:** Keep 'Apply changes' off while experimenting. "
        "Turn it on only when you're happy with the YAML."
    )

    # ===================== MAIN LAYOUT =====================
    col_left, col_right = st.columns([1.1, 1])

    # ---------------- LEFT: PROMPT + HISTORY ----------------
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

    # ---------------- RIGHT: RESULTS, REFINEMENT, EXECUTION ----------------
    with col_right:
        st.subheader("3. Latest Run – Rules & YAML")

        if run_button:
            if not prompt.strip():
                st.warning("Please enter an instruction.")
            else:
                with st.spinner("Running LangGraph NL → YAML workflow..."):
                    # with st.spinner("Thinking..."):
                    #     result = call_llm(messages)
                    #     # Sidebar: show which model actually answered
                    #     st.sidebar.subheader("⚙️ Model Info")
                    #     st.sidebar.markdown(
                    #     f"**Active model:** `{result['model']}`  \n"
                    #     f"**Provider:** `{result['provider']}`"
                    #     )
            
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
                        return

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

                # Validation / refinement messages
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

                # Inferred rules table
                st.markdown("#### Inferred / Refined Rules")
                if final_state.inferred_rules:
                    df_rules = pd.DataFrame([r.dict() for r in final_state.inferred_rules])
                    st.dataframe(df_rules, use_container_width=True)
                else:
                    st.write("No rules inferred.")
                    
                 
                
                st.markdown("#### Original Rules (from YAML / previous state)")
                if getattr(final_state, "rules", None):
                    df_rules_orig = pd.DataFrame([r.dict() for r in final_state.rules])
                    st.dataframe(df_rules_orig, use_container_width=True)
                else:
                    st.write("No original rules in state.")    
                    
                with st.expander("🔍 Debug: Raw GraphState"):
                    try:
                        st.json(final_state.model_dump())
                    except Exception:
                        st.write(final_state)
                 
                    

                # YAML preview
                st.markdown("#### YAML Preview")
                if final_state.merged_yaml:
                    st.code(
                        yaml.safe_dump(final_state.merged_yaml, sort_keys=False),
                        language="yaml",
                    )
                else:
                    st.write("No YAML generated (validation failed or no rules).")

                # Apply info
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

        # ---------- 4. Refinement with feedback ----------
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
                        # Save refinement interaction to memory
                        try:
                            save_interaction(
                                dataset=st.session_state.last_dataset,
                                prompt=st.session_state.last_prompt,
                                rules=[r.dict() for r in refined_state.inferred_rules],
                                messages=refined_state.validation_messages,
                            )
                        except Exception as e:
                            st.warning(f"Could not save refinement interaction to memory: {e}")

                        # Add to chat history
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
                            st.dataframe(df_refined, use_container_width=True)
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

        # ---------- 5. Execute DQ pipeline ----------
        st.markdown("---")
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

        # ---------- 6. Latest HTML report preview ----------
        st.markdown("---")
        st.subheader("6. Latest HTML Report Preview")

        if st.button("Load Latest Report", key="load_report"):
            html_content = load_latest_report_html(dataset)
            if html_content:
                st.success(f"Loaded report for dataset '{dataset}'.")
                components.html(html_content, height=800, scrolling=True)
            else:
                st.warning(
                    f"No latest.html found for dataset '{dataset}'. "
                    "Run 'Build HTML Report' first."
                )

        # ---------- 7. Compact multi-dataset view ----------
        st.markdown("---")
        st.subheader("7. Compact Multi-Dataset View (All Datasets)")

        all_ds = get_datasets_from_verification()
        if not all_ds:
            st.info("No verification data found yet. Run Spark jobs first.")
        else:
            tabs = st.tabs(all_ds)

            for ds_name, tab in zip(all_ds, tabs):
                with tab:
                    summary = load_compact_summary(ds_name)
                    overview = summary["overview"]
                    failed_df = summary["failed"]

                    st.markdown(f"##### Dataset: `{ds_name}`")

                    if overview is None:
                        st.write("No verification data available for this dataset.")
                        continue

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Checks", overview["total_checks"])
                    c2.metric("Failed Checks", overview["failed_checks"])
                    c3.metric("Pass Rate (%)", overview["pass_rate"])

                    if overview["failed_checks"] == 0:
                        st.success("All constraints passed ✅")
                    else:
                        st.error(f"{overview['failed_checks']} constraint(s) failed ❌")

                    st.markdown("**Failed Constraints (if any)**")
                    if failed_df is not None and not failed_df.empty:
                        st.dataframe(failed_df, use_container_width=True)
                    else:
                        st.write("No failed constraints.")
                    
        # ---------- 8. NL → YAML Agent Graph ----------
        st.markdown("---")
        st.subheader("8. NL → YAML Agent Graph")

        if st.button("Render LangGraph", key="render_graph"):
            with st.spinner("Rendering LangGraph structure..."):
                try:
                    app = build_graph()
                    g = app.get_graph()

                    # 1) Get mermaid source from LangGraph
                    raw_mermaid = g.draw_mermaid()

                    # 2) Clean it: keep only from 'graph ' onwards (drop leading config header)
                    cleaned = raw_mermaid
                    idx = cleaned.find("graph ")
                    if idx != -1:
                        cleaned = cleaned[idx:]

                    # 3) Build HTML with mermaid.js
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