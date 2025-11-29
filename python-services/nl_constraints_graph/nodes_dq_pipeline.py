# python-services/nl_constraints_graph/nodes_dq_pipeline.py

from __future__ import annotations

from typing import List

from nl_constraints_graph.models import GraphState
from nl_constraints_graph.mcp_client import run_dq_tool


def node_run_dq_pipeline(state: GraphState) -> GraphState:
    """
    LangGraph node: run Spark Profiling + Validation for the dataset via MCP.

    Uses dq_mcp_server's tool `dq_run_spark_pipeline`.
    """
    dataset = state.request.dataset.upper()

    result = run_dq_tool("dq_run_spark_pipeline", {"dataset": dataset})

    # Extract logs (string) from MCP result
    if isinstance(result, dict):
        logs = result.get("logs", "")
    else:
        logs = str(result)

    # Attach logs / message to validation_messages so you see it in Streamlit
    msgs: List[str] = list(state.validation_messages or [])
    msgs.append(f"[DQ_PIPELINE][MCP] Ran Spark profiling + validation for {dataset}")
    if logs:
        # avoid blowing up UI – truncate to first few KB
        msgs.append(logs[:4000])

    state.validation_messages = msgs
    return state