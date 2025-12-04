from __future__ import annotations

import os
from langgraph.graph import StateGraph, END

from .models import GraphState
from .nodes_intent import intent_node
from .nodes_validate import validator_node
from .nodes_yaml import yaml_node as finalize_rules_node
from .nodes_reflect import reflection_node
from .nodes_dq_pipeline import node_run_dq_pipeline

def build_graph():
    """
    Build the LangGraph for NL → rules:

      intent → validate → (finalize_rules | reflect → validate → ...)
    """
    g = StateGraph(GraphState)

    g.add_node("intent", intent_node)
    g.add_node("validate", validator_node)
    g.add_node("reflect", reflection_node)
    g.add_node("finalize_rules", finalize_rules_node)

    g.set_entry_point("intent")
    g.add_edge("intent", "validate")
    
    # If validation OK → finalize rules
    # If validation fails → reflection (unless max refinements)
    def route_after_validate(state: GraphState):
        if state.validation_ok:
            return "finalize_rules"
        if state.refinement_attempts < state.max_refinements:
            return "reflect"
        return END

    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "finalize_rules": "finalize_rules",
            "reflect": "reflect",
            END: END,
        },
    )

    # After reflection → validate again
    g.add_edge("reflect", "validate")
    g.add_edge("finalize_rules", END)

    return g.compile()


import os

def export_graph_png(output_path: str | None = None) -> str:
    """
    Export the LangGraph workflow as a PNG if possible.
    Fallback: save Mermaid text if PNG export is not supported.

    Returns:
        Path to the generated file (.png or .mermaid).
    """
    app = build_graph()
    g = app.get_graph()

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__),
            "nl_constraints_graph.png"
        )

    # 1) Try GraphViz via draw()
    try:
        dot = g.draw()  # often returns graphviz.Digraph
    except Exception:
        dot = None

    if dot is not None and hasattr(dot, "render"):
        # If user asked for .png, strip extension for render() base
        base = output_path[:-4] if output_path.endswith(".png") else output_path
        dot.render(base, format="png", cleanup=True)
        return base + ".png"

    # 2) Fallback: Mermaid text via draw_mermaid()
    try:
        mermaid = g.draw_mermaid()
    except Exception as e:
        raise RuntimeError(f"Cannot export graph: {e}")

    if output_path.endswith(".png"):
        mermaid_path = output_path[:-4] + ".mermaid"
    else:
        mermaid_path = output_path

    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid)

    return mermaid_path