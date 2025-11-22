from langgraph.graph import StateGraph, END
import os

from .models import GraphState
from .nodes_intent import intent_node
from .nodes_validate import validator_node
from .nodes_yaml import yaml_node
from .nodes_reflect import reflection_node


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("intent", intent_node)
    g.add_node("validate", validator_node)
    g.add_node("reflect", reflection_node)
    g.add_node("yaml", yaml_node)

    g.set_entry_point("intent")
    g.add_edge("intent", "validate")

    # If validation OK → YAML
    # If validation fails → reflection (unless max refinements)
    def route_after_validate(state: GraphState):
        if state.validation_ok:
            return "yaml"
        if state.refinement_attempts < state.max_refinements:
            return "reflect"
        return END

    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "yaml": "yaml",
            "reflect": "reflect",
            END: END,
        },
    )

    # After reflection → validate again
    g.add_edge("reflect", "validate")
    g.add_edge("yaml", END)

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


# =========================================================
# Optional: Export LangGraph structure as PNG for UI
# =========================================================

# from langgraph.graph.chart import draw_mermaid_png  # type: ignore

# try:
#     from langgraph.graph import draw_mermaid_png  # Newer versions
# except ImportError:
#     try:
#         from langgraph.graph.graph import draw_mermaid_png  # Some builds
#     except ImportError:
#         draw_mermaid_png = None  # Handled in function

# def export_graph_png(output_path: str | None = None) -> str:
#     """
#     Export the LangGraph workflow as a PNG. Works across multiple LangGraph releases.
#     Returns the path to the file.

#     If draw_mermaid_png is missing (older LangGraph), a .mermaid file is written instead.
#     """

#     graph = build_graph()

#     if output_path is None:
#         output_path = os.path.join(
#             os.path.dirname(__file__),
#             "nl_constraints_graph.png"
#         )

#     if draw_mermaid_png is None:
#         # Fallback: write Mermaid text instead of PNG
#         fallback_path = output_path.replace(".png", ".mermaid")
#         try:
#             with open(fallback_path, "w") as f:
#                 f.write(graph.get_graph().to_mermaid())
#         except Exception as e:
#             raise RuntimeError(f"Failed to write Mermaid file: {e}")
#         return fallback_path

#     # Standard PNG export
#     try:
#         draw_mermaid_png(graph, output_path)
#     except Exception as e:
#         raise RuntimeError(f"Failed to export PNG: {e}")

#     return output_path


