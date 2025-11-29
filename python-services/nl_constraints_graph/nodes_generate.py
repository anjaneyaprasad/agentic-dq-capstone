"""
Backwards-compat shim for rule generation.

Older code expects:
    from nl_constraints_graph.nodes_generate import node_generate_rules

For now, this provides a minimal implementation that just leaves the
GraphState unchanged. You can later wire this to the real generation
logic (LLM, templates, etc.).
"""

from __future__ import annotations

from typing import Any
from nl_constraints_graph.models import GraphState


def node_generate_rules(state: GraphState) -> GraphState:
    """
    Placeholder LangGraph node that is supposed to generate/infer rules
    from the NL request.

    For now:
      - ensures `inferred_rules` exists (list)
      - adds a debug message into `generation_messages` if present
    """
    # Ensure inferred_rules exists
    if getattr(state, "inferred_rules", None) is None:
        state.inferred_rules = []  # type: ignore[attr-defined]

    # Optional: attach a message so you can see it fired
    msgs = list(getattr(state, "generation_messages", []) or [])
    msgs.append("node_generate_rules placeholder: no rules generated.")
    state.generation_messages = msgs  # type: ignore[attr-defined]

    return state
