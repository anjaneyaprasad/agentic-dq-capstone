from __future__ import annotations
from typing import List
import json

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .llm_client import get_llm
from .models import GraphState, RuleSpec
from .nodes_validate import resolve_column_name, get_dataset_columns



class RulesList(BaseModel):
    """Wrapper for refined rules so PydanticOutputParser can work."""
    rules: List[RuleSpec]


def reflection_node(state: GraphState) -> GraphState:
    """
    Agent R: Refine rules when validation failed, using validation messages
    and optional user feedback.
    """

    # avoid infinite loops
    if state.refinement_attempts >= state.max_refinements:
        state.validation_messages.append(
            "Max refinements reached; not attempting further corrections."
        )
        return state

    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=RulesList)
    
    anomaly_text = "\n".join(state.anomaly_messages) if state.anomaly_messages else "None"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a data quality rule assistant. You are given:\n"
                    "- Dataset name\n"
                    "- Column names\n"
                    "- The user's original instruction\n"
                    "- Previously inferred rules\n"
                    "- Validation error messages\n"
                    "- Anomaly messages from recent profiling runs\n"
                    "- Optional user feedback\n\n"
                    "Your job is to refine or soften the rules so that they:\n"
                    "- Align with the user's intent\n"
                    "- Do not contradict the observed data/profile\n"
                    "- Reduce false positives while still enforcing good data quality.\n"
                    "Return only a JSON object matching the RulesList schema."
                ),
            ),
            (
                "user",
                (
                    "Dataset: {dataset}\n"
                    "Columns: {columns}\n\n"
                    "Original instruction:\n{instruction}\n\n"
                    "Previous rules (JSON):\n{rules_json}\n\n"
                    "Validation errors:\n{validation_errors}\n\n"
                    "Anomaly messages:\n{anomalies}\n\n"
                    "User feedback (may be empty):\n{user_feedback}\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )

    rules_json = json.dumps([r.dict() for r in state.inferred_rules], indent=2)
    validation_errors = "\n".join(state.validation_messages) or "None"
    format_instructions = parser.get_format_instructions()

    chain = prompt | llm | parser

    try:
        result: RulesList = chain.invoke(
            {
                "dataset": state.request.dataset,
                "columns": ", ".join(state.columns),
                "instruction": state.request.prompt,
                "rules_json": rules_json,
                "validation_errors": validation_errors,
                "anomalies": anomaly_text,
                "user_feedback": state.user_feedback or "",
                "format_instructions": format_instructions,
            }
        )
        new_rules = result.rules
    except Exception as e:
        state.validation_messages.append(f"Reflection failed: {e}")
        # Do not change inferred_rules if reflection fails
        return state
    
    # ---- NEW: column resolution with self-healing awareness ----
    dataset = (state.request.dataset or "").upper()

    if not getattr(state, "columns", None):
        try:
            state.columns = get_dataset_columns(dataset)
        except Exception:
            state.columns = []

    dataset_cols = {c.upper() for c in (state.columns or [])}
    healing_enabled = bool(getattr(state, "self_healing_enabled", False))

    for r in new_rules:
        raw_col = (r.column or "").strip()
        if not raw_col:
            continue

        if raw_col.upper() in dataset_cols:
            continue

        resolved = resolve_column_name(raw_col, state)

        if healing_enabled and resolved.upper() in dataset_cols:
            r.column = resolved
        else:
            if hasattr(r, "level") and not getattr(r, "level", None):
                r.level = "ERROR"
    # ------------------------------------------------------------


    state.inferred_rules = new_rules
    state.refinement_attempts += 1
    state.validation_ok = True  # let validator re-check these
    state.validation_messages.append(
        f"Reflection attempt {state.refinement_attempts} produced {len(new_rules)} rule(s)."
    )
    return state
