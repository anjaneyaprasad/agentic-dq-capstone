"""
High-level validation helpers + LangGraph validator node.

This is the module that tests import:

    from nl_constraints_graph import nodes_validate

Tests expect the following symbols to exist here:
- SUPPORTED_RULE_TYPES
- get_dataset_columns
- validate_rules(dataset, rules)
- detect_anomalies(dataset)
- load_latest_profile(dataset)  (they monkeypatch this)
- validator_node(state)

Internally, we delegate to Snowflake helpers in
`nl_constraints_graph.core.nodes_validate`.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple
from difflib import get_close_matches

from nl_constraints_graph.models import GraphState
from nl_constraints_graph.core.nodes_validate import (
    list_available_datasets_sf,
    get_dataset_columns_sf,
)

# -------------------------------------------------------------------
# Public constants expected by tests
# -------------------------------------------------------------------

SUPPORTED_RULE_TYPES: list[str] = [
    "COMPLETENESS",
    "UNIQUENESS",
    "IS_UNIQUE",
    "RANGE",
    "PATTERN",
    "REGEX",
    "DOMAIN",
    "completeness",
    "uniqueness",
    "is_unique",
    "range",
    "pattern",
    "regex",
]

RULE_TYPE_ALIASES = {
    # LLM-friendly names → internal canonical names
    "unique": "uniqueness",
    "uniqueness": "uniqueness",
    "completeness_threshold": "completeness",
    "completeness": "completeness",
    "DOMAIN" : "domain",
    "domain": "domain",
}

# -------------------------------------------------------------------
# Public wrappers over Snowflake helpers
# -------------------------------------------------------------------

def _append_validation_message(state: GraphState, msg: str) -> None:
    """
    Safely append a validation message to the GraphState.
    Ensures validation_messages is always a list.
    """
    msgs = list(getattr(state, "validation_messages", []) or [])
    msgs.append(msg)
    state.validation_messages = msgs


def resolve_column_name(
    requested: str,
    state: GraphState,
) -> str:
    """
    Resolve a user- or LLM-specified column name to an actual dataset column.

    Behavior:
    - If exact match exists -> return it.
    - If no match:
        - If self_healing_enabled is True:
            - pick closest match and record a healing message.
        - If self_healing_enabled is False:
            - record an error message and keep the original name (caller may mark rule ERROR).
    """
    requested_upper = (requested or "").upper()
    cols = [c.upper() for c in (state.columns or [])]

    # 1) exact match
    if requested_upper in cols:
        return requested_upper

    # 2) no exact match
    # If we have no columns at all, just complain.
    if not cols:
        _append_validation_message(
            state,
            f"Column '{requested}' does not exist for dataset {state.request.dataset}. "
            f"No columns are registered for this dataset.",
        )
        return requested_upper
    
    # find closest match (simple fuzzy)
    closest = get_close_matches(requested_upper, cols, n=1, cutoff=0.6)
    best = closest[0] if closest else None

    if state.self_healing_enabled and best:
        _append_validation_message(
            state,
            f"Column '{requested}' does not exist in dataset {state.request.dataset}. "
            f"Using closest match '{best}' (self-healing enabled).",
        )
        return best

    # non-self-healing path: do NOT substitute, just complain
    available = ", ".join(cols)
    _append_validation_message(
        state,
        f"Column '{requested}' does not exist in dataset {state.request.dataset}. "
        f"Available columns: {available}.",
    )
    
    # Let caller decide how to mark rule (likely STATUS = ERROR)
    return requested_upper

def list_available_datasets() -> List[str]:
    """Public version used at runtime; tests can monkeypatch if needed."""
    return list_available_datasets_sf()


def get_dataset_columns(dataset_name: str) -> List[str]:
    """
    Public version of column lookup.

    IMPORTANT: tests monkeypatch `nl_constraints_graph.nodes_validate.get_dataset_columns`
    to avoid hitting Snowflake. Because validate_rules() and validator_node()
    call *this* function, the monkeypatch works.
    """
    return get_dataset_columns_sf(dataset_name)


def load_latest_profile(dataset: str) -> Any:
    """
    Stub used only so tests can monkeypatch:

        monkeypatch.setattr(
            "nl_constraints_graph.nodes_validate.load_latest_profile",
            lambda dataset: {},
        )

    At runtime you could implement this to consult DQ_PROFILING_METRICS if you want
    anomaly-aware validation.
    """
    return {}


# -------------------------------------------------------------------
# Rule validation helpers (used by tests)
# -------------------------------------------------------------------


def _extract_rule_column(rule: Any) -> str | None:
    """
    Tests use DummyRule(column=..., type_=...)
    Real rules might use columnName / column.
    """
    return (
        getattr(rule, "column", None)
        or getattr(rule, "columnName", None)
    )


def _extract_rule_type(rule: Any) -> str | None:
    """
    Tests use DummyRule(type_=...)
    Real rules might use type / rule_type / ruleType.
    """
    return (
        getattr(rule, "type_", None)
        or getattr(rule, "rule_type", None)
        or getattr(rule, "ruleType", None)
        or getattr(rule, "type", None)
    )


def validate_rules(dataset: str, rules: Sequence[Any]) -> Tuple[bool, List[str]]:
    """
    Validate that:
      - rule type is supported (in SUPPORTED_RULE_TYPES)
      - referenced column exists in the dataset

    Returns (ok, messages).

    tests/nl_constraints_graph/test_nodes_validate.py calls this directly.

    NOTE: We deliberately call the *local* get_dataset_columns(), which tests
    monkeypatch, so unit tests never hit real Snowflake.
    """
    ok = True
    messages: List[str] = []

    dataset_cols = {c.upper() for c in get_dataset_columns(dataset)}

    for r in rules:
        col = _extract_rule_column(r)
        r_type = _extract_rule_type(r)

        # Normalize type for comparison
        normalized_type = RULE_TYPE_ALIASES.get(r_type, r_type)
        r_type_norm = (normalized_type or "").upper()

        if not r_type_norm or r_type_norm not in SUPPORTED_RULE_TYPES:
            ok = False
            messages.append(f"Unsupported rule type: {r_type} (normalized to '{r_type_norm}') for column {col}")
            # continue, but still check columns

        if col and col.upper() not in dataset_cols:
            ok = False
            messages.append(
                f"Column '{col}' does not exist in dataset '{dataset}'. "
                f"Known columns: {sorted(dataset_cols)}"
            )

    return ok, messages


def detect_anomalies(dataset: str) -> List[str]:
    """
    Stub anomaly detector – tests monkeypatch this:

        monkeypatch.setattr(
            "nl_constraints_graph.nodes_validate.detect_anomalies",
            lambda dataset: [],
        )

    At runtime you could implement this to look at profiling anomalies, etc.
    """
    return []


# -------------------------------------------------------------------
# LangGraph validator node (GraphState-aware)
# -------------------------------------------------------------------


def validator_node(state: GraphState) -> GraphState:
    """
    Validates inferred rules + columns and attaches messages / flags
    to the GraphState.

    tests expect, at minimum:
      - when there are no rules, validation_messages contains
        "No rules inferred to validate."
      - detect_anomalies(...) is called (they monkeypatch it)
    """
    request = state.request
    dataset = (request.dataset or "").upper()

    # Inferred rules from earlier graph steps
    rules = list(getattr(state, "inferred_rules", []) or [])

    # Existing messages (if any)
    messages: List[str] = list(getattr(state, "validation_messages", []) or [])
    anomaly_messages: List[str] = list(getattr(state, "anomaly_messages", []) or [])

    if not rules:
        messages.append("No rules inferred to validate.")
        # Run anomaly detection anyway
        anomaly_messages.extend(detect_anomalies(dataset))

        state.validation_messages = messages
        state.anomaly_messages = anomaly_messages
        # If GraphState defines validation_ok, set it to False
        if hasattr(state, "validation_ok"):
            state.validation_ok = False  # type: ignore[attr-defined]
        return state
    
    # ============ NEW: column resolution with self-healing awareness ============
    # Make sure we have the dataset columns on the state
    if not getattr(state, "columns", None):
        try:
            state.columns = get_dataset_columns(dataset)
        except Exception:
            # If this fails, we let validate_rules() below surface the error
            state.columns = []

    dataset_cols = {c.upper() for c in (state.columns or [])}
    healing_enabled = bool(getattr(state, "self_healing_enabled", False))

    for r in rules:
        col = _extract_rule_column(r)
        if not col:
            continue

        col_upper = col.upper()
        if col_upper in dataset_cols:
            # column is fine, nothing to do
            continue

        # Column is missing -> delegate to resolver (this will append messages)
        resolved = resolve_column_name(col, state)

        if healing_enabled and resolved.upper() in dataset_cols:
            # Self-healing ON: update the rule to use the resolved column
            if hasattr(r, "column"):
                setattr(r, "column", resolved)
            elif hasattr(r, "columnName"):
                setattr(r, "columnName", resolved)
        else:
            # Self-healing OFF (or no good match): keep the original name,
            # but mark rule as error if possible
            if hasattr(r, "status"):
                # do not override an existing non-empty status unless you want to
                existing = getattr(r, "status", None)
                if not existing:
                    setattr(r, "status", "ERROR")
    # ===========================================================================

    # We have some rules -> validate them against dataset metadata
    ok, rule_messages = validate_rules(dataset, rules)
    messages.extend(rule_messages)

    # Attach anomaly info
    anomaly_messages.extend(detect_anomalies(dataset))

    state.validation_messages = messages
    state.anomaly_messages = anomaly_messages
    if hasattr(state, "validation_ok"):
        state.validation_ok = ok  # type: ignore[attr-defined]

    return state

def node_validate_rules(state: GraphState) -> GraphState:
    """
    Backwards-compatible alias for validator_node.

    Older code imports:
        from nl_constraints_graph.nodes_validate import node_validate_rules

    Newer tests use validator_node(state).
    """
    return validator_node(state)