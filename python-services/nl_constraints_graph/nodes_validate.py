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
    "completeness",
    "uniqueness",
    "is_unique",
    "range",
    "pattern",
    "regex",
]

# -------------------------------------------------------------------
# Public wrappers over Snowflake helpers
# -------------------------------------------------------------------


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
        r_type_norm = (r_type or "").upper()

        if not r_type_norm or r_type_norm not in SUPPORTED_RULE_TYPES:
            ok = False
            messages.append(f"Unsupported rule type: {r_type} for column {col}")
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



# """
# Shim module for backwards compatibility.

# The real implementations now live in nl_constraints_graph.core.nodes_validate.
# This file just re-exports them so imports like
# `from nl_constraints_graph.nodes_validate import validator_node`
# keep working.
# """

# from nl_constraints_graph.core.nodes_validate import *  # noqa: F401,F403