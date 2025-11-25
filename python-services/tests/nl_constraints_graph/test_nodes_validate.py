import os
import sys

# --- Ensure python-services is on sys.path ---
CURRENT_DIR = os.path.dirname(__file__)                      # .../python-services/tests/nl_constraints_graph
PYTHON_SERVICES_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PYTHON_SERVICES_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_SERVICES_ROOT)

import pytest
import nl_constraints_graph.nodes_validate as nodes_validate
from nl_constraints_graph.nodes_validate import validator_node, SUPPORTED_RULE_TYPES

class DummyRequest:
    def __init__(self, dataset: str, prompt: str):
        self.dataset = dataset
        self.prompt = prompt

class DummyRule:
    """
    Minimal stand-in for RuleSpec.
    Must have the attributes that validate_rules uses:
    type, column, allowed_values, threshold, min, max
    """
    def __init__(
        self,
        column=None,
        type_=None,
        allowed_values=None,
        threshold=None,
        min_=None,
        max_=None,
    ):
        self.column = column
        self.type = type_
        self.allowed_values = allowed_values
        self.threshold = threshold
        self.min = min_
        self.max = max_

class DummyState:
    """
    Minimal stand-in for GraphState with only the fields that validator_node uses.
    """
    def __init__(self, dataset: str, prompt: str, rules):
        self.request = DummyRequest(dataset, prompt)
        self.inferred_rules = rules

        self.validation_ok = None
        self.validation_messages = []
        self.anomaly_messages = []
        self.columns = []

        self.refinement_attempts = 0
        self.max_refinements = 1
        self.self_healing_enabled = False
        self.dq_metrics_all = None  # if you use this elsewhere, it's safe to have

def test_validate_rules_unsupported_type(monkeypatch):
    """
    If a rule has a type not in SUPPORTED_RULE_TYPES,
    validate_rules should return ok=False and a clear message.
    """

    # Make dataset lookup simple & in-memory
    monkeypatch.setattr(
        nodes_validate,
        "get_dataset_columns",
        lambda dataset: ["order_id", "customer_id"],
    )

    rules = [
        DummyRule(column="order_id", type_="weird_type")
    ]

    ok, messages = nodes_validate.validate_rules("sales_orders", rules)

    assert ok is False
    joined = " ".join(messages).lower()
    assert "unsupported rule type" in joined
    assert "order_id" in joined


def test_validate_rules_missing_column(monkeypatch):
    """
    If a rule refers to a column not in the dataset config,
    validate_rules should flag it.
    """

    # Dataset has only customer_id
    monkeypatch.setattr(
        nodes_validate,
        "get_dataset_columns",
        lambda dataset: ["customer_id"],
    )

    rules = [
        DummyRule(column="order_id", type_="completeness")
    ]

    ok, messages = nodes_validate.validate_rules("sales_orders", rules)

    assert ok is False
    joined = " ".join(messages).lower()
    assert "order_id" in joined
    assert "does not exist" in joined or "known columns" in joined
    
def test_validator_node_no_rules(monkeypatch):
    """
    If there are no inferred rules, validator_node should flag validation_ok=False
    and add a 'No rules inferred to validate.' message.
    """

    # Make dataset + profiling trivial and in-memory
    monkeypatch.setattr(
        "nl_constraints_graph.nodes_validate.get_dataset_columns",
        lambda dataset: ["order_id", "customer_id"],
    )
    monkeypatch.setattr(
        "nl_constraints_graph.nodes_validate.detect_anomalies",
        lambda dataset: [],
    )
    monkeypatch.setattr(
        "nl_constraints_graph.nodes_validate.load_latest_profile",
        lambda dataset: {},  # no metrics
    )

    state = DummyState(dataset="sales_orders", prompt="just test", rules=[])

    new_state = validator_node(state)

    assert new_state.validation_ok is False
    joined = " ".join(new_state.validation_messages).lower()
    assert "no rules inferred" in joined


def test_validator_node_simple_valid_rule(monkeypatch):
    """
    With a supported rule type and existing column, validator_node should
    mark validation_ok=True and add a success message.
    """

    monkeypatch.setattr(
        "nl_constraints_graph.nodes_validate.get_dataset_columns",
        lambda dataset: ["order_id", "customer_id"],
    )
    monkeypatch.setattr(
        "nl_constraints_graph.nodes_validate.detect_anomalies",
        lambda dataset: [],
    )
    monkeypatch.setattr(
        "nl_constraints_graph.nodes_validate.load_latest_profile",
        lambda dataset: {},  # no anomalies from profile
    )

    # Pick any supported type (e.g. completeness)
    rule_type = "completeness"
    assert rule_type in SUPPORTED_RULE_TYPES

    rules = [DummyRule(column="order_id", type_=rule_type)]
    state = DummyState(dataset="sales_orders", prompt="check completeness", rules=rules)

    new_state = validator_node(state)

    assert new_state.validation_ok is True
    # Either explicit message or at least no errors
    joined = " ".join(new_state.validation_messages).lower()
    assert "error" not in joined