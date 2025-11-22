# python-services/nl_constraints_graph/nodes_validate.py
from __future__ import annotations
from typing import List, Tuple
from dotenv import load_dotenv
import os
import yaml

from .models import GraphState, RuleSpec, RuleType

load_dotenv()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATASETS_YAML = os.path.join(
    ROOT_DIR,
    "dq-spark-project",
    "src",
    "main",
    "resources",
    "configs",
    "datasets.yaml",
)

# Allow override via .env, but make it relative to ROOT_DIR if not absolute
env_path = os.getenv("DATASETS_YAML_PATH")
if env_path:
    if os.path.isabs(env_path):
        DATASETS_YAML = env_path
    else:
        DATASETS_YAML = os.path.abspath(os.path.join(ROOT_DIR, env_path))
else:
    DATASETS_YAML = DEFAULT_DATASETS_YAML

SUPPORTED_RULE_TYPES: List[RuleType] = [
    "completeness",
    "completeness_threshold",
    "non_negative",
    "domain",
    "unique",
    "size_greater_than",
    "min_value",
    "max_value",
]


def load_datasets_config():
    with open(DATASETS_YAML, "r") as f:
        return yaml.safe_load(f)


def get_dataset_columns(dataset: str) -> List[str]:
    cfg = load_datasets_config()
    for ds in cfg.get("datasets", []):
        if ds.get("name") == dataset:
            cols = set(ds.get("critical_columns", []))
            cols.update(ds.get("primary_keys", []))
            cols.update(ds.get("partitions", []))
            return sorted(cols)
    raise ValueError(f"Dataset '{dataset}' not found in datasets.yaml")


def validate_rules(dataset: str, rules: List[RuleSpec]) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    ok = True

    try:
        columns = set(get_dataset_columns(dataset))
    except Exception as e:
        return False, [f"Dataset lookup failed: {e}"]

    for r in rules:
        if r.type not in SUPPORTED_RULE_TYPES:
            ok = False
            messages.append(f"Unsupported rule type: {r.type} for column {r.column}")
            continue

        if r.type not in ("size_greater_than",) and r.column:
            if r.column not in columns:
                ok = False
                messages.append(
                    f"Column '{r.column}' does not exist in dataset '{dataset}'. "
                    f"Known columns: {sorted(columns)}"
                )

        if r.type == "domain" and not r.allowed_values:
            ok = False
            messages.append(
                f"Rule 'domain' for column '{r.column}' must have allowed_values"
            )

        if r.type in ("completeness_threshold", "size_greater_than"):
            if r.threshold is None:
                ok = False
                messages.append(
                    f"Rule '{r.type}' for column '{r.column}' requires threshold"
                )
            elif r.type == "completeness_threshold" and not (0.0 <= r.threshold <= 1.0):
                ok = False
                messages.append(
                    f"completeness_threshold for '{r.column}' must be between 0 and 1; got {r.threshold}"
                )

        if r.type == "min_value" and r.min is None:
            ok = False
            messages.append(
                f"min_value rule for column '{r.column}' requires 'min'"
            )
        if r.type == "max_value" and r.max is None:
            ok = False
            messages.append(
                f"max_value rule for column '{r.column}' requires 'max'"
            )

    if ok:
        messages.append("All rules validated successfully.")
    return ok, messages


def validator_node(state: GraphState) -> GraphState:
    """
    Agent C – validate newly inferred rules before applying.
    Uses dataset schema + a bit of prompt context.
    """
    try:
        columns = set(get_dataset_columns(state.request.dataset))
    except Exception as e:
        state.validation_ok = False
        state.validation_messages.append(str(e))
        return state

    state.columns = sorted(columns)

    if not state.inferred_rules:
        state.validation_ok = False
        state.validation_messages.append("No rules inferred to validate.")
        return state

    ok = True
    messages: List[str] = []

    prompt_lower = state.request.prompt.lower()
    cols_lower = {c.lower() for c in columns}

    for r in state.inferred_rules:
        # 1) supported type?
        if r.type not in SUPPORTED_RULE_TYPES:
            ok = False
            messages.append(f"Unsupported rule type: {r.type} for column {r.column}")
            continue

        # 2) column existence for column-based rules
        if r.type not in ("size_greater_than",) and r.column:
            if r.column not in columns:
                ok = False
                messages.append(
                    f"Column '{r.column}' does not exist in dataset '{state.request.dataset}'. "
                    f"Known columns: {sorted(columns)}"
                )
                continue

        # 3) SPECIAL GUARD: user mentioned 'currency' but dataset has no currency column
        #    In that case, DO NOT allow a domain rule on some random column.
        if r.type == "domain" and "currency" in prompt_lower:
            if "currency" not in cols_lower and r.column.lower() != "currency":
                ok = False
                messages.append(
                    "Prompt mentions 'currency', but dataset has no 'currency' column. "
                    f"Refusing to attach currency-like domain rule to column '{r.column}'."
                )
                continue

        # 4) domain requires allowed_values
        if r.type == "domain" and not r.allowed_values:
            ok = False
            messages.append(
                f"Rule 'domain' for column '{r.column}' must have allowed_values"
            )

        # 5) thresholds
        if r.type in ("completeness_threshold", "size_greater_than"):
            if r.threshold is None:
                ok = False
                messages.append(
                    f"Rule '{r.type}' for column '{r.column}' requires threshold"
                )
            elif r.type == "completeness_threshold" and not (0.0 <= r.threshold <= 1.0):
                ok = False
                messages.append(
                    f"completeness_threshold for '{r.column}' must be between 0 and 1; got {r.threshold}"
                )

        # 6) min/max
        if r.type == "min_value" and r.min is None:
            ok = False
            messages.append(
                f"min_value rule for column '{r.column}' requires 'min'"
            )
        if r.type == "max_value" and r.max is None:
            ok = False
            messages.append(
                f"max_value rule for column '{r.column}' requires 'max'"
            )

    state.validation_ok = ok
    state.validation_messages.extend(messages)
    if ok and not messages:
        state.validation_messages.append("All rules validated successfully.")
    return state

