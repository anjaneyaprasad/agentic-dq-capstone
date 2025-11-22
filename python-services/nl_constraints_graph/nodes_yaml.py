# python-services/nl_constraints_graph/nodes_yaml.py
from __future__ import annotations
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
import yaml

from .models import GraphState, RuleSpec

load_dotenv()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONSTRAINTS_DIR = os.path.join(
    ROOT_DIR,
    "dq-spark-project",
    "src",
    "main",
    "resources",
    "configs",
    "constraints",
)


env_constraints = os.getenv("CONSTRAINTS_PATH")
if env_constraints:
    if os.path.isabs(env_constraints):
        CONSTRAINTS_DIR = env_constraints
    else:
        CONSTRAINTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, env_constraints))
else:
    CONSTRAINTS_DIR = DEFAULT_CONSTRAINTS_DIR


def load_constraints_yaml(dataset: str) -> Dict[str, Any]:
    path = os.path.join(CONSTRAINTS_DIR, f"{dataset}.yaml")
    if not os.path.exists(path):
        return {"dataset": dataset, "constraints": []}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_constraints_yaml(dataset: str, data: Dict[str, Any]) -> str:
    path = os.path.join(CONSTRAINTS_DIR, f"{dataset}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def merge_rules(existing: Dict[str, Any], new_rules: List[RuleSpec]) -> Dict[str, Any]:
    if "constraints" not in existing or existing["constraints"] is None:
        existing["constraints"] = []
    if "dataset" not in existing or not existing["dataset"]:
        existing["dataset"] = new_rules[0].dataset if new_rules else ""

    existing_constraints: List[Dict[str, Any]] = existing["constraints"]

    def rule_key(d: Dict[str, Any]):
        return (
            d.get("type"),
            d.get("column"),
            d.get("level", "ERROR"),
            d.get("threshold"),
            tuple(sorted(d.get("allowed_values", []))) if d.get("allowed_values") else None,
            d.get("min"),
            d.get("max"),
        )

    existing_keys = {rule_key(d) for d in existing_constraints}

    for spec in new_rules:
        d = spec.to_yaml_dict()
        k = rule_key(d)
        if k not in existing_keys:
            existing_constraints.append(d)
            existing_keys.add(k)

    existing["constraints"] = existing_constraints
    return existing


def yaml_node(state: GraphState) -> GraphState:
    """
    Agent B: merge validated rules into YAML, optionally write to disk.
    """
    if not state.validation_ok:
        # skip modification
        return state

    dataset = state.request.dataset
    existing = load_constraints_yaml(dataset)
    merged = merge_rules(existing, state.inferred_rules)
    state.merged_yaml = merged

    if state.request.apply:
        path = save_constraints_yaml(dataset, merged)
        state.yaml_path = path

    return state