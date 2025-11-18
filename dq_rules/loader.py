from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from dq_rules.models import DQConfig, DQRule, ForeignKeyRule, CheckType


def load_dq_config(path: str | Path) -> DQConfig:
    """
    Load YAML and validate it via Pydantic.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = DQConfig(**raw)
    return config


def rule_to_deequ_spec(table_name: str, rule: DQRule) -> Dict[str, Any]:
    """
    Convert one DQRule into a generic "constraint spec" that
    your Scala/Deequ layer can interpret.

    Example output item:

    {
      "table": "fact_sales",
      "rule_id": "fact_sales_quantity_non_negative",
      "severity": "error",
      "constraint_type": "isNonNegative",
      "column": "quantity",
      "params": {}
    }
    """
    if rule.check == CheckType.not_null:
        constraint_type = "isComplete"        # Deequ: check.isComplete(column)
        params: Dict[str, Any] = {}
    elif rule.check == CheckType.unique:
        constraint_type = "isUnique"          # Deequ: check.isUnique(column)
        params = {}
    elif rule.check == CheckType.greater_or_equal:
        constraint_type = "isGreaterThanOrEqualTo"  # custom mapping
        params = {"threshold": rule.threshold}
    elif rule.check == CheckType.less_or_equal:
        constraint_type = "isLessThanOrEqualTo"
        params = {"threshold": rule.threshold}
    elif rule.check == CheckType.in_set:
        constraint_type = "isContainedIn"
        params = {"allowed_values": rule.allowed_values}
    else:
        raise ValueError(f"Unsupported check type: {rule.check}")

    return {
        "table": table_name,
        "rule_id": rule.id,
        "severity": rule.severity.value,
        "constraint_type": constraint_type,
        "column": rule.column,
        "params": params,
    }


def fk_to_deequ_spec(table_name: str, fk: ForeignKeyRule) -> Dict[str, Any]:
    """
    Represent FK constraints for Deequ / Spark:

    We'll enforce them by:
      - join between fact and dim
      - require no nulls / orphans in joined keys
    """
    return {
        "table": table_name,
        "rule_id": fk.id,
        "severity": fk.severity.value,
        "constraint_type": "foreignKey",
        "column": fk.local_column,
        "params": {
            "referenced_table": fk.referenced_table,
            "referenced_column": fk.referenced_column,
        },
    }


def build_deequ_spec(config: DQConfig) -> List[Dict[str, Any]]:
    """
    Flatten all tables + rules + FK rules into a list of constraint specs.
    These can then be written as JSON and consumed by Scala/Deequ code.
    """
    specs: List[Dict[str, Any]] = []

    for table in config.tables:
        for r in table.rules:
            specs.append(rule_to_deequ_spec(table.name, r))

        for fk in table.foreign_keys:
            specs.append(fk_to_deequ_spec(table.name, fk))

    return specs


def export_deequ_spec(
    yaml_path: str | Path = "config/rules.yaml",
    output_json: str | Path = "config/deequ_rules_resolved.json",
) -> None:
    """
    Helper: load YAML -> validate -> convert -> dump JSON
    to be used by Scala/Spark/Deequ side.
    """
    cfg = load_dq_config(yaml_path)
    specs = build_deequ_spec(cfg)

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)

    print(f"[INFO] Exported {len(specs)} constraints to {output_json}")


if __name__ == "__main__":
    export_deequ_spec()
