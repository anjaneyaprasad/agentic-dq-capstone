import json
import pandas as pd
import pytest

from nl_constraints_graph.dq_brain import (
    metrics_to_json_per_column,
    build_dq_brain_payload,
    build_dq_brain_prompt,
    parse_llm_rules_json,
    save_rules_to_snowflake,
)


def test_metrics_to_json_per_column_basic():
    df = pd.DataFrame(
        [
            {"COLUMN_NAME": "ID", "METRIC_NAME": "Completeness",       "METRIC_VALUE": 0.99},
            {"COLUMN_NAME": "ID", "METRIC_NAME": "ApproxCountDistinct","METRIC_VALUE": 1000},
            {"COLUMN_NAME": "AMOUNT", "METRIC_NAME": "Completeness",   "METRIC_VALUE": 0.95},
        ]
    )

    result = metrics_to_json_per_column(df)

    # Expect 2 entries: ID and AMOUNT
    cols = {r["column"] for r in result}
    assert cols == {"ID", "AMOUNT"}

    id_entry = next(r for r in result if r["column"] == "ID")
    assert id_entry["metrics"]["Completeness"] == 0.99
    assert id_entry["metrics"]["ApproxCountDistinct"] == 1000


def test_build_dq_brain_payload_shape():
    df = pd.DataFrame(
        [
            {"COLUMN_NAME": "ID", "METRIC_NAME": "Completeness", "METRIC_VALUE": 0.99},
        ]
    )
    payload = build_dq_brain_payload("FACT_SALES", df)

    assert payload["dataset"] == "FACT_SALES"
    assert isinstance(payload["columns"], list)
    assert len(payload["columns"]) == 1
    assert payload["columns"][0]["column"] == "ID"
    assert payload["columns"][0]["metrics"]["Completeness"] == 0.99


def test_build_dq_brain_prompt_contains_dataset_and_json():
    df = pd.DataFrame(
        [
            {"COLUMN_NAME": "ID", "METRIC_NAME": "Completeness", "METRIC_VALUE": 0.99},
        ]
    )
    prompt = build_dq_brain_prompt("FACT_SALES", df)

    # sanity checks
    assert "FACT_SALES" in prompt
    assert "PROFILING_JSON" in prompt
    assert '"dataset": "FACT_SALES"' in prompt  # JSON snippet embedded


def test_parse_llm_rules_json_valid():
    raw = json.dumps(
        {
            "rules": [
                {
                    "ruleType": "COMPLETENESS",
                    "column": "ID",
                    "level": "ERROR",
                    "threshold": 0.99,
                    "minValue": None,
                    "maxValue": None,
                    "allowedValues": None,
                    "pattern": None,
                    "explanation": "ID is almost always present.",
                }
            ]
        }
    )

    rules = parse_llm_rules_json(raw)
    assert len(rules) == 1
    r = rules[0]
    assert r["ruleType"] == "COMPLETENESS"
    assert r["column"] == "ID"
    assert r["threshold"] == 0.99


def test_parse_llm_rules_json_raises_on_missing_rules():
    raw = json.dumps({"not_rules": []})
    with pytest.raises(ValueError):
        parse_llm_rules_json(raw)



def test_save_rules_to_snowflake_uses_insert(monkeypatch):
    """
    Pure behavioral test: ensure save_rules_to_snowflake executes insert
    with correct number of calls. Does NOT hit real Snowflake.
    """
    calls = []

    class FakeCursor:
        def execute(self, sql, params):
            calls.append((sql, params))

    class FakeConn:
        def cursor(self):
            return FakeCursor()
        def commit(self):
            calls.append(("COMMIT", None))
        def close(self):
            calls.append(("CLOSE", None))

    # monkeypatch get_snowflake_connection to return fake conn
    from nl_constraints_graph import dq_brain

    monkeypatch.setattr(dq_brain, "get_snowflake_connection", lambda: FakeConn())

    rules = [
        {
            "ruleType": "COMPLETENESS",
            "column": "ID",
            "level": "ERROR",
            "threshold": 0.99,
            "minValue": None,
            "maxValue": None,
            "allowedValues": None,
            "pattern": None,
            "explanation": "ID is almost always present.",
        },
        {
            "ruleType": "UNIQUENESS",
            "column": "ID",
            "level": "ERROR",
            "threshold": None,
            "minValue": None,
            "maxValue": None,
            "allowedValues": None,
            "pattern": None,
            "explanation": "ID appears to be unique.",
        },
    ]

    save_rules_to_snowflake("FACT_SALES", rules, created_by="TEST_USER")

    # Expect 2 insert calls + commit + close
    insert_calls = [c for c in calls if isinstance(c[0], str) and "INSERT INTO" in c[0]]
    assert len(insert_calls) == 2
    assert any(c[0] == "COMMIT" for c in calls)
    assert any(c[0] == "CLOSE" for c in calls)
