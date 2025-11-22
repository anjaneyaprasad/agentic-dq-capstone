from __future__ import annotations
import json
import os
from datetime import datetime
from typing import List, Dict, Any

MEMORY_PATH = os.path.join(
    os.path.dirname(__file__),
    "nl_rules_memory.jsonl"
)


def save_interaction(
    dataset: str,
    prompt: str,
    rules: List[Dict[str, Any]],
    messages: List[str],
) -> None:
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    record = {
        "ts": datetime.utcnow().isoformat(),
        "dataset": dataset,
        "prompt": prompt,
        "rules": rules,
        "messages": messages,
    }
    with open(MEMORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_recent_examples(dataset: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not os.path.exists(MEMORY_PATH):
        return []

    examples: List[Dict[str, Any]] = []
    with open(MEMORY_PATH, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("dataset") == dataset:
                examples.append(rec)

    return examples[-limit:]