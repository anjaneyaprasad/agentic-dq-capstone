# python-services/nl_constraints_graph/nodes_intent.py

from __future__ import annotations
from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .llm_client import get_llm
from .models import GraphState, RuleSpec
from .rules_memory import load_recent_examples


class RulesList(BaseModel):
    """Wrapper model so PydanticOutputParser can work."""
    rules: List[RuleSpec]

"""
Agent A — Intent Node (NL → RuleSpec Conversion)

This function interprets natural language (NL) instructions and converts them
into structured data quality rule specifications (`RuleSpec`). It forms the
first step of the NL-to-YAML rule generation workflow within the
`nl_constraints_graph` system.

**Responsibilities**
- Uses an LLM (via LangChain) to translate user intent into machine-readable rules.
- Applies few-shot examples sourced from previous valid rules for the same dataset.
- Produces output conforming to the `RulesList` schema using a Pydantic output parser.
- Validates the generated response structure and records errors if parsing fails.

**Behavior**
- Populates `state.inferred_rules` with generated rule objects.
- Adds validation warnings or errors to `state.validation_messages`.
- Sets `state.validation_ok = False` when generation fails or rules cannot be parsed.
- Includes a heuristic fix: if the prompt references “currency” and the dataset
  contains a matching column, a domain constraint is generated appropriately.

**Inputs**
- `GraphState` containing:
  - The natural language request (`NLRequest`)
  - The dataset column list

**Outputs**
- Updated `GraphState` with inferred rules attached.
- No external side effects (pure transformation of state).

**Usage Example**
state = intent_node(state)
"""

def intent_node(state: GraphState) -> GraphState:
    """
    Agent A: NL → RuleSpec list (via LLM + structured output).
    """
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=RulesList)
    
    # Load recent examples for this dataset to guide the LLM
    examples = load_recent_examples(state.request.dataset, limit=3)
    examples_text_parts = []
    for ex in examples:
        examples_text_parts.append(
            f"- Prompt: {ex['prompt']}\n  Rules: {ex['rules']}"
        )
    examples_text = "\n".join(examples_text_parts) if examples_text_parts else "None"
    
    print("=== FEW-SHOT EXAMPLES USED ===")
    print(examples_text)
    print("=== END EXAMPLES ===")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a data quality rule assistant. "
                    "Given a dataset name, its columns, and a natural language description, "
                    "extract ONE OR MORE constraint rules for a Deequ-like engine.\n\n"
                    "You MUST respond ONLY with JSON that matches the provided schema."
                ),
            ),
            (
                "user",
                (
                    "Dataset: {dataset}\n"
                    "Columns: {columns}\n\n"
                    "Previous good examples for this dataset (if any):\n"
                    "{examples}\n\n"
                    "New instruction:\n{instruction}\n\n"
                    "Allowed rule types (field 'type'):\n"
                    "  - completeness\n"
                    "  - completeness_threshold\n"
                    "  - non_negative\n"
                    "  - domain\n"
                    "  - unique\n"
                    "  - size_greater_than\n"
                    "  - min_value\n"
                    "  - max_value\n\n"
                    "Rules should follow these guidelines:\n"
                    "- For completeness_threshold: 0.0 <= threshold <= 1.0\n"
                    "- For domain: fill allowed_values with a list of strings\n"
                    "- For size_greater_than: set column to empty string\n"
                    "- level should usually be 'ERROR' unless the instruction suggests a warning\n\n"
                    "{format_instructions}"
                ),
            ),
        ]
    )

    format_instructions = parser.get_format_instructions()

    chain = prompt | llm | parser

    try:
        result: RulesList = chain.invoke(
            {
                "dataset": state.request.dataset,
                "columns": ", ".join(state.columns),
                "instruction": state.request.prompt,
                "examples": examples_text,
                "format_instructions": format_instructions,
            }
        )
        rules = result.rules
    except Exception as e:
        state.validation_ok = False
        state.validation_messages.append(f"Intent extraction failed: {e}")
        return state

    # --- HEURISTIC FIX: domain on currency must use 'currency' column ---
    prompt_lower = state.request.prompt.lower()
    cols_lower_map = {c.lower(): c for c in state.columns}

    has_currency_column = "currency" in cols_lower_map
    mentions_currency_word = "currency" in prompt_lower

    for r in rules:
        # if it's a domain rule and prompt talks about 'currency'
        if r.type == "domain" and has_currency_column and mentions_currency_word:
            # if model chose some other column, override to proper 'currency'
            if r.column.lower() != "currency":
                r.column = cols_lower_map["currency"]

    state.inferred_rules = rules
    return state