from __future__ import annotations
import argparse
import yaml

from .models import GraphState, NLRequest
from .graph_nl_to_yaml import build_graph
from .nodes_validate import get_dataset_columns
from .rules_memory import save_interaction

"""
Main entry point for NL → YAML constraints workflow 
Usage example:
python -m nl_constraints_graph.main --dataset fact_sales --prompt "customer_id must be unique and 99% complete; currency only INR, USD and EUR" --apply
python -m nl_constraints_graph.main --dataset users --prompt "Ensure age is non-negative and completeness >= 0.9" --apply
Optional: --feedback "The previous rules missed the uniqueness constraint on user_id."
The --apply flag writes changes to YAML if validation passes.
The script prints inferred rules, validation messages, and YAML preview to console.
It also saves the interaction to memory for future reference.
Note: Ensure that the necessary environment (LLM access, dataset configs) is set up before running.
Adjust the prompt as needed for different datasets and constraints.
The script uses LangGraph to orchestrate the workflow.
The final YAML is compatible with Deequ-like engines.
"""
def main():
    parser = argparse.ArgumentParser(
        description="LangGraph NL → YAML constraints workflow"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to YAML file if validation passes.",
    )
    parser.add_argument(
        "--feedback",
        required=False,
        help="Optional user feedback to guide refinement.",
    )
    args = parser.parse_args()

    request = NLRequest(dataset=args.dataset, prompt=args.prompt, apply=args.apply)

    # initial state
    columns = get_dataset_columns(args.dataset)
    
    init_state = GraphState(
        request=request,
        columns=columns,
        user_feedback=args.feedback,
    )

    app = build_graph()
    raw_result = app.invoke(init_state)

    # LangGraph often returns a dict; convert to GraphState if needed
    if isinstance(raw_result, GraphState):
        final_state = raw_result
    else:
        final_state = GraphState.model_validate(raw_result)

    # persist to memory (even if validation failed; you can filter later)
    try:
        save_interaction(
            dataset=args.dataset,
            prompt=args.prompt,
            rules=[r.dict() for r in final_state.inferred_rules],
            messages=final_state.validation_messages,
        )
    except Exception as e:
        print(f"[WARN] Could not save interaction to memory: {e}")

    print("\n=== Validation messages ===")
    if final_state.validation_messages:
        for m in final_state.validation_messages:
            print(" -", m)
    else:
        print(" (none)")

    print("\n=== Inferred / refined rules ===")
    if final_state.inferred_rules:
        for r in final_state.inferred_rules:
            print(" -", r)
    else:
        print(" (none)")

    print("\n=== YAML Preview ===")
    if final_state.merged_yaml:
        print(yaml.safe_dump(final_state.merged_yaml, sort_keys=False))
    else:
        print(" (no YAML generated)")

    if final_state.yaml_path:
        print(f"\n[INFO] YAML file updated at: {final_state.yaml_path}")
    else:
        if args.apply:
            print("\n[INFO] Apply was requested, but YAML was not written (likely validation failed).")
        else:
            print("\n[INFO] Dry-run only; YAML not written (use --apply to persist changes).")


if __name__ == "__main__":
    main()
