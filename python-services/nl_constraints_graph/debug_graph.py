import os
import sys

# Ensure project root is in PYTHONPATH
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv

from nl_constraints_graph.models import GraphState, NLRequest
from nl_constraints_graph.graph_nl_to_yaml import build_graph
from nl_constraints_graph.nodes_validate import get_dataset_columns


def main():
    load_dotenv()

    dataset = os.getenv("DEBUG_DATASET", "fact_sales")
    prompt = os.getenv(
        "DEBUG_PROMPT",
        "Ensure customer_id is unique and at least 99% complete.",
    )

    print(f"Using dataset={dataset}, prompt={prompt!r}")

    columns = get_dataset_columns(dataset)
    request = NLRequest(dataset=dataset, prompt=prompt, apply=False)

    init_state = GraphState(
        request=request,
        columns=columns,
        user_feedback=None,
        self_healing_enabled=True,
    )

    app = build_graph()

    print("\n=== Streaming graph execution (recursion_limit=8, debug=True) ===\n")
    try:
        for event in app.stream(
            init_state,
            config={"recursion_limit": 8, "debug": True},
        ):
            # `event` is usually a dict with node + data info.
            print(event)

    except Exception as e:
        print("\n*** ERROR during graph execution ***")
        print(repr(e))


if __name__ == "__main__":
    main()