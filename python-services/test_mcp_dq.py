from pprint import pprint
from nl_constraints_graph.mcp_client import run_dq_tool

if __name__ == "__main__":
    print(">>> Starting test_mcp_dq.py")

    profiling = run_dq_tool("dq_latest_profiling", {"dataset": "FACT_SALES"})
    print(">>> dq_latest_profiling result (summary only):")
    for col in profiling.get("summary", []):
        print(
            f"  - {col['column']}: type={col['inferred_type']}, "
            f"non_null={col['non_null_pct']}%, approx_distinct={col['approx_distinct']}"
        )

    print("[mcp_client] calling asyncio.run(_runner())")
    verification = run_dq_tool("dq_latest_verification", {"dataset": "FACT_SALES"})
    print("\n>>> dq_latest_verification result:")
    pprint(verification)