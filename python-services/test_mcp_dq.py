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


# from pprint import pprint
# from nl_constraints_graph.mcp_client import run_dq_tool

# if __name__ == "__main__":
#     print(">>> Starting test_mcp_dq.py")

#     # 1) Latest profiling
#     profiling = run_dq_tool("dq_latest_profiling", {"dataset": "FACT_SALES"})
#     print(">>> dq_latest_profiling result (summary only):")
#     for col in profiling.get("summary", []):
#         print(
#             f"  - {col['column']}: type={col['inferred_type']}, "
#             f"non_null={col['non_null_pct']}%, approx_distinct={col['approx_distinct']}"
#         )

#     # 2) Latest verification (DQ results)
#     verification = run_dq_tool("dq_latest_verification", {"dataset": "FACT_SALES"})
#     print("\n>>> dq_latest_verification result:")
#     pprint(verification)

#     # 3) Run Spark DQ pipeline
#     # run_result = run_dq_tool("dq_run_spark_pipeline", {"dataset": "FACT_SALES"})
#     # print("\n>>> dq_run_spark_pipeline result:")
#     # pprint(run_result)


# # from pprint import pprint

# # from nl_constraints_graph.mcp_client import run_dq_tool

# # if __name__ == "__main__":
# #     print(">>> Starting test_mcp_dq.py")
# #     result = run_dq_tool(dataset="FACT_SALES")
# #     print(">>> Clean result from MCP tool (dq_latest_profiling):")
# #     pprint(result)

# #     # Optionally print just a small summary
# #     print("\n>>> Columns summary:")
# #     for col in result.get("summary", []):
# #         print(
# #             f"  - {col['column']}: type={col['inferred_type']}, "
# #             f"non_null={col['non_null_pct']}%, approx_distinct={col['approx_distinct']}"
# #         )


# # # from nl_constraints_graph.mcp_client import run_dq_tool

# # # if __name__ == "__main__":
# # #     print(">>> Starting test_mcp_dq.py")
# # #     result = run_dq_tool(dataset="FACT_SALES")
# # #     print(">>> Result from MCP tool:")
# # #     print(result)


# # # # # python-services/test_mcp_dq.py

# # # # from nl_constraints_graph.mcp_client import run_dq_tool

# # # # if __name__ == "__main__":
# # # #     print(">>> Starting test_mcp_dq.py")
# # # #     result = run_dq_tool(
# # # #         input_text="Check completeness rules for dataset FACT_SALES",
# # # #         dataset="FACT_SALES",
# # # #     )
# # # #     print(">>> Result from MCP tool:")
# # # #     print(result)
