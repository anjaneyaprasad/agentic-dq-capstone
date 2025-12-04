from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

# Correct imports for the MCP client pieces
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import CallToolResult, ListToolsResult  # type: ignore[attr-defined]


# ---- Paths ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = python-services
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


class DQMCPClient:
    """
    Thin async wrapper around the MCP stdio client for dq_mcp_server.
    Starts:  python -m mcp_servers.dq_mcp_server  in the python-services folder.
    """

    def __init__(self) -> None:
        # StdioServerParameters wants a single string command + args list
        self.server_params = StdioServerParameters(
            command=sys.executable,                  # e.g. /usr/bin/python
            args=["-m", "mcp_servers.dq_mcp_server"],
            cwd=PROJECT_ROOT,                        # run from python-services
        )
        self._ctx_manager = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "DQMCPClient":
        # stdio_client takes StdioServerParameters
        self._ctx_manager = stdio_client(self.server_params)
        read_stream, write_stream = await self._ctx_manager.__aenter__()

        # name is optional, just helpful for logs
        self.session = ClientSession(read_stream, write_stream)
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None:
            await self.session.close()
        if self._ctx_manager is not None:
            await self._ctx_manager.__aexit__(exc_type, exc, tb)

    async def list_tools(self) -> List[str]:
        """
        Return list of tool names exposed by dq_mcp_server.
        """
        assert self.session is not None, "Session not initialized"

        # For your version, this simple call should work.
        # If it later complains about missing params, we can switch to:
        #   from mcp.types import ListToolsRequestParams
        #   result = await self.session.list_tools(ListToolsRequestParams())
        result: ListToolsResult = await self.session.list_tools()
        return [t.name for t in result.tools]

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Call a tool on dq_mcp_server and return a plain Python object.

        Unwraps CallToolResult.content into:
          - dict/list for JSON-like content
          - string for text content
          - or list of items if multiple.
        """
        assert self.session is not None, "Session not initialized"
        arguments = arguments or {}

        result: CallToolResult = await self.session.call_tool(
            name=name,
            arguments=arguments,
        )

        if not result.content:
            return None

        parsed_items: List[Any] = []
        for item in result.content:
            # Text content
            if hasattr(item, "text"):
                parsed_items.append(item.text)
            # JSON-like content (newer SDKs often use `.data`)
            elif hasattr(item, "data"):
                parsed_items.append(item.data)
            else:
                # Fallback: Pydantic model -> dict
                try:
                    parsed_items.append(item.model_dump())
                except Exception:
                    parsed_items.append(str(item))

        if len(parsed_items) == 1:
            return parsed_items[0]
        return parsed_items
    
def run_dq_tool(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    validate_tool: bool = True,
) -> Any:
    """
    Generic MCP client runner for DQ tools.

    Args:
        tool_name: Name of the MCP tool to call (e.g. "dq_latest_profiling").
        args: Dict of tool arguments (e.g. {"dataset": "FACT_SALES"}).
        validate_tool: If True, checks that tool_name exists in list_tools() first.

    Returns:
        Preferably the `structuredContent["result"]` dict if present,
        otherwise structuredContent, otherwise JSON-decoded text content,
        otherwise the raw ToolResponse object.
    """
    if args is None:
        args = {}

    print("[mcp_client] calling asyncio.run(_runner())", file=sys.stderr, flush=True)

    async def _runner():
        print("[mcp_client] _runner() starting", file=sys.stderr, flush=True)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_servers.dq_mcp_server"],
        )

        print(
            f"[mcp_client] launching MCP stdio_client: {server_params.command} {server_params.args}",
            file=sys.stderr,
            flush=True,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            print("[mcp_client] connected to MCP server", file=sys.stderr, flush=True)

            async with ClientSession(read_stream, write_stream) as session:
                print("[mcp_client] initializing session...", file=sys.stderr, flush=True)
                await session.initialize()
                print("[mcp_client] session initialized", file=sys.stderr, flush=True)

                tools = await session.list_tools()
                print(
                    f"[mcp_client] tools detected: {[t.name for t in tools.tools]}",
                    file=sys.stderr,
                    flush=True,
                )

                if validate_tool:
                    available = {t.name for t in tools.tools}
                    if tool_name not in available:
                        raise ValueError(
                            f"Tool '{tool_name}' not found. Available: {sorted(available)}"
                        )

                print(
                    f"[mcp_client] calling tool '{tool_name}' with args={args}",
                    file=sys.stderr,
                    flush=True,
                )
                tool_result = await session.call_tool(tool_name, args)
                print(
                    f"[mcp_client] tool '{tool_name}' response received",
                    file=sys.stderr,
                    flush=True,
                )
                
                # If tool reported an error, surface it clearly
                if getattr(tool_result, "isError", False):
                    msg = None
                    if getattr(tool_result, "content", None):
                        # Usually the server sticks error text in the first TextContent
                        msg = tool_result.content[0].text
                    raise RuntimeError(
                        f"MCP tool '{tool_name}' failed: {msg or 'unknown error'}"
                    )

                # 1) Prefer structuredContent["result"]
                sc = getattr(tool_result, "structuredContent", None)
                if sc is not None:
                    if isinstance(sc, dict) and "result" in sc:
                        return sc["result"]
                    return sc

                # 2) Fallback: JSON from first text content
                try:
                    from json import loads

                    if tool_result.content:
                        text = tool_result.content[0].text
                        return loads(text)
                except Exception:
                    pass

                # 3) Last resort: raw ToolResponse
                return tool_result

    return asyncio.run(_runner())

def dq_latest_profiling(dataset: str) -> Dict[str, Any]:
    return run_dq_tool("dq_latest_profiling", {"dataset": dataset})

def dq_latest_verification(dataset: str) -> Dict[str, Any]:
    return run_dq_tool("dq_latest_verification", {"dataset": dataset})

def dq_run_spark_pipeline(dataset: str, batch_date: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"dataset": dataset}
    if batch_date is not None:
        payload["batch_date"] = batch_date
    return run_dq_tool("dq_run_spark_pipeline", payload)

def dq_query_snowflake(sql: str) -> Dict[str, Any]:
    return run_dq_tool("dq_query_snowflake", {"sql": sql})

# -------------------------------------------------------------------
# Convenience DQ wrappers for app / Streamlit
# -------------------------------------------------------------------
from typing import Any, Dict


def dq_latest_profiling(dataset: str) -> Dict[str, Any]:
    """
    Convenience wrapper over MCP 'dq_latest_profiling'.
    Returns:
        {
          "dataset": "...",
          "metrics": [...],
          "summary": [...]
        }
    """
    return run_dq_tool("dq_latest_profiling", {"dataset": dataset})


def dq_latest_verification(dataset: str) -> Dict[str, Any]:
    """
    Wrapper over MCP 'dq_latest_verification'.
    Returns:
        {
          "dataset": "...",
          "run_id": "...",
          "run_ts": "...",
          "total_rules": int,
          "failed_rules": int,
          "constraints": [...]
        }
    """
    return run_dq_tool("dq_latest_verification", {"dataset": dataset})


def dq_run_spark_pipeline(dataset: str, batch_date: str | None = None) -> Dict[str, Any]:
    """
    Wrapper over MCP 'dq_run_spark_pipeline'.
    Kicks off the Spark DQ job for a dataset.
    """
    payload: Dict[str, Any] = {"dataset": dataset}
    if batch_date is not None:
        payload["batch_date"] = batch_date
    return run_dq_tool("dq_run_spark_pipeline", payload)


def dq_query_snowflake(sql: str) -> Dict[str, Any]:
    """
    Wrapper over MCP 'dq_query_snowflake'.
    """
    return run_dq_tool("dq_query_snowflake", {"sql": sql})