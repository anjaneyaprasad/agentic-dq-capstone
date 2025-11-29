from __future__ import annotations

import os
from typing import Dict, List

from openai import OpenAI

# Import from your MCP server module
from mcp_servers.dq_mcp_server import (
    DQ_NL_SQL_SYSTEM_PROMPT,
    DQ_NL_SQL_FEW_SHOTS,
    _build_nl_sql_messages,
)

client = OpenAI()  # uses OPENAI_API_KEY from env


def main() -> None:
    prompt = "give me query to see uniqueness of sales_id on sales table"

    messages = _build_nl_sql_messages(prompt)

    print("=== SYSTEM MESSAGE (truncated) ===")
    print(DQ_NL_SQL_SYSTEM_PROMPT[:600], "...\n")

    print("=== FEW-SHOT EXAMPLES (roles + first line) ===")
    for m in DQ_NL_SQL_FEW_SHOTS:
        first_line = m["content"].strip().splitlines()[0]
        print(f"{m['role'].upper()}: {first_line}")
    print()

    print("=== CALLING OPENAI WITH PROMPT ===")
    print(prompt)
    print("==================================\n")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )

    sql = (resp.choices[0].message.content or "").strip()

    print("=== RAW SQL FROM MODEL ===")
    print(sql)
    print("===========================")


if __name__ == "__main__":
    main()
