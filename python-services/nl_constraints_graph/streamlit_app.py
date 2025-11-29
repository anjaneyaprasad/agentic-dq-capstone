from __future__ import annotations

"""
Thin wrapper so other modules (like the MCP server) can import these helpers
without worrying about the ui/ subpackage path.

Do NOT put any MCP client logic here – this stays a pure helper re-export.
"""

from .ui.app_local import (  # type: ignore
    run_spark_jobs,
    load_latest_verification_run,
    load_bad_rows_for_run,
)

__all__ = [
    "run_spark_jobs",
    "load_latest_verification_run",
    "load_bad_rows_for_run",
]
