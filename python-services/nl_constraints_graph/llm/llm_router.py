from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI  # make sure `openai` package is installed

Messages = List[Dict[str, Any]]


# ----------------- Low-level model calls -----------------


def call_openai(messages: Messages, model: str | None = None, **kwargs) -> str:
    """
    Actual OpenAI chat call.
    Returns the assistant message content as a plain string.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    model_name = model or os.getenv("PRIMARY_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        **kwargs,
    )

    return response.choices[0].message.content or ""


def call_gemini(messages: Messages, model: str | None = None, **kwargs) -> str:
    """
    Stub for Google Gemini.
    Wire this to the real Gemini client if/when you want.
    """
    raise RuntimeError(
        "call_gemini is not configured. "
        "Implement a real client call or remove Gemini models from PRIMARY/FALLBACK."
    )


# ----------------- Router helpers -----------------


def _is_gpt_model(name: str | None) -> bool:
    return bool(name) and name.startswith("gpt-")


def _is_gemini_model(name: str | None) -> bool:
    return bool(name) and name.startswith("gemini")


# ----------------- Main router -----------------


def call_llm(messages: Messages, model: str | None = None) -> str:
    """
    Generic LLM router.
    - Uses PRIMARY_MODEL by default.
    - Optionally falls back to FALLBACK_MODEL.
    - Only calls Gemini if explicitly configured.
    """
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "")

    chosen = model or primary

    def _run_one(model_name: str) -> str:
        if _is_gpt_model(model_name):
            return call_openai(messages, model=model_name)

        if _is_gemini_model(model_name):
            # Only allow Gemini if user really wants it
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise RuntimeError(
                    "Gemini model requested but GEMINI_API_KEY is not set"
                )
            return call_gemini(messages, model=model_name)

        raise ValueError(f"Unsupported model name: {model_name}")

    # 1) try chosen model first
    try:
        return _run_one(chosen)
    except Exception as e:
        # if no fallback, or already tried fallback, re-raise
        if not fallback or chosen == fallback:
            raise

        # 2) try fallback model
        try:
            return _run_one(fallback)
        except Exception:
            # nothing left to try
            raise e
