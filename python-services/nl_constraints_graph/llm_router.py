# python-services/nl_constraints_graph/llm_router.py

from __future__ import annotations

import os
from typing import Any, Dict, List

Messages = List[Dict[str, Any]]


def call_openai(messages: Messages, **kwargs) -> str:
    """
    Low-level helper to call an OpenAI model.

    In tests, this function is monkeypatched.
    In default runtime, we just return a stub string.
    """
    return "Hello! How can I assist you today?"


def call_gemini(messages: Messages, **kwargs) -> str:
    """
    Low-level helper to call a Google Gemini model.

    In tests, this function is monkeypatched.
    In default runtime, we just return a stub string.
    """
    return "Hello! Welcome back! How can I assist you today?"


def call_llm(messages: Messages, **kwargs) -> Dict[str, Any]:
    """
    High-level LLM router with OpenAI as primary and Gemini as fallback.

    Returns a dict:
        {
            "content": <string>,
            "provider": "OpenAI" | "Google Gemini",
            "fallback_used": bool,
            "model": <model name from env>,
        }

    - In production, this function uses call_openai / call_gemini.
    - In tests, call_openai and call_gemini are monkeypatched.
    """

    primary_model = os.getenv("PRIMARY_MODEL")
    fallback_model = os.getenv("FALLBACK_MODEL")

    # Primary attempt: OpenAI
    try:
        content = call_openai(messages, model_name=primary_model, **kwargs)
        return {
            "content": content,
            "provider": "OpenAI",
            "fallback_used": False,
            "model": primary_model,
        }
    except Exception:
        # Fallback: Gemini
        content = call_gemini(messages, model_name=fallback_model, **kwargs)
        return {
            "content": content,
            "provider": "Google Gemini",
            "fallback_used": True,
            "model": fallback_model,
        }