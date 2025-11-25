import os
import sys
import types

# --- Ensure python-services is on sys.path ---
CURRENT_DIR = os.path.dirname(__file__)                      # .../python-services/tests/nl_constraints_graph
PYTHON_SERVICES_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PYTHON_SERVICES_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_SERVICES_ROOT)

# --- Provide a fake start_up module so `from start_up import *` doesn't explode ---
if "start_up" not in sys.modules:
    fake_start_up = types.ModuleType("start_up")
    sys.modules["start_up"] = fake_start_up

import pytest
import nl_constraints_graph.llm_router as llm_router


def test_call_llm_uses_openai_on_success(monkeypatch):
    """
    When call_openai succeeds, call_llm should:
      - return that content
      - set provider='OpenAI'
      - fallback_used=False
      - model=PRIMARY_MODEL env var
    """

    # Arrange
    messages = [{"role": "user", "content": "hello"}]

    # Fake primary and fallback models
    os.environ["PRIMARY_MODEL"] = "gpt-4o-mini"
    os.environ["FALLBACK_MODEL"] = "gemini-2.0-flash"

    def fake_call_openai(msgs, **kwargs):
        assert msgs == messages
        # you can also assert on kwargs.get("model_name") if you want
        return "primary response"

    def fake_call_gemini(msgs, **kwargs):
        # Should not be called in this test
        raise AssertionError("call_gemini should not be called when primary succeeds")

    monkeypatch.setattr(llm_router, "call_openai", fake_call_openai)
    monkeypatch.setattr(llm_router, "call_gemini", fake_call_gemini)

    # Act
    result = llm_router.call_llm(messages)

    # Assert
    assert result["content"] == "primary response"
    assert result["provider"] == "OpenAI"
    assert result["fallback_used"] is False
    assert result["model"] == "gpt-4o-mini"


def test_call_llm_falls_back_to_gemini_on_exception(monkeypatch):
    """
    When call_openai raises, call_llm should:
      - call call_gemini
      - mark provider='Google Gemini'
      - fallback_used=True
      - model=FALLBACK_MODEL env var
    """

    messages = [{"role": "user", "content": "hello again"}]

    os.environ["PRIMARY_MODEL"] = "gpt-4o-mini"
    os.environ["FALLBACK_MODEL"] = "gemini-2.0-flash"

    def fake_call_openai(msgs, **kwargs):
        raise RuntimeError("primary model failed")

    def fake_call_gemini(msgs, **kwargs):
        assert msgs == messages
        # you can also assert kwargs.get("model_name") == "gemini-2.0-flash"
        return "fallback response"

    monkeypatch.setattr(llm_router, "call_openai", fake_call_openai)
    monkeypatch.setattr(llm_router, "call_gemini", fake_call_gemini)

    result = llm_router.call_llm(messages)

    assert result["content"] == "fallback response"
    assert result["provider"] == "Google Gemini"
    assert result["fallback_used"] is True
    assert result["model"] == "gemini-2.0-flash"