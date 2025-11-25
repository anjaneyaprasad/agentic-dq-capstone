# from start_up import PRIMARY_MODEL, FALLBACK_MODEL, OPENAI_API_KEY, GEMINI_API_KEY
import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai

load_dotenv()

def _get_model_settings():
    """
    Central place to fetch model names from environment with sensible defaults.
    Returns (primary_model, fallback_model) as strings.
    """
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")
    return primary, fallback

def call_llm(messages):
    """
    Universal LLM entrypoint with automatic fallback:
        1. Try GPT-4o-mini
        2. If failure → use Gemini 2.0 Flash
    """
    
    primary_model, fallback_model = _get_model_settings()

    try:
        print(f"Primary Model: {primary_model}")
        content = call_openai(messages, model_name=primary_model)
        return {
            "content": content,
            "model": primary_model, #os.getenv("PRIMARY_MODEL"),  # Indicate which model was used
            "provider": "OpenAI",
            "fallback_used": False
        }

    except Exception as error:
        print(f"{primary_model} failed: {error}")
        print(f" Switching to fallback model: {fallback_model}")
        content = call_gemini(messages, model_name=fallback_model)
        return {
            "content": content,
            "model": fallback_model,  # Indicate which model was used
            "provider": "Google Gemini",
            "fallback_used": True
        }


def call_openai(messages, model_name=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
    
    if model_name is None:
        primary_model, _ = _get_model_settings()
        model_name = primary_model
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1024,
        temperature=0
    )
    return response.choices[0].message.content


def call_gemini(messages, model_name=None):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Convert OpenAI message format → Gemini content format
    gemini_messages = []
    for msg in messages:
        role = "user" if msg["role"] != "assistant" else "model"
        gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        
    if model_name is None:
        _, fallback_model = _get_model_settings()
        model_name = fallback_model

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(gemini_messages)

    # model_name = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")
    # model = genai.GenerativeModel(model_name)
    # response = model.generate_content(gemini_messages)

    return response.text