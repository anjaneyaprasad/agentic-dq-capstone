from start_up import *
import openai
import google.generativeai as genai


# load_dotenv()


def call_llm(messages):
    """
    Universal LLM entrypoint with automatic fallback:
        1. Try GPT-4o-mini
        2. If failure → use Gemini 2.0 Flash
    """

    try:
        print("🟢 Primary Model: GPT-4o-mini")
        content = call_openai(messages)
        return {
            "content": content,
            "model": os.getenv("PRIMARY_MODEL"),  # Indicate which model was used
            "provider": "OpenAI",
            "fallback_used": False
        }

    except Exception as error:
        print(f"⚠️ GPT-4o-mini failed: {error}")
        print("🟡 Switching to fallback model: Gemini 2.0 Flash")
        content = call_gemini(messages)
        return {
            "content": content,
            "model": os.getenv("FALLBACK_MODEL"),  # Indicate which model was used
            "provider": "Google Gemini",
            "fallback_used": True
        }


def call_openai(messages):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=os.getenv("PRIMARY_MODEL", "gpt-4o-mini"),
        messages=messages,
        max_tokens=1024,
        temperature=0
    )
    return response.choices[0].message.content


def call_gemini(messages):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Convert OpenAI message format → Gemini content format
    gemini_messages = []
    for msg in messages:
        role = "user" if msg["role"] != "assistant" else "model"
        gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})

    model_name = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(gemini_messages)

    return response.text