# python-services/nl_constraints_graph/llm_client.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # load .env variables into environment

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", 0)),
        api_key=os.getenv("OPENAI_API_KEY")
    )
