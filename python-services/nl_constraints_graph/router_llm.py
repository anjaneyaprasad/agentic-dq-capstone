from typing import Dict, Any
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage
from llm_router import call_llm


class RouterLLM(Runnable):

    def invoke(self, input: str | Dict[str, Any], config=None) -> AIMessage:
        """
        Wrapper so LangGraph/LC uses our fallback LLM.
        Accepts raw strings or LangChain-style message format.
        """

        # If LangGraph sends a raw string → treat as user message
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]

        # If LangGraph provides dict with messages list
        elif isinstance(input, dict) and "messages" in input:
            # Normalize LangChain message format (AIMessage/HumanMessage) to router style
            messages = []
            for msg in input["messages"]:
                role = "assistant" if isinstance(msg, AIMessage) else "user"
                messages.append({"role": role, "content": msg.content})

        else:
            raise ValueError(f"Unsupported input format for RouterLLM: {input}")

        # Call your fallback-enabled router
        result = call_llm(messages)

        # LangChain expects an AIMessage object back
        return AIMessage(content=result["text"])