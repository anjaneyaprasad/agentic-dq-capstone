from llm_router import call_llm

prompt = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Give me a one-line definition of data quality."}
]

response = call_llm(prompt)
print("\n🔍 Model Response:\n", response["content"])