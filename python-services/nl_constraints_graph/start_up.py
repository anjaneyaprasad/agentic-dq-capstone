"""
startup.py

Centralized initialization for:
- Warning suppression
- Environment variable loading
- API key validation (optional)
"""

import os
import warnings
from dotenv import load_dotenv


# -----------------------------
# 1) Load .env variables safely
# -----------------------------
load_dotenv()

# Optional: Validate keys exist (warn only)
missing = []
for key in ["OPENAI_API_KEY", "GEMINI_API_KEY"]:
    if not os.getenv(key):
        missing.append(key)

if missing:
    print(f"Missing environment variables: {', '.join(missing)}")


# -----------------------------
# 2) Suppress ONLY noisy warnings
# -----------------------------
# warnings.filterwarnings(
#     "ignore",
#     message="You are using a Python version .* google.api_core",
#     category=FutureWarning
# )

# You can add future suppressions here if needed:
# warnings.simplefilter("ignore", DeprecationWarning)


# -----------------------------
# 3) Optional: Confirm environment
# -----------------------------
print("Environment initialized")
print(f"Python runtime OK | LLM Primary: {os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')}")