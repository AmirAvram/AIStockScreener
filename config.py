import os
from google.genai import types

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,  # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)
DEFAULT_AGENT_DESCRIPTION = "A simple agent that can answer general questions."
DEFAULT_AGENT_INSTRUCTION = (
    "You are a helpful assistant. Use Google Search for current info or if unsure."
)
