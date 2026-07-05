""" model-construction """

import re
import sys
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

DEFAULT_MODEL = "gemini-2.5-flash"


def build_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    # temperature 0
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def invoke_with_retry(model, messages, max_retries: int = 5):
    for attempt in range(1, max_retries + 1):
        try:
            return model.invoke(messages)
        except ChatGoogleGenerativeAIError as error:
            if not _is_rate_limit(error) or attempt == max_retries:
                raise

            delay = _retry_delay_seconds(error)
            print(f"Rate limited; waiting {delay:.0f}s "
                  f"(attempt {attempt}/{max_retries})", file=sys.stderr)
            time.sleep(delay)


def _is_rate_limit(error) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _retry_delay_seconds(error, default: float = 15.0) -> float:
    
    match = re.search(r"retry in ([\d.]+)s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1
    return default
