"""Sole model-construction point"""

from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MODEL = "gemini-2.5-flash"


def build_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    # temperature 0 
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)
