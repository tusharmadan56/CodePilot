"""Command-line entrypoint: python agent.py "your task" --root ./project"""

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.backend import LocalBackend
from agent.graph import build_graph


def extract_text(content) -> str:
    
    if isinstance(content, str):
        return content

    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            texts.append(part["text"])
    return "".join(texts)


def main(task: str, root: str = ".", max_iters: int = 25):
    """Run the CodePilot agent on a task inside the given project directory."""
    load_dotenv()

    backend = LocalBackend(root)
    graph = build_graph(backend, max_iters)

    print(f"Task: {task}")
    print(f"Root: {backend.root}\n")

    result = graph.invoke({"messages": [HumanMessage(task)], "iterations": 0})

    final = result["messages"][-1]
    print(extract_text(final.content))


if __name__ == "__main__":
    typer.run(main)
