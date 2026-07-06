"""Command-line entrypoint: python agent.py "your task" --root ./project"""

import sys

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from agent.backend import LocalBackend
from agent.graph import build_graph
from agent.llm import DEFAULT_MODEL, is_rate_limit


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


def describe_tool_call(call) -> str:
    name = call["name"]
    args = call["args"]
    if name == "read_file":
        return f"reading {args.get('path', '')}"
    if name == "write_file":
        return f"editing {args.get('path', '')}"
    if name == "list_files":
        return f"listing {args.get('directory', '.')}"
    if name == "run_command":
        return f"running: {args.get('cmd', '')}"
    return f"{name} {args}"


def main(task: str, root: str = ".", max_iters: int = 25):
    """Run the CodePilot agent on a task inside the given project directory."""
    load_dotenv()

    backend = LocalBackend(root)
    graph = build_graph(backend, max_iters)

    print(f"Task: {task}")
    print(f"Root: {backend.root}\n")

    final_text = ""
    try:
        for chunk in graph.stream({"messages": [HumanMessage(task)], "iterations": 0},
                                  stream_mode="updates"):
            agent_update = chunk.get("agent")
            if not agent_update:
                continue

            for message in agent_update["messages"]:
                if getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        print(f"  -> {describe_tool_call(call)}")
                else:
                    final_text = extract_text(message.content)
    except ChatGoogleGenerativeAIError as error:
        if not is_rate_limit(error):
            raise
        print(f"\nOut of free-tier quota for {DEFAULT_MODEL}. It resets at midnight "
              f"Pacific - try again later, or switch models in agent/llm.py.",
              file=sys.stderr)
        raise typer.Exit(code=1)

    print()
    if final_text:
        print(final_text)
    else:
        print("(stopped without a final summary - try raising --max-iters)")


if __name__ == "__main__":
    typer.run(main)
