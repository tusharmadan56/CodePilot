"""Command-line entrypoint: python agent.py "your task" --root ./project"""

import uuid

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

    typer.secho(f"Task: {task}", bold=True)
    typer.secho(f"Root: {backend.root}\n", fg=typer.colors.BRIGHT_BLACK)

    config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    final_text = ""
    try:
        for chunk in graph.stream({"messages": [HumanMessage(task)], "iterations": 0},
                                  config, stream_mode="updates"):
            agent_update = chunk.get("agent")
            if not agent_update:
                continue

            for message in agent_update["messages"]:
                if getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        typer.secho(f"  -> {describe_tool_call(call)}", fg=typer.colors.CYAN)
                else:
                    final_text = extract_text(message.content)
    except ChatGoogleGenerativeAIError as error:
        if not is_rate_limit(error):
            raise
        typer.secho(f"\nOut of free-tier quota for {DEFAULT_MODEL}. It resets at "
                    f"midnight Pacific - try again later, or switch models in "
                    f"agent/llm.py.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    print()
    if final_text:
        typer.secho(final_text, fg=typer.colors.GREEN)
    else:
        typer.secho("(stopped without a final summary - try raising --max-iters)",
                    fg=typer.colors.YELLOW)


if __name__ == "__main__":
    typer.run(main)
