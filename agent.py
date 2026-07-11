"""Command-line entrypoint: python agent.py "your task" --root ./project"""

import uuid
from typing import Optional

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.types import Command

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
        return f"writing {args.get('path', '')}"
    if name == "edit_file":
        return f"editing {args.get('path', '')}"
    if name == "list_files":
        return f"listing {args.get('directory', '.')}"
    if name == "run_command":
        return f"running: {args.get('cmd', '')}"
    if name == "ask_user":
        return "asking you a question"
    return f"{name} {args}"


def print_diff(diff: str):
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            typer.secho(f"  {line}", fg=typer.colors.GREEN)
        elif line.startswith("-") and not line.startswith("---"):
            typer.secho(f"  {line}", fg=typer.colors.RED)
        elif line.startswith("@@"):
            typer.secho(f"  {line}", fg=typer.colors.CYAN)
        else:
            typer.secho(f"  {line}", fg=typer.colors.BRIGHT_BLACK)


def answer_interrupt(request):
    # ask_user sends {question, options}; the write gate sends {action, path, diff};
    # the command gate sends a plain string.
    if isinstance(request, dict) and "question" in request:
        typer.secho(f"\n  {request['question']}", fg=typer.colors.MAGENTA, bold=True)
        options = request.get("options") or []
        for number, option in enumerate(options, 1):
            typer.echo(f"    {number}. {option}")

        raw = typer.prompt("  your choice").strip()
        if options and raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        return raw

    if isinstance(request, dict) and "diff" in request:
        typer.secho(f"\n  {request['action']}: {request['path']}",
                    fg=typer.colors.YELLOW, bold=True)
        print_diff(request["diff"])
        return typer.confirm(
            typer.style("  approve?", fg=typer.colors.YELLOW), default=True)

    return typer.confirm(
        typer.style(f"  approve {request}?", fg=typer.colors.YELLOW),
        default=True,
    )


def run_task(graph, task: str, config: dict):
    stream_input = {"messages": [HumanMessage(task)], "iterations": 0}

    final_text = ""
    try:
        while True:
            interrupted = False
            for chunk in graph.stream(stream_input, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    request = chunk["__interrupt__"][0].value
                    stream_input = Command(resume=answer_interrupt(request))
                    interrupted = True
                    break

                agent_update = chunk.get("agent")
                if not agent_update:
                    continue
                for message in agent_update["messages"]:
                    if getattr(message, "tool_calls", None):
                        for call in message.tool_calls:
                            typer.secho(f"  -> {describe_tool_call(call)}", fg=typer.colors.CYAN)
                    else:
                        final_text = extract_text(message.content)

            if not interrupted:
                break
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


def chat(graph, root):
    typer.secho("CodePilot session - type a task, /clear for a fresh thread, "
                "exit to quit", bold=True)
    typer.secho(f"Root: {root}\n", fg=typer.colors.BRIGHT_BLACK)

    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    while True:
        try:
            line = typer.prompt("codepilot", prompt_suffix="> ").strip()
        except typer.Abort:
            typer.echo("\nbye")
            return

        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            typer.echo("bye")
            return
        if line == "/clear":
            config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            typer.secho("(fresh session started)\n", fg=typer.colors.BRIGHT_BLACK)
            continue

        try:
            run_task(graph, line, config)
        except KeyboardInterrupt:
            typer.secho("\n(task cancelled - session continues)", fg=typer.colors.YELLOW)
        print()


def main(task: Optional[str] = typer.Argument(None, help="Task to run; omit to start a chat session"),
         root: str = ".", max_iters: int = 25,
         yes: bool = typer.Option(False, "--yes", "-y",
                                  help="Auto-approve all writes and commands")):
    """Run the CodePilot agent on a task inside the given project directory."""
    load_dotenv()

    backend = LocalBackend(root)
    graph = build_graph(backend, max_iters, require_approval=not yes)

    if task is None:
        chat(graph, backend.root)
        return

    typer.secho(f"Task: {task}", bold=True)
    typer.secho(f"Root: {backend.root}\n", fg=typer.colors.BRIGHT_BLACK)
    run_task(graph, task, {"configurable": {"thread_id": uuid.uuid4().hex}})


if __name__ == "__main__":
    typer.run(main)
