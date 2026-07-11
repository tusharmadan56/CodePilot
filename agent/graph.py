"""The agent loop: the model picks actions, tools run them, repeat until done."""

from typing import Annotated, TypedDict

from langchain_core.messages import (BaseMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.backend import Backend
from agent.llm import build_llm, invoke_with_retry
from agent.tools import make_tools

SYSTEM_PROMPT = """You are CodePilot, an autonomous coding agent working in a single project directory through a terminal.

Not every message is a coding task. If the user greets you, chats, or asks something that needs no project work, just reply in plain text - no tools. For actual tasks, work through this loop:
1. Understand - use list_files and read_file to learn the project and the relevant code. Never guess a file's contents.
2. Plan - decide the smallest change that fully solves the task.
3. Edit - use edit_file to change existing files (copy the snippet exactly, then replace it) and write_file only for new files or full rewrites. Preserve the existing style; change only what the task needs.
4. Verify - if the project has tests, a build, or a run step, use run_command to check your work. Read the output; if it fails, diagnose, fix, and re-run.

Rules:
- Base every edit on what you actually read, not on assumptions.
- Keep changes minimal and focused; don't refactor unrelated code.
- Work autonomously: make reasonable choices yourself (file names, structure, defaults) and note assumptions in your final summary. Only when a decision genuinely needs the user's judgment - ambiguous requirements or several meaningfully different approaches - use the ask_user tool with 2-4 options. Never ask a question in plain text; a plain-text reply ends the run.
- When a command or edit fails, use the error output to correct yourself and try again.
- Write the best code for the task, including interactive programs that use input(). You run in a non-interactive terminal, so verify such code by exercising its functions directly with sample values (e.g. python -c "from solver import solve; print(solve(1, 5, 6))") rather than running an entry point that waits for input you can't type.

When the task is complete and verified, stop calling tools and reply with a brief summary of what you changed."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int


_MAX_TRANSCRIPT = 40


def _trim(messages: list[BaseMessage]) -> list[BaseMessage]:
    # Long chat sessions grow the transcript without bound; send the model only
    # a recent window, cut at a task boundary so no tool result arrives orphaned.
    if len(messages) <= _MAX_TRANSCRIPT:
        return messages

    kept = list(messages[-_MAX_TRANSCRIPT:])
    while kept and not isinstance(kept[0], HumanMessage):
        kept.pop(0)

    if not kept:  # one giant task fills the window - just avoid orphaned tool results
        kept = list(messages[-_MAX_TRANSCRIPT:])
        while kept and isinstance(kept[0], ToolMessage):
            kept.pop(0)

    return kept


def build_graph(backend: Backend, max_iters: int = 25, checkpointer=None,
                require_approval: bool = False):
    if checkpointer is None:
        checkpointer = InMemorySaver()

    tools = make_tools(backend, require_approval)
    model = build_llm().bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT)] + _trim(state["messages"])
        response = invoke_with_retry(model, messages)
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def should_continue(state: AgentState) -> str:
        if state["iterations"] >= max_iters:
            return "end"

        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
