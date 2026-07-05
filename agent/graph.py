"""The agent loop: the model picks actions, tools run them, repeat until done."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.backend import Backend
from agent.llm import build_llm
from agent.tools import make_tools


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(backend: Backend):
    tools = make_tools(backend)
    model = build_llm().bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
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

    return graph.compile()
