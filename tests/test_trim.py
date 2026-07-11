from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import _trim, _MAX_TRANSCRIPT


def exchange(task_number):
    # one task: the user asks, the agent calls a tool, the tool answers, the agent replies
    return [
        HumanMessage(f"task {task_number}"),
        AIMessage(f"calling tool for {task_number}"),
        ToolMessage(f"result {task_number}", tool_call_id=str(task_number)),
        AIMessage(f"done with {task_number}"),
    ]


def test_short_transcript_is_untouched():
    messages = exchange(1) + exchange(2)
    assert _trim(messages) == messages


def test_long_transcript_is_cut_at_a_task_boundary():
    messages = []
    for n in range(20):  # 80 messages, well over the cap
        messages.extend(exchange(n))

    kept = _trim(messages)

    assert len(kept) <= _MAX_TRANSCRIPT
    assert isinstance(kept[0], HumanMessage)
    assert kept[-1] == messages[-1]  # newest messages survive


def test_one_giant_task_never_starts_with_a_tool_result():
    messages = [HumanMessage("huge task")]
    for n in range(60):
        messages.append(AIMessage(f"step {n}"))
        messages.append(ToolMessage(f"result {n}", tool_call_id=str(n)))

    kept = _trim(messages)

    assert kept
    assert not isinstance(kept[0], ToolMessage)
