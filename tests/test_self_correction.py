import os
import subprocess
import sys

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.backend import LocalBackend
from agent.graph import build_graph


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1",
    reason="live LLM test; set RUN_LLM_TESTS=1 to run (uses Gemini quota)",
)

BUGGY_CODE = """def add(a, b):
    return a - b
"""

TEST_CODE = """from calc import add

def test_add():
    assert add(2, 3) == 5
"""


def test_agent_fixes_a_failing_test(tmp_path):
    load_dotenv()

    backend = LocalBackend(tmp_path)
    backend.write_file("calc.py", BUGGY_CODE)
    backend.write_file("test_calc.py", TEST_CODE)

    task = "Run the tests with pytest. They fail. Fix the code so the tests pass."
    graph = build_graph(backend, max_iters=25)
    graph.invoke({"messages": [HumanMessage(task)], "iterations": 0})

    # Judge by reality, not by what the agent claimed: run the tests ourselves.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
