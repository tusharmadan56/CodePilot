"""Tools the model can call to inspect and edit the project."""

from langchain_core.tools import tool
from langgraph.types import interrupt

from agent.backend import Backend


def _safe(func, *args):
    # Hand tool failures back to the model as text so it can recover (e.g. a
    # missing file or bad path) instead of the exception crashing the run.
    try:
        return func(*args)
    except Exception as error:
        return f"Error: {error}"


def make_tools(backend: Backend, require_approval: bool = False):
    @tool
    def list_files(directory: str = ".") -> str:
        """List the project's files, relative to the root. Skips node_modules, .git, and caches."""
        return _safe(backend.list_files, directory)

    @tool
    def read_file(path: str) -> str:
        """Read a file and return its contents."""
        return _safe(backend.read_file, path)

    @tool
    def write_file(path: str, text: str) -> str:
        """Create or overwrite a file. Missing parent directories are created."""
        if require_approval and not interrupt(f"write file: {path}"):
            return "The user denied this write."
        return _safe(backend.write_file, path, text)

    @tool
    def run_command(cmd: str) -> str:
        """Run a shell command in the project root. Returns stdout, stderr, and the exit code."""
        if require_approval and not interrupt(f"run command: {cmd}"):
            return "The user denied this command."
        return _safe(backend.run_command, cmd)

    return [list_files, read_file, write_file, run_command]
