"""Tools the model can call to inspect and edit the project."""

import difflib

from langchain_core.tools import tool
from langgraph.types import interrupt

from agent.backend import Backend

_MAX_DIFF_LINES = 80


def _make_diff(old: str, new: str, path: str) -> str:
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
    ))

    if len(lines) > _MAX_DIFF_LINES:
        omitted = len(lines) - _MAX_DIFF_LINES
        lines = lines[:_MAX_DIFF_LINES]
        lines.append(f"... ({omitted} more diff lines)")

    return "\n".join(lines)


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
        if require_approval:
            try:
                old = backend.read_file(path)
            except Exception:
                old = ""  # new file

            request = {"action": "write file", "path": path,
                       "diff": _make_diff(old, text, path)}
            if not interrupt(request):
                return "The user denied this write."
        return _safe(backend.write_file, path, text)

    @tool
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace one exact snippet in an existing file. old_text must be copied
        exactly from the file (matching indentation) and must appear exactly once;
        include surrounding lines to make it unique. Prefer this over write_file
        for changing existing files."""
        if require_approval:
            current = _safe(backend.read_file, path)
            # Only ask the user when the edit can actually apply; otherwise fall
            # through and let the backend report the precise problem.
            if current.count(old_text) == 1:
                proposed = current.replace(old_text, new_text, 1)
                request = {"action": "edit file", "path": path,
                           "diff": _make_diff(current, proposed, path)}
                if not interrupt(request):
                    return "The user denied this edit."
        return _safe(backend.edit_file, path, old_text, new_text)

    @tool
    def run_command(cmd: str) -> str:
        """Run a shell command in the project root. Returns stdout, stderr, and the exit code."""
        if require_approval and not interrupt(f"run command: {cmd}"):
            return "The user denied this command."
        return _safe(backend.run_command, cmd)

    @tool
    def ask_user(question: str, options: list[str]) -> str:
        """Ask the user to decide something you cannot decide from the project itself,
        offering 2-4 short options. Use sparingly - make routine choices yourself."""
        answer = interrupt({"question": question, "options": options})
        return f"The user chose: {answer}"

    return [list_files, read_file, write_file, edit_file, run_command, ask_user]
