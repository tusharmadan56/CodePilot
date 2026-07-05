"""Tools the model can call to inspect and edit the project."""

from langchain_core.tools import tool

from agent.backend import Backend


def make_tools(backend: Backend):
    @tool
    def list_files(directory: str = ".") -> str:
        """List the project's files, relative to the root. Skips node_modules, .git, and caches."""
        return backend.list_files(directory)

    @tool
    def read_file(path: str) -> str:
        """Read a file and return its contents."""
        return backend.read_file(path)

    @tool
    def write_file(path: str, text: str) -> str:
        """Create or overwrite a file. Missing parent directories are created."""
        return backend.write_file(path, text)

    @tool
    def run_command(cmd: str) -> str:
        """Run a shell command in the project root. Returns stdout, stderr, and the exit code."""
        return backend.run_command(cmd)

    return [list_files, read_file, write_file, run_command]
