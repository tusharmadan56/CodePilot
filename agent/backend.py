"""Filesystem and shell access, confined to a project root."""

import os
import signal
import subprocess
from pathlib import Path
from typing import Protocol

_IGNORED_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv",
                 ".pytest_cache", ".mypy_cache"}

_MAX_OUTPUT_CHARS = 4000
_COMMAND_TIMEOUT = 120


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text

    half = _MAX_OUTPUT_CHARS // 2
    omitted = len(text) - 2 * half
    head = text[:half]
    tail = text[-half:]
    return f"{head}\n... [{omitted} characters truncated] ...\n{tail}"


def _kill_process_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


class Backend(Protocol):
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, text: str) -> str: ...
    def list_files(self, directory: str = ".") -> str: ...
    def run_command(self, cmd: str) -> str: ...


class LocalBackend:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        # resolve the path and then refuse anything that climbs outside the project root.
        
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root):
            raise ValueError(f"path escapes project root: {path!r}")
        return full

    def read_file(self, path: str) -> str:
        full = self._resolve(path)
        return full.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, text: str) -> str:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8")
        return f"wrote {path} ({len(text)} bytes)"

    def list_files(self, directory: str = ".") -> str:
        base = self._resolve(directory)

        files = []
        for p in base.rglob("*"):
            if not p.is_file():
                continue

            rel = p.relative_to(self.root)
            if _IGNORED_DIRS & set(rel.parts):  # drop node_modules, .git, etc.
                continue

            files.append(rel.as_posix())

        files.sort()
        if not files:
            return "(empty)"
        return "\n".join(files)

    def run_command(self, cmd: str) -> str:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=self.root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name != "nt"),
        )

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=_COMMAND_TIMEOUT)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            timed_out = True

        output = (stdout + stderr).strip()
        if not output:
            output = "(no output)"
        output = _truncate(output)

        if timed_out:
            status = f"[timed out after {_COMMAND_TIMEOUT}s]"
        else:
            status = f"[exit code: {proc.returncode}]"

        return f"$ {cmd}\n{output}\n{status}"
