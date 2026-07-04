import pytest

from agent.backend import LocalBackend


@pytest.fixture
def backend(tmp_path):
    return LocalBackend(tmp_path)


def test_write_then_read_roundtrip(backend):
    backend.write_file("notes.txt", "hello")
    assert backend.read_file("notes.txt") == "hello"


def test_write_creates_missing_directories(backend):
    backend.write_file("src/deep/file.py", "x = 1")
    assert backend.read_file("src/deep/file.py") == "x = 1"


def test_list_files_is_sorted_and_skips_ignored_dirs(backend, tmp_path):
    backend.write_file("b.py", "")
    backend.write_file("a.py", "")

    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "dep.js").write_text("")

    assert backend.list_files(".") == "a.py\nb.py"


def test_list_files_on_empty_directory(backend):
    assert backend.list_files(".") == "(empty)"


def test_run_command_reports_output_and_exit_code(backend):
    result = backend.run_command("echo hello")
    assert "hello" in result
    assert "[exit code: 0]" in result


def test_run_command_reports_nonzero_exit_code(backend):
    result = backend.run_command("exit 3")
    assert "[exit code: 3]" in result


@pytest.mark.parametrize("path", ["../outside.txt", "../../etc/passwd"])
def test_read_rejects_paths_outside_root(backend, path):
    with pytest.raises(ValueError):
        backend.read_file(path)


def test_write_rejects_paths_outside_root(backend):
    with pytest.raises(ValueError):
        backend.write_file("../evil.txt", "pwned")
