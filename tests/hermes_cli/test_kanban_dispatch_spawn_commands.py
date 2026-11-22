"""Config-mapped spawn commands for non-profile kanban assignees."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_spawn_commands as ksc


AGENT_SCRIPT = """\
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps({
    "argv": sys.argv[2:],
    "env": {k: os.environ.get(k) for k in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB", "HERMES_KANBAN_CLAIM_LOCK", "HERMES_TENANT",
        "ANTHROPIC_API_KEY", "HERMES_PROFILE")},
    "cwd": str(pathlib.Path.cwd()),
}))
"""


@pytest.fixture(autouse=True)
def _no_real_home_probe(monkeypatch, tmp_path):
    # build_subprocess_env prepends a bin dir derived from the source tree;
    # this test tree lives under the real HERMES_HOME and the test guard
    # refuses file IO there. The bin-dir detail is irrelevant to these tests.
    from tools.environments import local as local_env
    monkeypatch.setattr(
        local_env, "_prepend_hermes_bin_dir", lambda value: value
    )
    (tmp_path / "worker.py").write_text(AGENT_SCRIPT)
    # Board log dirs are irrelevant here; a single scratch log keeps child
    # stdout/stderr available for debugging without touching the real home.
    monkeypatch.setattr(
        kbd, "_open_worker_log", lambda task, board: open(tmp_path / "worker.log", "ab")
    )
    # These unit tests exercise argv substitution/env, not systemd topology.
    # The restart-safe wrapper is tested explicitly below.
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda task, cmd: cmd)


def _isolation_marker_env(monkeypatch):
    """Make systemd_user_bus_env a no-op while recording its invocation."""
    import tools.process_registry as pr
    seen = []
    def _bus_env(env):
        seen.append(dict(env))
        return env
    monkeypatch.setattr(pr, "systemd_user_bus_env", _bus_env)
    return seen


def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()


def _agent(tmp_path) -> str:
    script = tmp_path / "worker.py"
    script.write_text(AGENT_SCRIPT)
    return f"{sys.executable} {script} {tmp_path / 'capture.json'} {{task_id}}"


def _read_capture(capture: Path, tmp_path: Path) -> dict:
    deadline = time.monotonic() + 5.0  # detached child: wait for its write, not a fixed sleep
    while time.monotonic() < deadline:
        if capture.exists():
            try:
                return json.loads(capture.read_text())
            except (ValueError, OSError):
                pass
        time.sleep(0.02)
    log = tmp_path / "worker.log"
    raise AssertionError(f"worker did not write {capture}; log={log.read_text() if log.exists() else '<none>'}")


def _task(task_id, workspace) -> SimpleNamespace:
    return SimpleNamespace(
        id=task_id, current_run_id=7, claim_lock="host:1:abc", tenant="lab",
        branch_name=None, skills=None,
    )


def test_resolve_spawn_commands_validation(caplog):
    resolved = ksc.resolve_spawn_commands({
        "good": "/bin/echo {task_id} --board '{board}'",
        "list": ["/bin/echo", "{workspace}", "{run_id}"],
        "bad-field": "/bin/echo {nope}",
        "empty": "",
        "bad-type": 42,
    })
    assert resolved == {
        "good": ["/bin/echo", "{task_id}", "--board", "{board}"],
        "list": ["/bin/echo", "{workspace}", "{run_id}"],
    }
    assert "unknown placeholder" in caplog.text
    assert "empty command" in caplog.text
    assert "expected a string" in caplog.text


def test_resolve_spawn_commands_non_mapping_is_empty(caplog):
    assert ksc.resolve_spawn_commands("not-a-mapping") == {}
    assert "must be a mapping" in caplog.text


def test_command_spawn_substitutes_inside_argv_and_scrubs_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-should-not-pass")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture = tmp_path / "capture.json"
    template = (
        f"{sys.executable} {tmp_path / 'worker.py'} {capture} "
        "--task '{task_id}' --ws '{workspace}'"
    )
    task_id = "t with spaces; $(touch /tmp/nope)"
    spawn = ksc.make_command_spawn(ksc.resolve_spawn_commands({"lane": template})["lane"])
    pid = spawn(_task(task_id, workspace), str(workspace), board="board-a")
    assert isinstance(pid, int) and pid > 0
    data = _read_capture(capture, tmp_path)
    assert data["argv"] == ["--task", task_id, "--ws", str(workspace)]
    assert data["cwd"] == str(workspace)
    assert data["env"]["HERMES_KANBAN_TASK"] == task_id
    assert data["env"]["HERMES_KANBAN_RUN_ID"] == "7"
    assert data["env"]["HERMES_KANBAN_CLAIM_LOCK"] == "host:1:abc"
    assert data["env"]["HERMES_TENANT"] == "lab"
    assert data["env"]["HERMES_KANBAN_BOARD"] == "board-a"
    assert data["env"]["ANTHROPIC_API_KEY"] is None
    assert data["env"]["HERMES_PROFILE"] is None


def test_list_form_keeps_shell_metacharacters_in_one_argument(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture = tmp_path / "capture.json"
    argv = [sys.executable, str(tmp_path / "worker.py"), str(capture), "{task_id}"]
    spawn = ksc.make_command_spawn(argv)
    task_id = "a;echo b &"
    spawn(_task(task_id, workspace), str(workspace), board="b")
    assert _read_capture(capture, tmp_path)["argv"] == [task_id]


def test_substitution_is_single_pass(tmp_path, monkeypatch):
    """Inserted values must never be scanned again for placeholders."""
    # This is the case the lab-hub trial reproduced: the workspace path itself
    # contains {board} and {run_id}, and the task id contains {workspace}.
    workspace = tmp_path / "ws dir {board} {run_id}"
    workspace.mkdir()
    capture = tmp_path / "capture.json"
    argv = [
        sys.executable, str(tmp_path / "worker.py"), str(capture),
        "--task={task_id}", "{workspace}", "{board}", "{run_id}",
    ]
    seen = _isolation_marker_env(monkeypatch)
    spawn = ksc.make_command_spawn(argv)
    task_id = "task {workspace}; rm -rf / $(id)"
    board = "bd"
    task = _task(task_id, workspace)
    spawn(task, str(workspace), board=board)
    data = _read_capture(capture, tmp_path)
    assert len(data["argv"]) == 4
    assert data["argv"] == [
        f"--task={task_id}", str(workspace), board, str(task.current_run_id),
    ]
    assert seen  # bus env applied


def test_restart_safe_wrapper_and_infrastructure_failure(tmp_path, monkeypatch):
    """Command workers get the same restart-safe scope as profile workers.

    A missing scope is infrastructure: spawn_failed is recorded but the card's
    retry budget is not charged, and the tick continues.
    """
    _isolated_home(tmp_path, monkeypatch)
    from tools.process_registry import RestartSafeScopeUnavailable

    calls = []
    def _wrap(task, cmd):
        calls.append(list(cmd))
        return ["/wrapped", *cmd]
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", _wrap)
    seen = _isolation_marker_env(monkeypatch)

    def _fail(task, cmd):
        raise RestartSafeScopeUnavailable("no systemd scope")

    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", _fail)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="remote-hub")
        res = kbd.dispatch_once(
            conn, spawn_commands={"remote-hub": ["/nonexistent", "{task_id}"]}
        )
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
    assert calls == []
    assert seen == []
    assert res.spawned == [] and res.auto_blocked == []
    assert task is not None and task.status == "ready"
    assert task.consecutive_failures == 0
    assert any(
        e.kind == "spawn_failed" and "infrastructure" in str(getattr(e, "payload", None))
        for e in events
    )
    # Normal argv wrapping keeps the wrapper's command, proving the seam.
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", _wrap)
    spawn = ksc.make_command_spawn(["/bin/echo", "{task_id}"])
    monkeypatch.setattr(
        "subprocess.Popen", lambda *a, **kw: SimpleNamespace(pid=4242)
    )
    assert spawn(_task("abc", tmp_path), str(tmp_path), board="b") == 4242
    assert calls == [["/bin/echo", "abc"]]


def test_missing_binary_is_spawn_failure_not_tick_exception(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="remote-hub")
        commands = {"remote-hub": ["/nonexistent/definitely-not-here", "{task_id}"]}
        res = kbd.dispatch_once(conn, spawn_commands=commands)
        task = kb.get_task(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert res.spawned == []
    assert task is not None and task.status == "ready"
    assert "spawn_failed" in kinds


def test_configured_non_profile_assignee_is_claimed_and_spawned(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: False)
    cmd = _agent(tmp_path)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="remote-hub")
        res = kbd.dispatch_once(conn, spawn_commands={"remote-hub": cmd})
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        details = [(e.kind, getattr(e, "payload", None)) for e in events]
    assert len(res.spawned) == 1, f"res={res!r} details={details}"
    assert res.spawned[0][0] == tid and res.spawned[0][1] == "remote-hub"
    workspace = res.spawned[0][2]
    assert res.skipped_nonspawnable == []
    assert task is not None and task.status == "running"
    assert task.worker_pid and task.worker_pid > 0
    assert any(e.kind == "claimed" for e in events)
    data = _read_capture(tmp_path / "capture.json", tmp_path)
    assert data["argv"] == [tid]
    assert data["cwd"] == workspace
    assert data["env"]["HERMES_KANBAN_TASK"] == tid
    assert data["env"]["HERMES_KANBAN_RUN_ID"] == str(task.current_run_id)


def test_key_absent_keeps_non_profile_skipped(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="remote-hub")
        res = kbd.dispatch_once(conn, spawn_commands={})
    assert res.skipped_nonspawnable == [tid]
    assert res.spawned == []


def test_explicit_spawn_fn_wins_over_config(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: False)
    seen = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="demo", assignee="remote-hub")
        res = kbd.dispatch_once(
            conn, spawn_commands={"remote-hub": ["/nonexistent"]},
            spawn_fn=lambda task, ws, board=None: seen.append(task.id) or 999,
        )
        task = kb.get_task(conn, tid)
    assert seen == [tid]
    assert task is not None and task.worker_pid == 999
    assert len(res.spawned) == 1 and res.spawned[0][0] == tid


def test_gateway_and_cli_entry_points_resolve_key(monkeypatch):
    from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _resolve_dispatcher_settings
    from hermes_cli import kanban_ops

    commands = {"remote-hub": ["/bin/echo", "{task_id}"]}
    settings = _resolve_dispatcher_settings({"spawn_commands": commands}, kb)
    assert settings.spawn_commands == commands
    assert _DispatcherSettings(
        interval=1.0, max_spawn=None, max_in_progress=None, failure_limit=2,
        stale_timeout_seconds=0, reconcile_orphans=True, default_assignee=None,
        max_in_progress_per_profile=None,
    ).spawn_commands is None

    class Args:
        dry_run = True
        max = None
        json = False
        failure_limit = 2

    calls = []
    cfg = {"kanban": {"spawn_commands": commands}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(kanban_ops.kbc, "connect_closing", _FakeCtx)
    monkeypatch.setattr(
        kanban_ops.kbd, "dispatch_once",
        lambda conn, **kw: calls.append(kw) or kbd.DispatchResult(),
    )
    kanban_ops._cmd_dispatch(Args())  # type: ignore[arg-type]
    assert calls[0]["spawn_commands"] == commands


class _FakeCtx:
    def __enter__(self):
        return object()

    def __exit__(self, *_args):
        return False
