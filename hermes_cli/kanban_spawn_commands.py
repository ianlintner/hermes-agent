"""Config-mapped spawn commands for kanban worker lanes.

``kanban.spawn_commands`` maps an assignee name to a command template.  An
assignee with a valid command is spawnable without being a Hermes profile; its
command *is* the worker and owns the kanban lifecycle through the API/CLI.
Validation happens once per config load (the config loader is mtime-cached):
bad entries are logged and dropped, never raised into a dispatcher tick.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

PLACEHOLDERS = ("task_id", "workspace", "board", "run_id")
_PLACEHOLDER_RE = re.compile(r"\{(task_id|workspace|board|run_id)\}")


def _validate_template(value: Any, assignee: str) -> Optional[list[str]]:
    """Return the split argv template, or None when the value is invalid."""
    if isinstance(value, str):
        try:
            argv = shlex.split(value)
        except ValueError as exc:
            logger.warning(f"kanban spawn_commands[{assignee!r}]: invalid shell quoting: {exc}")
            return None
    elif isinstance(value, (list, tuple)) and all(isinstance(part, str) for part in value):
        argv = list(value)
    else:
        logger.warning(f"kanban spawn_commands[{assignee!r}]: expected a string or a list of strings")
        return None
    if not argv:
        logger.warning(f"kanban spawn_commands[{assignee!r}]: empty command")
        return None
    known = set(PLACEHOLDERS)
    for part in argv:
        for field in _braced_fields(part):
            if field not in known:
                logger.warning(f"kanban spawn_commands[{assignee!r}]: unknown placeholder {{{field}}}")
                return None
    return argv


def _braced_fields(text: str) -> list[str]:
    """Return names inside a single pair of braces (``{name}``)."""
    fields: list[str] = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start < 0:
            break
        end = text.find("}", start + 1)
        if end < 0:
            break
        fields.append(text[start + 1:end])
        i = end + 1
    return fields


_validated_cache: dict[str, dict[str, list[str]]] = {}


def resolve_spawn_commands(raw: Any) -> dict[str, list[str]]:
    """Validate the ``kanban.spawn_commands`` mapping once per config load.

    The config loader caches file reads by mtime; this caches validation by the
    raw value too, so an invalid entry is reported once per load rather than
    once per dispatcher tick.  Invalid entries are excluded; anything not a
    mapping (including ``None``) yields an empty map.
    """
    try:
        key = repr(sorted(raw.items())) if isinstance(raw, Mapping) else repr(raw)
    except Exception:
        key = repr(raw)
    if key in _validated_cache:
        return dict(_validated_cache[key])
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        logger.warning("kanban.spawn_commands must be a mapping of assignee -> command template")
        _validated_cache[key] = {}
        return {}
    commands: dict[str, list[str]] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            logger.warning(f"kanban spawn_commands: invalid assignee name {name!r}")
            continue
        argv = _validate_template(value, name)
        if argv is not None:
            commands[name] = argv
    _validated_cache[key] = dict(commands)
    return commands


def configured_spawn_commands() -> dict[str, list[str]]:
    """Read and validate ``kanban.spawn_commands`` from the user config."""
    from hermes_cli.config_effective import load_user_config_effective

    try:
        cfg = load_user_config_effective(fail_closed=True) or {}
    except Exception:
        return {}
    kanban = cfg.get("kanban")
    if not isinstance(kanban, Mapping):
        return {}
    return resolve_spawn_commands(kanban.get("spawn_commands"))


def _worker_env(task: Any, workspace: str, board: Optional[str]) -> dict[str, str]:
    """Secrets-scrubbed dispatcher env plus the board/task pins a worker needs."""
    from hermes_cli import kanban_db as _kb
    from tools.environments.local import build_subprocess_env

    env = build_subprocess_env(scrub_secrets=True, inherit_profile_home=False)
    for key in ("HERMES_PROFILE", "HERMES_HOME", "HERMES_KANBAN_GOAL_MODE",
                "HERMES_KANBAN_GOAL_MAX_TURNS"):
        env.pop(key, None)
    env.pop("HERMES_TUI", None)
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    env["HERMES_KANBAN_DB"] = str(_kb.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(_kb.workspaces_root(board=board))
    env["HERMES_KANBAN_BOARD"] = _kb._normalize_board_slug(board) or _kb.get_current_board()
    return env


def make_command_spawn(argv: list[str]) -> Callable[..., Optional[int]]:
    """Build a spawn callable from a validated argv template.

    Placeholders are substituted *inside each argument* after the template has
    been split, so a value containing spaces or shell metacharacters can never
    add an argument or reach a shell.  The command runs detached with the
    task's workspace as its working directory.  OSError is converted to a
    RuntimeError so the dispatcher's normal spawn-failure accounting applies.
    """
    from hermes_cli import kanban_db as _kb

    def spawn(task: Any, workspace: str, *, board: Optional[str] = None) -> Optional[int]:
        values = {
            "task_id": str(task.id),
            "workspace": str(workspace),
            "board": str(board or _kb._normalize_board_slug(board) or _kb.get_current_board() or ""),
            "run_id": str(task.current_run_id) if task.current_run_id is not None else "",
        }
        cmd: list[str] = [
            _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], part) for part in argv
        ]
        env = _worker_env(task, workspace, board)
        from hermes_cli import kanban_db_dispatch as _kbd
        # A worker spawned by a managed systemd gateway must leave its cgroup
        # before startup, exactly like a profile worker; otherwise a gateway
        # restart kills the external worker and the card is charged as a crash.
        cmd = _kbd._restart_safe_worker_argv(task, cmd)
        from tools.process_registry import systemd_user_bus_env
        env = systemd_user_bus_env(env)
        log_file = _kbd._open_worker_log(task, board)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=workspace if os.path.isdir(workspace) else None,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                shell=False,
                start_new_session=True,  # detached: a worker outlives the dispatcher's tick
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if _kb._IS_WINDOWS else 0,
            )
        except OSError as exc:
            log_file.close()
            raise RuntimeError(f"kanban spawn command {cmd[0]!r} could not start: {exc}") from exc
        if _kb._IS_WINDOWS:
            from hermes_cli.kanban_db_dispatch import _live_worker_procs
            _live_worker_procs[proc.pid] = proc
        return proc.pid

    return spawn
