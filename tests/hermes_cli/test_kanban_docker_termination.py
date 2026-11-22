"""Regression tests: a terminated kanban worker's Docker container must not outlive it.

The card id and run id form the container's identity: ``task_runs.id`` is a
per-board counter, so a run id alone can match a live container on another
board sharing the same Docker daemon.
"""

import subprocess

import pytest

from hermes_cli import kanban_docker_cleanup as cleanup
from tools.environments import docker as docker_env


def _result(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def test_cleanups_only_containers_of_the_exact_run(monkeypatch):
    calls = []

    def fake_query(argv, timeout):
        calls.append((list(argv), timeout))
        if "ps" in argv:
            return _result(0, "abc123\ndef456\n")
        return _result(0)

    assert cleanup.remove_kanban_run_containers("t_card42", 42, query=fake_query) == 2
    ps_args = next(a for a, _ in calls if "ps" in a)
    assert "label=hermes-agent=1" in ps_args
    assert "label=hermes-kanban-run=42" in ps_args
    assert "label=hermes-kanban-task=t_card42" in ps_args
    assert [a[-1] for a, _ in calls if "rm" in a] == ["abc123", "def456"]
    assert all(t <= 15 for _, t in calls)


def test_same_run_id_on_other_card_is_not_removed(monkeypatch):
    """Two boards can both have run 86; terminating one must spare the other."""
    listings = []

    def fake_query(argv, timeout):
        if "ps" in argv:
            filters = [a for a in argv if a.startswith("label=")]
            listings.append(filters)
            if "label=hermes-kanban-task=t_cardA" in filters:
                return _result(0, "container-A\n")
            if "label=hermes-kanban-task=t_cardB" in filters:
                return _result(0, "container-B\n")
            return _result(0, "both\n")
        return _result(0)

    assert cleanup.remove_kanban_run_containers("t_cardA", 86, query=fake_query) == 1
    assert all("label=hermes-kanban-run=86" in f and "label=hermes-kanban-task=t_cardA" in f
               for f in listings)
    assert all("label=" in f for f in listings[0])
    assert "label=hermes-kanban-task=t_cardB" not in listings[0]


def test_missing_identity_refuses_all_removal():
    calls = []

    def record(argv, timeout):
        calls.append(argv)
        return _result(0, "should-not-be-removed\n")

    for task, run in [(None, 42), ("t_card", None), ("t_card", "not-a-run"),
                      ("t_card; rm -rf /", 42)]:
        assert cleanup.remove_kanban_run_containers(task, run, query=record) == 0
    assert calls == []  # no docker ps, no docker rm


def test_docker_failure_and_timeout_are_swallowed():
    def boom(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    assert cleanup.remove_kanban_run_containers("t_card", 7, query=boom) == 0
    assert cleanup.remove_kanban_run_containers(
        "t_card", 7, query=lambda a, t: _result(1, err="daemon down")) == 0


def test_cleanup_resolves_podman_via_find_docker(monkeypatch):
    seen = []

    def fake_query(argv, timeout):
        seen.append(argv)
        return _result(0, "")

    monkeypatch.setattr(cleanup, "_resolve_docker_exe", lambda: "/usr/bin/podman")
    cleanup.remove_kanban_run_containers("t_card", 7, query=fake_query)
    assert seen[0][0] == "/usr/bin/podman"


def test_kanban_labels_present_only_for_kanban_worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_card1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    assert docker_env._kanban_labels(persist_across_processes=False) == {
        "hermes-kanban-task": "t_card1", "hermes-kanban-run": "42"}
    assert docker_env._kanban_labels(persist_across_processes=True) == {}
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    assert docker_env._kanban_labels(persist_across_processes=False) == {}
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    assert docker_env._kanban_labels(persist_across_processes=False) == {}  # interactive/gateway/cron unchanged


def _kanban_run_labels_from_docker(monkeypatch, persist):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_card1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "99")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def capture(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 1 and cmd[1] == "run":
            calls.append(cmd)
            return _result(0, "cid\n")
        return _result(0)

    monkeypatch.setattr(docker_env.subprocess, "run", capture)
    docker_env._cgroup_limits_ok = True
    docker_env.DockerEnvironment(
        image="python:3.11", cwd="/tmp", timeout=60, task_id="session-abc",
        persist_across_processes=persist,
    )
    assert calls
    return [calls[0][i + 1] for i, a in enumerate(calls[0][:-1])
            if a in ("--label", "-l")]


def test_docker_run_stamps_task_and_run_labels_on_kanban_worker(monkeypatch):
    labels = _kanban_run_labels_from_docker(monkeypatch, persist=False)
    assert "hermes-kanban-run=99" in labels
    assert "hermes-kanban-task=t_card1" in labels


def test_persistent_shared_container_has_no_kanban_labels(monkeypatch):
    """A profile-wide persistent container is shared; no per-run kill labels."""
    labels = _kanban_run_labels_from_docker(monkeypatch, persist=True)
    assert all(not label.startswith("hermes-kanban-") for label in labels)


@pytest.mark.parametrize("alive,expected", [(False, [("t_card77", 77)]), (True, [])])
def test_terminate_reclaimed_worker_container_identity(monkeypatch, alive, expected):
    from hermes_cli import kanban_db_dispatch as kbd

    removed = []
    monkeypatch.setattr(kbd, "_kill_fn", lambda fn: (lambda pid, sig: None))
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, started_at=None: alive)
    monkeypatch.setattr(kbd, "_poll_worker_exit", lambda *a, **kw: False)
    monkeypatch.setattr(kbd._kb, "_host_prefix", lambda: "testhost:")
    import hermes_cli.kanban_docker_cleanup as mod
    monkeypatch.setattr(
        mod, "remove_kanban_run_containers",
        lambda task, run: removed.append((task, run)) or 1)

    info = kbd._terminate_reclaimed_worker(
        12345, "testhost:lock", started_at=1000, run_id=77, task_id="t_card77")
    assert info["terminated"] is (not alive)
    assert removed == expected


def test_terminate_without_card_id_skips_cleanup(monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd

    removed = []
    monkeypatch.setattr(kbd, "_kill_fn", lambda fn: (lambda pid, sig: None))
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, started_at=None: False)
    monkeypatch.setattr(kbd._kb, "_host_prefix", lambda: "testhost:")
    import hermes_cli.kanban_docker_cleanup as mod
    monkeypatch.setattr(
        mod, "remove_kanban_run_containers",
        lambda task, run: removed.append((task, run)) or 1)

    info = kbd._terminate_reclaimed_worker(
        12345, "testhost:lock", started_at=1000, run_id=77)
    assert info["terminated"] is True
    assert removed == []
