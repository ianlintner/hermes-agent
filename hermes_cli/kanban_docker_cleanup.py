"""Best-effort Docker cleanup after a kanban worker is terminated.

The worker's own atexit hook cannot be relied on: a kanban single-query worker's
SIGTERM handler calls ``os._exit(0)`` so a stuck worker dies immediately instead
of hanging its claim, and SIGKILL skips atexit entirely. The dispatcher therefore
removes containers by the identity stamped at creation time: the kanban *card*
id plus the *run* id. Both labels are required. ``task_runs.id`` is a per-board
counter, so a run id alone collides across boards sharing one Docker daemon; the
card id is a host-unique random ``t_<hex>``. Filtering on both keeps a respawned
run's container untouched, including on another board.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from tools.environments.remote_common import run_capture

logger = logging.getLogger(__name__)

# Label names owned by tools/environments/docker.py; kept here as literals so the dispatcher
# does not import the Docker backend just to clean up after a dead worker.
_KANBAN_RUN_LABEL_KEY = "hermes-kanban-run"
_KANBAN_TASK_LABEL_KEY = "hermes-kanban-task"
_LIST_TIMEOUT_SECONDS = 10
_REMOVE_TIMEOUT_SECONDS = 15


def _safe_label_value(value: object) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    if not text.replace("-", "").replace("_", "").replace(".", "").isalnum():
        logger.debug("kanban docker cleanup: refusing unsafe label value %r", text)
        return None
    return text


def _safe_run_id(value: object) -> Optional[str]:
    """Run ids are per-board numeric counters; refuse anything else."""
    text = _safe_label_value(value)
    return text if text and text.isdigit() else None


def remove_kanban_run_containers(
    task_id: object,
    run_id: object,
    *,
    docker_exe: Optional[str] = None,
    query: Optional[Callable[[list[str], float], object]] = None,
) -> int:
    """Remove containers stamped with both *task_id* and *run_id*; never raises.

    Both parts of the identity are mandatory: refusing an incomplete identity is
    what makes cross-board removal impossible. ``query`` is a test seam taking
    ``(argv, timeout)`` and returning an object with ``returncode``/``stdout``/
    ``stderr``. Docker absent, slow, or failing is logged at debug and yields no
    removal. Every subprocess is bounded by a timeout.
    """
    task_label = _safe_label_value(task_id)
    run_label = _safe_run_id(run_id)
    if not task_label or not run_label:
        return 0
    docker = docker_exe or _resolve_docker_exe()
    if not docker:
        return 0
    invoke = query or (lambda argv, timeout: run_capture(argv, timeout=timeout))

    def _call(argv: list[str], timeout: float):
        try:
            return invoke(argv, timeout)
        except Exception as exc:  # docker missing, daemon down, timeout, ...
            logger.debug("kanban docker cleanup failed (%s): %s", argv[:2], exc)
            return None

    # Repeated --filter label= arguments are ANDed by both docker and podman.
    listing = _call([
        docker, "ps", "-aq",
        "--filter", "label=hermes-agent=1",
        "--filter", f"label={_KANBAN_RUN_LABEL_KEY}={run_label}",
        "--filter", f"label={_KANBAN_TASK_LABEL_KEY}={task_label}",
        "--format", "{{.ID}}",
    ], _LIST_TIMEOUT_SECONDS)
    if listing is None or getattr(listing, "returncode", 1) != 0:
        if listing is not None:
            logger.debug("kanban docker cleanup ps failed: %s", getattr(listing, "stderr", ""))
        return 0

    removed = 0
    for line in str(getattr(listing, "stdout", "") or "").splitlines():
        cid = line.strip()
        if not cid:
            continue
        result = _call([docker, "rm", "-f", cid], _REMOVE_TIMEOUT_SECONDS)
        if result is not None and getattr(result, "returncode", 1) == 0:
            removed += 1
            logger.info("removed container %s for terminated kanban task %s run %s",
                        cid[:12], task_label, run_label)
        elif result is not None:
            logger.debug("kanban docker cleanup rm %s failed: %s", cid[:12], getattr(result, "stderr", ""))
    return removed


def _resolve_docker_exe() -> Optional[str]:
    """Same resolution as the Docker backend (including podman); never raises."""
    try:
        from tools.environments.docker import find_docker
        return find_docker()
    except Exception as exc:
        logger.debug("kanban docker cleanup: runtime lookup failed: %s", exc)
        return None
