"""Self-heal for leaked POSIX shared-memory segments in CI.

SGLang processes are torn down with SIGKILL (kill_process_tree, PDEATHSIG),
which skips every Python-level unlink path, so /dev/shm segments accumulate
until the tmpfs is full and the next scheduler init dies with SIGBUS.

Pid-stamped names (see _creator_pid) are unlinked once their creator is dead;
pid-less families (_ORPHAN_PREFIXES) are unlinked unconditionally, safe only
because the sweep runs at CI job start right after killall.py. CI-only
(SGLANG_IS_IN_CI): both rules assume a single-tenant runner container.
"""

import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_SHM_DIR = Path("/dev/shm")
_SGL_SHM_PREFIX = "sgl_shm"

_ORPHAN_PREFIXES = (
    "sglang_loads_",  # managers/load_snapshot.py slot files
    "cuda.shm.",  # CUDA IPC segments
    "nccl-",  # NCCL communicator segments
    "sem.loky-",  # loky/joblib semaphores
)


def make_shm_name(kind: str) -> str:
    """Pid-stamped name (sgl_shm_<kind>_<pid>_<rand>) the sweep can reclaim."""
    return f"{_SGL_SHM_PREFIX}_{kind}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _creator_pid(filename: str) -> int | None:
    pid = None
    if filename.startswith(f"{_SGL_SHM_PREFIX}_"):
        # sgl_shm_<kind>_<pid>_<rand>
        parts = filename.split("_")
        if len(parts) >= 4:
            try:
                pid = int(parts[-2])
            except ValueError:
                return None
    elif filename.startswith("multi_tokenizer_args_"):
        try:
            pid = int(filename.rsplit("_", 1)[-1])
        except ValueError:
            return None
    # os.kill(0, ...) / os.kill(-1, ...) probe process groups, not a process.
    if pid is not None and pid <= 0:
        return None
    return pid


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else.
        return True


def cleanup_stale_shm() -> None:
    """Unlink leaked shared-memory segments (rules in module docstring).

    Best-effort: never raises, since a failed sweep must not block server
    startup.
    """
    try:
        _cleanup_stale_shm_impl()
    except Exception:
        logger.warning(
            "cleanup_stale_shm: sweep failed, continuing startup", exc_info=True
        )


def _is_in_ci() -> bool:
    # Same semantics as sglang.utils.is_in_ci, read directly so the module
    # stays import-free (CI runs it by path before sglang is installed).
    return os.environ.get("SGLANG_IS_IN_CI", "false").lower() in ("true", "1")


def _cleanup_stale_shm_impl() -> None:
    if not _is_in_ci():
        return
    if not _SHM_DIR.is_dir():
        return

    removed = 0
    freed_bytes = 0
    try:
        entries = list(_SHM_DIR.iterdir())
    except OSError as e:
        logger.warning("cleanup_stale_shm: cannot list %s, skipping: %s", _SHM_DIR, e)
        return
    for entry in entries:
        pid = _creator_pid(entry.name)
        if pid is not None:
            # A recycled pid reads as alive, so pid-reuse degrades to
            # under-collection (segment leaks), never to deleting a live
            # segment. Keep that bias when changing this check.
            if pid == os.getpid() or _pid_alive(pid):
                continue
        elif not entry.name.startswith(_ORPHAN_PREFIXES):
            continue
        try:
            size = entry.stat().st_size
            entry.unlink()
            removed += 1
            freed_bytes += size
        except FileNotFoundError:
            pass  # raced with another cleaner
        except OSError as e:
            logger.warning("cleanup_stale_shm: failed to remove %s: %s", entry.name, e)
    if removed:
        logger.info(
            "cleanup_stale_shm: removed %d stale segment(s), freed %.1f MiB",
            removed,
            freed_bytes / (1 << 20),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_stale_shm()
