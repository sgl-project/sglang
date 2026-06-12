"""Self-heal for leaked POSIX shared-memory segments in CI.

SGLang processes are torn down with SIGKILL (kill_process_tree, PDEATHSIG),
which skips every Python-level unlink path, so /dev/shm segments accumulate
until the tmpfs is full and the next scheduler init dies with SIGBUS.

Segments created through make_shm_name() embed the creator pid, which lets a
later server startup safely unlink segments whose creator is gone. The sweep
only runs in CI (single-tenant runner containers); on shared dev machines a
pid check against another user's process is not authoritative, so we skip.
"""

import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_SHM_DIR = Path("/dev/shm")
_SGL_SHM_PREFIX = "sgl_shm"


def make_shm_name(kind: str) -> str:
    """Name a shared-memory segment so cleanup_stale_shm can identify and
    reclaim it after its creator process dies: sgl_shm_<kind>_<pid>_<rand>."""
    return f"{_SGL_SHM_PREFIX}_{kind}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _creator_pid(filename: str) -> int | None:
    if filename.startswith(f"{_SGL_SHM_PREFIX}_"):
        # sgl_shm_<kind>_<pid>_<rand>
        parts = filename.split("_")
        if len(parts) >= 4:
            try:
                return int(parts[-2])
            except ValueError:
                return None
    if filename.startswith("multi_tokenizer_args_"):
        try:
            return int(filename.rsplit("_", 1)[-1])
        except ValueError:
            return None
    return None


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
    """Unlink shared-memory segments whose creator process is dead.

    CI-only: gated on SGLANG_IS_IN_CI because the pid-liveness check is only
    trustworthy when the container runs one job at a time.
    """
    from sglang.utils import is_in_ci

    if not is_in_ci():
        return
    if not _SHM_DIR.is_dir():
        return

    removed = 0
    freed_bytes = 0
    try:
        entries = list(_SHM_DIR.iterdir())
    except OSError:
        return
    for entry in entries:
        pid = _creator_pid(entry.name)
        if pid is None or pid == os.getpid() or _pid_alive(pid):
            continue
        try:
            size = entry.stat().st_size
            entry.unlink()
            removed += 1
            freed_bytes += size
        except OSError:
            pass
    if removed:
        logger.info(
            "cleanup_stale_shm: removed %d stale segment(s), freed %.1f MiB",
            removed,
            freed_bytes / (1 << 20),
        )
