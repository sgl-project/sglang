"""NPU-only pytest hook: evict stale model page cache between test cases.

Diffusion server tests in one pytest session load models sequentially. Each
finished case leaves its mmap'ed checkpoint files in the kernel page cache,
and cgroup v2 ``memory.current`` charges that cache to the container even
though the pages are clean and reclaimable. The layerwise-offload planner
reads ``memory.current`` when computing the host pin budget, so by the third
or fourth case the budget reads 0 GiB available and every component falls
back to slow paths (pageable host copies, checkpoint re-streaming) --
measured at 2-9x the baseline stage latency.

``POSIX_FADV_DONTNEED`` asks the kernel to drop the clean page-cache pages
of one file. It needs no privileges, unlike ``/proc/sys/vm/drop_caches``
(which requires a privileged container and acts host-wide), so we evict
every OTHER model's weight files right before a case starts its server.
The current case's own model stays cached: its pages are re-faulted during
load anyway and keeping them avoids a pointless disk re-read.
"""

from __future__ import annotations

import os

import pytest

# Only bother with files big enough to move the memory.current needle.
_MIN_EVICT_BYTES = 128 * 1024 * 1024


def _cgroup_usage_bytes() -> int | None:
    """Current cgroup memory usage, in the planner's own accounting terms.

    Reuses the same helper the layerwise-offload planner reads, so the
    before/after numbers in the log match what the host pin budget sees.
    """
    try:
        from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
            cgroup_memory_limit_bytes,
        )

        capped = cgroup_memory_limit_bytes()
        if capped is not None:
            return capped[1]
    except Exception:
        pass
    return None


def _evict_stale_page_cache(cache_root: str, keep_dir: str, case_id: str) -> None:
    keep = os.path.realpath(keep_dir)
    usage_before = _cgroup_usage_bytes()
    evicted: list[tuple[str, int]] = []
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                if os.path.realpath(path).startswith(keep + os.sep):
                    continue
                size = os.path.getsize(path)
                if size < _MIN_EVICT_BYTES:
                    continue
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                evicted.append((os.path.relpath(path, cache_root), size))
            except OSError:
                continue
    total_bytes = sum(size for _, size in evicted)
    prefix = f"[page-cache-evict] [{case_id}]"
    if not evicted:
        print(f"{prefix} nothing to evict under {cache_root} (keep: {keep})", flush=True)
        return
    for relpath, size in evicted:
        print(f"{prefix} evicted {relpath} ({size / 1024**3:.2f} GiB)", flush=True)
    usage_after = _cgroup_usage_bytes()
    tail = (
        f", cgroup memory.current: {usage_before / 1024**3:.1f} GiB -> "
        f"{usage_after / 1024**3:.1f} GiB"
        if usage_before is not None and usage_after is not None
        else ""
    )
    print(
        f"{prefix} dropped page cache of {len(evicted)} weight file(s), "
        f"~{total_bytes / 1024**3:.1f} GiB outside {keep}{tail}",
        flush=True,
    )


@pytest.fixture(autouse=True)
def _evict_previous_models_page_cache(request):
    """Evict other models' page cache before the per-case server starts.

    Autouse function-scoped fixtures run before the non-autouse
    ``diffusion_server`` fixture that launches the server process, so the
    cgroup memory accounting is clean by the time the layerwise-offload
    planner computes its host pin budget.
    """
    if not hasattr(os, "posix_fadvise"):
        # Not Linux (e.g. a developer machine); nothing to do.
        return
    callspec = getattr(request.node, "callspec", None)
    case = callspec.params.get("case") if callspec else None
    model_path = getattr(getattr(case, "server_args", None), "model_path", None)
    if not model_path or not os.path.isdir(model_path):
        return
    real = os.path.realpath(model_path)
    # Typical layout: <cache>/models/<org>/<repo> -> walk <cache>/models.
    # Fall back to the parent directory if the layout is unexpected; evicting
    # clean page cache is always safe, only the sweep breadth changes.
    root = os.path.dirname(os.path.dirname(real))
    if os.path.basename(root) != "models":
        root = os.path.dirname(real)
    case_id = getattr(case, "id", None) or request.node.name
    _evict_stale_page_cache(root, real, case_id)
