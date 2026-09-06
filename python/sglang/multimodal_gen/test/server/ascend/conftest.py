"""Ascend NPU conftest: evict stale model page cache before each test case.

Memory-capped CI runners (e.g. a 128 GiB cgroup on the 4-NPU A3 pool) count
reclaimable page cache from previously loaded models in cgroup
``memory.current``, so ``host_memory_available_bytes()`` reports ~0 GiB
available and layerwise offload degrades to the slow checkpoint-mapping path
(observed as multi-fold latency regressions on minimax/mova perf cases).
Dropping the page cache of non-current models before each case keeps the host
memory budget healthy for the upcoming model load.
"""

from __future__ import annotations

import os
import sys

import pytest

from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import (
    MODELSCOPE_MODEL_WEIGHTS_DIR,
)

_CGROUP_V2_CURRENT = "/sys/fs/cgroup/memory.current"
_CGROUP_V1_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"


def _read_cgroup_memory_current() -> str:
    """Best-effort read of the container's cgroup memory usage in bytes."""
    for path in (_CGROUP_V2_CURRENT, _CGROUP_V1_USAGE):
        try:
            with open(path, encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            continue
    return "unknown"


def _collect_case_weight_paths(server_args) -> set[str]:
    """Real paths of every weight dir/file the upcoming case will load."""
    paths = set()
    for attr in ("model_path", "transformer_weights_path", "lora_path"):
        value = getattr(server_args, attr, None)
        if value:
            paths.add(os.path.realpath(value))
    for attr in ("component_paths", "component_weights_paths"):
        mapping = getattr(server_args, attr, None)
        if mapping:
            paths.update(
                os.path.realpath(value)
                for value in mapping.values()
                if value
            )
    return paths


def _evict_dir_page_cache(root: str, keep: set[str]) -> tuple[int, int]:
    """Drop page cache of files under ``root`` except those under ``keep``.

    ``posix_fadvise(POSIX_FADV_DONTNEED)`` only drops clean cache pages, so it
    is safe on read-only model checkpoints.
    """
    evicted_files = 0
    evicted_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.realpath(os.path.join(dirpath, name))
            if any(path == kept or path.startswith(kept + os.sep) for kept in keep):
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    evicted_bytes += os.fstat(fd).st_size
                    evicted_files += 1
                finally:
                    os.close(fd)
            except OSError:
                continue
    return evicted_files, evicted_bytes


@pytest.fixture(autouse=True)
def _evict_stale_model_page_cache(request):
    """Evict page cache of models not used by the upcoming test case.

    For parametrized diffusion cases the current model's weights are kept so
    the server load is not penalized; only stale cache from previous cases is
    dropped. For standalone (non-parametrized) tests everything is dropped.
    """
    if sys.platform != "linux" or not os.path.isdir(MODELSCOPE_MODEL_WEIGHTS_DIR):
        yield
        return

    keep: set[str] = set()
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None:
        for value in callspec.params.values():
            server_args = getattr(value, "server_args", None)
            if server_args is not None:
                keep |= _collect_case_weight_paths(server_args)

    before = _read_cgroup_memory_current()
    print(
        f"[CONFTEST] Evicting stale model page cache before {request.node.nodeid} "
        f"(keep: {sorted(os.path.basename(p) for p in keep) or 'none'})"
    )
    print(f"[CONFTEST] cgroup memory.current before: {before}")

    total_files, total_bytes = _evict_dir_page_cache(
        MODELSCOPE_MODEL_WEIGHTS_DIR, keep
    )
    print(
        f"[CONFTEST] Page cache eviction summary: {total_files} files, "
        f"{total_bytes / 1024 ** 3:.2f} GiB evicted"
    )
    print(f"[CONFTEST] cgroup memory.current after: {_read_cgroup_memory_current()}")
    yield
