"""Minimal CUPTI activity collector (experimental), a tvm-ffi JIT module.

The C++ source is ``csrc/profiling/cupti_collector.cpp``. Unlike the header-only
JIT kernels it is a plain host library linked against a chosen CUPTI install:
the default is the ``nvidia/cu13`` CUPTI torch ships with; hardware event
tracing (``hes=True``, Blackwell) needs CUPTI >= 13.3, e.g. the
``nvidia-cuda-cupti`` 13.4 wheel in a separate directory. To make torch and the
collector share one libcupti, ``LD_PRELOAD`` that ``libcupti.so.13`` before the
process starts. Never run a kineto session in the same process, and keep CUPTI
attached once initialized: finalizing breaks later sessions.

Usage::

    collector = CuptiCollector(hes=True)  # before any CUDA context with hes
    collector.start()
    ...                                   # run / replay graphs
    collector.stop()
    rows = collector.take_records()       # [n, len(RECORD_FIELDS)] int64
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.jit.utils.compile.toolchain import cuda_home

# Column order of `take_records` rows; must match `Field` in the C++ source.
RECORD_FIELDS = (
    "start",
    "end",
    "graph_node_id",
    "graph_id",
    "correlation_id",
    "name_id",
    "stream_id",
    "kind",  # 0 kernel, 1 memcpy, 2 memset
    "grid_x",
    "grid_y",
    "grid_z",
)


def default_cupti_root() -> str:
    """The ``nvidia/cu13`` directory holding CUPTI's ``include/`` and ``lib/``."""
    for entry in sys.path:
        root = Path(entry) / "nvidia" / "cu13"
        if (root / "include" / "cupti_activity.h").is_file():
            return str(root)
    raise FileNotFoundError("no nvidia/cu13 CUPTI install found on sys.path")


def load_module(*, cupti_root: Optional[str] = None):
    root = Path(cupti_root or default_cupti_root())
    lib_dir = root / "lib"
    # The build cache key covers the include and link paths, so one module name
    # serves every CUPTI root.
    return load_jit(
        "cupti_collector",
        cpp_files=["profiling/cupti_collector.cpp"],
        # The CUPTI wheel ships no cuda.h; fall back to the toolkit's.
        extra_include_paths=[str(root / "include"), f"{cuda_home()}/include"],
        extra_ldflags=[
            f"-L{lib_dir}",
            "-l:libcupti.so.13",
            f"-Wl,-rpath,{lib_dir}",
        ],
        header_only=False,
    )


class CuptiCollector:
    """Handle on the process-wide collector; CUPTI state is global, so create one."""

    def __init__(
        self, *, hes: bool, track_graphs: bool = True, cupti_root: Optional[str] = None
    ):
        self._module = load_module(cupti_root=cupti_root)
        assert self._module.num_fields() == len(RECORD_FIELDS)
        self._module.init(hes, track_graphs)
        self.cupti_version = self._module.cupti_version()

    def start(self, *, with_memory: bool = True) -> None:
        self._module.start(with_memory)

    def stop(self) -> None:
        self._module.stop()

    def flush(self) -> None:
        self._module.flush()

    def take_records(self) -> np.ndarray:
        """Moves every buffered record out: [n, len(RECORD_FIELDS)] int64."""
        out = torch.empty(
            (self._module.num_records(), len(RECORD_FIELDS)), dtype=torch.int64
        )
        n = self._module.take_records(out)
        return out[:n].numpy()

    def names(self) -> list[str]:
        """Kernel names, indexed by the `name_id` column."""
        return [str(name) for name in self._module.names()]

    def graph_execs(self) -> list[tuple[int, int]]:
        """(exec id, handle of the user graph it came from), in creation order."""
        flat = list(self._module.graph_execs())
        return list(zip(flat[0::2], flat[1::2]))

    def dropped(self) -> int:
        return self._module.dropped()

    def callback_errors(self) -> int:
        return self._module.callback_errors()
