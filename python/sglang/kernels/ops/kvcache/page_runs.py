# SPDX-License-Identifier: Apache-2.0
"""Direct HiCache copies by page run for layer-major host pools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_page_runs_module() -> Module:
    return load_jit(
        "page_runs",
        cuda_files=["kvcacheio/page_runs.cuh"],
        cuda_wrappers=[("run", "&PageRunCopy::run")],
    )


def warmup_page_runs() -> None:
    """Compile at pool construction so the first transfer does not pay the JIT build."""
    _jit_page_runs_module()


def transfer_page_runs(
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    host_ptrs: torch.Tensor,
    device_ptrs: torch.Tensor,
    *,
    host_rows: int,
    device_rows: int,
    rows_per_page: int,
    row_bytes: int,
    device: torch.device,
    to_host: bool,
) -> None:
    """Copy whole pages between per-layer device rows and per-layer host rows.

    ``host_indices`` / ``device_indices`` are CPU int64 row lists of whole pages
    (``rows_per_page`` consecutive rows each); ``host_ptrs`` / ``device_ptrs`` are CPU
    uint64 ``[layers]`` addresses of row 0 of each layer, which holds ``host_rows`` /
    ``device_rows`` rows of ``row_bytes``. Pages whose host and device rows both advance by
    one page form a run, copied with one descriptor per (run, layer); all descriptors go
    out in one ``cudaMemcpyBatchAsync`` on ``device``'s current torch stream. Raises on
    pages that are out of range or do not span consecutive rows, before any copy is issued.
    """
    if host_indices.numel() == 0:
        return
    # Every tensor argument is on the CPU, so the stream cannot come from them.
    stream = torch.cuda.current_stream(device).cuda_stream
    _jit_page_runs_module().run(
        host_indices,
        device_indices,
        host_ptrs,
        device_ptrs,
        host_rows,
        device_rows,
        rows_per_page,
        row_bytes,
        device.index,
        stream,
        to_host,
    )
