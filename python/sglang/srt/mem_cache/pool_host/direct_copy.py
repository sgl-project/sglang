"""Direct-backend copies for layer-major host pools, batched by page run."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence

import torch

from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()

# Set on the first failure of the batched op; every later transfer takes the fallback.
_op_failed = False


@functools.cache
def _op_available() -> bool:
    if not _is_cuda:
        return False
    try:
        from sglang.kernels.ops.kvcache.page_runs import warmup_page_runs

        warmup_page_runs()
        return True
    except Exception as e:
        logger.warning(
            "Batched page-run copy unavailable; direct-backend layer_first HiCache "
            "transfers will use transfer_kv_direct: %s",
            e,
        )
        return False


def warmup_transfer_kv_direct_runs() -> None:
    """Build the batched op before serving; a first-use JIT build stalls a transfer.

    A failed build only logs: every transfer then takes the fallback.
    """
    _op_available()


# Pointer tables by layer addresses; buffers live as long as their pools, so the set of
# keys stays small (one per layer plus one per all-layer list).
_PTR_TABLES: dict[tuple[int, ...], torch.Tensor] = {}


def _ptr_table(layers: Sequence[torch.Tensor]) -> torch.Tensor:
    key = tuple(t.data_ptr() for t in layers)
    table = _PTR_TABLES.get(key)
    if table is None:
        table = torch.tensor(key, dtype=torch.uint64)
        _PTR_TABLES[key] = table
    return table


def _cpu_int64(t: torch.Tensor) -> torch.Tensor:
    return t if not t.is_cuda and t.dtype == torch.int64 else t.to("cpu", torch.int64)


def transfer_kv_direct_runs(
    *,
    src_layers: Sequence[torch.Tensor],
    dst_layers: Sequence[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
    fallback: Callable[..., None],
) -> None:
    """`transfer_kv_direct` with the same arguments and bytes moved, batched by page run.

    Row `src_indices[i]` of `src_layers[l]` goes to row `dst_indices[i]` of
    `dst_layers[l]`. Pages (`page_size` consecutive indices each) that advance together on
    both sides form a run, copied with one descriptor per (run, layer) in one batched call
    on the current stream, instead of one torch copy per (run, layer). Layers of one side
    must share shape and strides; only the first is inspected. `fallback` (the platform's
    `transfer_kv_direct`) takes over off CUDA, for non-contiguous layers, when the two
    sides differ in dtype or row size or neither side is on the host, and from the first
    failure of the op on (indices that are not whole in-range pages, or a CUDA error),
    which it logs once.
    """
    to_host = src_layers[0].is_cuda
    device_layers, host_layers = (
        (src_layers, dst_layers) if to_host else (dst_layers, src_layers)
    )
    device0, host0 = device_layers[0], host_layers[0]
    row_bytes = device0.stride(0) * device0.element_size()
    if (
        not _op_failed
        and device0.is_cuda
        and not host0.is_cuda
        and device0.is_contiguous()
        and host0.is_contiguous()
        and host0.dtype == device0.dtype
        and host0.stride(0) * host0.element_size() == row_bytes
        and _op_available()
    ):
        device_indices, host_indices = (
            (src_indices, dst_indices) if to_host else (dst_indices, src_indices)
        )
        try:
            _transfer(
                host_indices=_cpu_int64(host_indices),
                device_indices=_cpu_int64(device_indices),
                host_layers=host_layers,
                device_layers=device_layers,
                page_size=page_size,
                row_bytes=row_bytes,
                to_host=to_host,
            )
            return
        except Exception as e:
            _record_failure(e)
    fallback(
        src_layers=src_layers,
        dst_layers=dst_layers,
        src_indices=src_indices,
        dst_indices=dst_indices,
        page_size=page_size,
    )


def _transfer(
    *,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    host_layers: Sequence[torch.Tensor],
    device_layers: Sequence[torch.Tensor],
    page_size: int,
    row_bytes: int,
    to_host: bool,
) -> None:
    from sglang.kernels.ops.kvcache.page_runs import transfer_page_runs

    transfer_page_runs(
        host_indices,
        device_indices,
        _ptr_table(host_layers),
        _ptr_table(device_layers),
        host_rows=host_layers[0].shape[0],
        device_rows=device_layers[0].shape[0],
        rows_per_page=page_size,
        row_bytes=row_bytes,
        device=device_layers[0].device,
        to_host=to_host,
    )


def _record_failure(e: Exception) -> None:
    global _op_failed
    logger.warning(
        "Batched page-run copy failed; direct-backend layer_first HiCache transfers "
        "use transfer_kv_direct from now on: %s",
        e,
    )
    _op_failed = True
