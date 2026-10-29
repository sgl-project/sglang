"""Temperature softmax, optionally top-p renormalized, with each row over many blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CHUNK = 4096  # vocabulary entries a block (softmax_top_p::kChunk)
_ROW_WORDS = 4096 + 2048 + 256 + 9  # a row's histograms, state and counters (kRowWords)
_FUSED_ROW_WORDS = _ROW_WORDS + 6  # plus the one launch's barriers (kFusedRowWords)

# a device's zeroed words for the one launch, which leaves them zeroed: allocated
# outside graph capture (a capture records no zeroing it could rely on)
_fused_work: dict = {}


def _fused_work_for(rows: int, dev: torch.device) -> Optional[torch.Tensor]:
    need = rows * _FUSED_ROW_WORDS
    buf = _fused_work.get(dev)
    if buf is not None and buf.numel() >= need:
        return buf
    if torch.cuda.is_current_stream_capturing():
        return None
    buf = torch.zeros(need, dtype=torch.int64, device=dev)
    _fused_work[dev] = buf
    return buf


@cache_once
def _cu_count(dev_index: int) -> int:
    return torch.cuda.get_device_properties(dev_index).multi_processor_count


@cache_once
def _jit_softmax_top_p_module() -> Module:
    return load_jit(
        "softmax_top_p",
        cuda_files=["sampling/softmax_top_p.cuh"],
        cuda_wrappers=[("softmax_top_p", "SoftmaxTopPKernel::run")],
    )


def softmax_top_p_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: Optional[torch.Tensor] = None,
    rows_per_request: int = 1,
) -> torch.Tensor:
    """``softmax(logits / T)`` for each row, then with ``top_ps`` FlashInfer's top-p
    threshold (ties kept) and the kept mass renormalized.

    ``logits`` [rows, vocab] fp32 (rows may be strided); ``temperatures`` and
    ``top_ps`` one per request, a request ``rows_per_request`` consecutive rows."""
    if logits.dtype != torch.float32:
        logits = logits.float()
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    rows, vocab = logits.shape
    chunks = -(-vocab // _CHUNK)
    dev = logits.device
    out = torch.empty((rows, vocab), dtype=torch.float32, device=dev)
    stats = torch.empty(rows * chunks * 2, dtype=torch.float32, device=dev)
    kept = torch.empty(rows * chunks, dtype=torch.float32, device=dev)
    temps = temperatures.reshape(-1).to(dtype=torch.float32).contiguous()
    top = (
        torch.empty(0, dtype=torch.float32, device=dev)
        if top_ps is None
        else top_ps.reshape(-1).to(dtype=torch.float32).contiguous()
    )
    # one launch when every block is resident at once: its barriers wait on the others
    work = (
        _fused_work_for(rows, dev)
        if chunks * rows <= _cu_count(dev.index or 0)
        else None
    )
    fused = work is not None
    if not fused:
        work = torch.empty(
            rows * _ROW_WORDS if top_ps is not None else 0,
            dtype=torch.int64,
            device=dev,
        )
    _jit_softmax_top_p_module().softmax_top_p(
        logits, temps, top, out, stats, kept, work, rows_per_request, int(fused)
    )
    return out
