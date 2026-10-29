"""Classic chain rejection sampling with each request's vocabulary over many blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CHUNK = 8192  # vocabulary entries a block (chain_rejection::kChunk)

# a device's zeroed barrier words (2 a request), which the launch leaves zeroed:
# allocated outside graph capture (a capture records no zeroing it could rely on)
_work: dict = {}


@cache_once
def _jit_chain_rejection_module() -> Module:
    return load_jit(
        "chain_rejection",
        cuda_files=["speculative/chain_rejection.cuh"],
        cuda_wrappers=[("chain_rejection", "ChainRejectionKernel::run")],
    )


@cache_once
def _cu_count(dev_index: int) -> int:
    return torch.cuda.get_device_properties(dev_index).multi_processor_count


def _work_for(bs: int, dev: torch.device) -> Optional[torch.Tensor]:
    buf = _work.get(dev)
    if buf is not None and buf.numel() >= 2 * bs:
        return buf
    if torch.cuda.is_current_stream_capturing():
        return None
    buf = torch.zeros(2 * bs, dtype=torch.int64, device=dev)
    _work[dev] = buf
    return buf


def _rows_aligned(t: torch.Tensor) -> bool:
    return (
        t.stride(-1) == 1
        and t.stride(0) % 4 == 0
        and t.stride(1) % 4 == 0
        and t.data_ptr() % 16 == 0
    )


def chain_rejection_supported(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> bool:
    """Whether every block of the launch is resident at once (its barrier waits on
    the others), its rows load 16 B at a time, and its barrier words exist (made
    here outside graph capture)."""
    bs, _, vocab = target_probs.shape
    dev = target_probs.device
    chunks = -(-vocab // _CHUNK)
    return (
        chunks * bs <= _cu_count(dev.index or 0)
        and vocab % 4 == 0
        and _rows_aligned(target_probs)
        and _rows_aligned(draft_probs)
        and _work_for(bs, dev) is not None
    )


def chain_rejection_sampling(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> None:
    """``chain_speculative_sampling_triton``'s classic chain path (no block verification)."""
    bs, _, vocab = target_probs.shape
    dev = target_probs.device
    work = _work_for(bs, dev)
    assert work is not None, "chain_rejection_supported first"
    chunk_mass = torch.empty(bs * -(-vocab // _CHUNK), dtype=torch.float32, device=dev)
    _jit_chain_rejection_module().chain_rejection(
        predicts,
        accept_index,
        accept_token_num,
        candidates.to(torch.int64),
        retrive_index.to(torch.int64),
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        chunk_mass,
        work,
    )
