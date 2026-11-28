"""One target sample per EAGLE chain verify row from top-p(softmax(logits / T)).

A single cluster kernel (``speculative/verify_topp_chain.cuh``) replaces the
softmax, top-p renorm and tree-sampling kernels of the sampled verify; its
reductions have a fixed order and its coins come from Philox keyed by the
TP-synchronized seed and a device step counter, so every rank draws the same
tokens from the same gathered logits, eagerly and under CUDA-graph replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# kCluster of the kernel; tokens per row = CLUSTER * threads * items.
CLUSTER = 8
THREADS = 1024
MAX_ITEMS = 32
MAX_VOCAB = CLUSTER * THREADS * MAX_ITEMS
MAX_ROWS = 65535
# StateSlot of the kernel: step, arrive, rows with > 1 round, rows at the round cap.
STATE_SLOTS = 4

# Per device: the persistent coin-stream state. Never replaced: captured graphs hold it.
_STATES: Dict[int, torch.Tensor] = {}


@cache_once
def _module(threads: int, items: int, argmax: bool, use_pdl: bool) -> Module:
    args = make_cpp_args(threads, items, argmax, use_pdl)
    return load_jit(
        "verify_topp_chain",
        *args,
        cuda_files=["speculative/verify_topp_chain.cuh"],
        cuda_wrappers=[("run", f"verify_topp_chain_sample<{args}>")],
    )


def items_for(vocab: int, threads: int = THREADS) -> int:
    return -(-vocab // (CLUSTER * threads))


def state_for(device: torch.device) -> Optional[torch.Tensor]:
    """The device's persistent state, created on first use outside graph capture."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    state = _STATES.get(index)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        state = torch.zeros((STATE_SLOTS,), dtype=torch.int64, device=device)
        _STATES[index] = state
    return state


def verify_topp_chain_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: Optional[torch.Tensor],
    num_draft: int,
    seed: int,
    *,
    want_argmax: bool,
    state: torch.Tensor,
    threads: int = THREADS,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample every verify row of ``logits`` [bs * num_draft, vocab] fp32.

    temperatures [bs, 1] fp32; top_ps [bs] fp32 or None (top_p = 1).
    Returns tokens [rows] int32 and, with want_argmax, the raw-logit argmax [rows].
    """
    rows, vocab = logits.shape
    items = items_for(vocab, threads)
    module = _module(threads, items, want_argmax, is_arch_support_pdl())
    tokens = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    argmax = (
        torch.empty((rows,), dtype=torch.int32, device=logits.device)
        if want_argmax
        else None
    )
    module.run(
        logits,
        temperatures,
        top_ps if top_ps is not None else temperatures,
        tokens,
        argmax if argmax is not None else tokens,
        state,
        num_draft,
        int(top_ps is not None),
        seed,
    )
    return tokens, argmax
