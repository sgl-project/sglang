"""The EAGLE verify's target-only tree sampling as multi-CTA JIT kernels, bitwise equal
to the stock tree_speculative_sampling_target_only."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Rows = requests x draft tokens (checked bitwise up to 2048); the scratch holds this many requests.
MAX_ROWS = 1024
# Below this the stock kernel's GPU time is within the three launches' extra host time.
_MIN_VOCAB = 128000
_TREE_THREADS = 1024
_MAX_CHUNKS = 512  # multi_cta_verify_sampling.cuh kMaxChunks

# Per-device scratch, allocated once and never freed: the post-verify CUDA graphs share one
# memory pool across batch sizes and slots, and a slot's outputs must outlive the other slot's
# replay, so the captured body allocates exactly what the stock tree sampling does.
_scratch: dict = {}


def _scratch_for(device: torch.device) -> dict:
    bufs = _scratch.get(device.index)
    if bufs is None:
        bufs = _scratch[device.index] = {
            "meta": torch.empty((MAX_ROWS, 3), dtype=torch.int64, device=device),
            "chunk_sums": torch.empty(
                (2 * MAX_ROWS * _MAX_CHUNKS,), dtype=torch.float32, device=device
            ),
            "chunk_max_valid": torch.empty(
                (MAX_ROWS * _MAX_CHUNKS,), dtype=torch.int32, device=device
            ),
        }
    return bufs


@cache_once
def _jit_module(vec_size: int, deterministic: bool) -> Module:
    deterministic_cpp = "true" if deterministic else "false"
    return load_jit(
        "multi_cta_verify_sampling",
        vec_size,
        deterministic,
        cuda_files=["speculative/multi_cta_verify_sampling.cuh"],
        cuda_wrappers=[
            ("tree", f"MultiCtaVerifyTree<{vec_size}, {deterministic_cpp}>::run"),
        ],
        extra_dependencies=["flashinfer"],
    )


def applies(target_probs: torch.Tensor) -> bool:
    """Whether the verify's target-only tree sampling of these [bs, draft, vocab] fp32 probabilities
    is on a shape the multi-CTA kernels were checked bitwise on (CUDA, not HIP)."""
    return (
        target_probs.is_cuda
        and torch.version.hip is None
        and target_probs.dtype == torch.float32
        and target_probs.dim() == 3
        and 0 < target_probs.shape[0] * target_probs.shape[1] <= MAX_ROWS
        and target_probs.shape[2] >= _MIN_VOCAB
    )


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """sglang.kernels.ops.speculative.sampling.tree_speculative_sampling_target_only, bitwise:
    the same arguments, outputs and draft_probs writes."""
    bs, _, vocab = target_probs.shape
    vec_size = math.gcd(4, vocab)
    nchunks = -(-vocab // (_TREE_THREADS * vec_size))
    if bs > MAX_ROWS or nchunks > _MAX_CHUNKS:
        raise ValueError(
            f"multi-CTA tree sampling: batch {bs} or {nchunks} chunks past the scratch"
        )
    bufs = _scratch_for(target_probs.device)
    meta = bufs["meta"][:bs]
    chunk_sums = bufs["chunk_sums"][: 2 * bs * nchunks].view(2, bs, nchunks)
    chunk_max_valid = bufs["chunk_max_valid"][: bs * nchunks].view(bs, nchunks)
    _jit_module(vec_size, deterministic).tree(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        meta,
        chunk_sums,
        chunk_max_valid,
        threshold_single,
        threshold_acc,
    )
