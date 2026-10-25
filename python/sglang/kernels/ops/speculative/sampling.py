from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@cache_once
def _jit_sampling_module(
    vec_size: int, deterministic: bool, no_draft_probs: bool = False
) -> Module:
    deterministic_cpp = "true" if deterministic else "false"
    no_draft_probs_cpp = "true" if no_draft_probs else "false"
    return load_jit(
        "speculative_sampling",
        vec_size,
        deterministic,
        *(("no_draft_probs",) if no_draft_probs else ()),
        cuda_files=["speculative/sampling.cuh"],
        cuda_wrappers=[
            (
                "sample",
                f"tree_speculative_sampling_target_only<{vec_size}, "
                f"{deterministic_cpp}, {no_draft_probs_cpp}>",
            ),
        ],
        extra_dependencies=["flashinfer"],
    )


@cache_once
def _jit_chunked_sampling_module(vec_size: int, deterministic: bool) -> Module:
    deterministic_cpp = "true" if deterministic else "false"
    return load_jit(
        "speculative_sampling_chunked",
        vec_size,
        deterministic,
        cuda_files=["speculative/sampling.cuh"],
        cuda_wrappers=[
            (
                "sample",
                f"chain_speculative_sampling_target_only_two_pass<{vec_size}, "
                f"{deterministic_cpp}>",
            ),
        ],
        extra_dependencies=["flashinfer"],
    )


# Vocabulary chunk of the sampling kernels: 1024 threads x vec_size elements.
_CHUNK_THREADS = 1024


@debug_kernel_api
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
    draft_probs: Optional[torch.Tensor],
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
    chunked: bool = False,
) -> None:
    """``draft_probs=None`` (chains only: one candidate per tree level) samples
    as a zero-filled ``draft_probs`` would, without the buffer.

    ``chunked`` (with ``draft_probs=None`` only) runs the same sampling as two
    launches, the vocabulary passes split over one CTA per chunk; the outputs
    are the same bits."""
    vec_size = math.gcd(4, target_probs.shape[-1])
    no_draft_probs = draft_probs is None
    if chunked and no_draft_probs:
        bs = target_probs.shape[0]
        chunk = _CHUNK_THREADS * vec_size
        num_chunks = (target_probs.shape[-1] + chunk - 1) // chunk
        chunk_sums = target_probs.new_empty((2, bs, num_chunks))
        chunk_last_valid = accept_token_num.new_empty((bs, num_chunks))
        _jit_chunked_sampling_module(vec_size, deterministic).sample(
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
            chunk_sums[0],
            chunk_sums[1],
            chunk_last_valid,
            threshold_single,
            threshold_acc,
        )
        return
    module = _jit_sampling_module(vec_size, deterministic, no_draft_probs)
    module.sample(
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
        target_probs if no_draft_probs else draft_probs,
        threshold_single,
        threshold_acc,
    )
