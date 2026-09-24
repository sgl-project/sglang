from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@cache_once
def _jit_sampling_module(vec_size: int, deterministic: bool) -> Module:
    deterministic_cpp = "true" if deterministic else "false"
    return load_jit(
        "speculative_sampling",
        vec_size,
        deterministic,
        cuda_files=["speculative/sampling.cuh"],
        cuda_wrappers=[
            (
                "sample",
                f"tree_speculative_sampling_target_only<{vec_size}, {deterministic_cpp}>",
            ),
        ],
        extra_dependencies=["flashinfer"],
    )


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
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    vec_size = math.gcd(4, target_probs.shape[-1])
    module = _jit_sampling_module(vec_size, deterministic)
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
        draft_probs,
        threshold_single,
        threshold_acc,
    )
