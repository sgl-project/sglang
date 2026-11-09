"""Qwen3.8 top-10 of 512 RenormalizeNaive routing at any row count (ids int32, weights bf16).

The per-row half of q38_fused_routing (kernels/jit/csrc/moe/q38_prefill_routing.cuh): same selection order and
weight expressions as FlashInfer 0.7.2's trtllm-gen routing, built with the same fast-math flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_EXPERTS = 512
TOP_K = 10


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "q38_prefill_routing",
        cuda_files=["moe/q38_prefill_routing.cuh"],
        cuda_wrappers=[("run", "Q38PrefillRoutingKernel::run")],
        # FlashInfer builds its routing kernels with fast math; the softmax must lower identically.
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def covered(router_logits: torch.Tensor) -> bool:
    return (
        router_logits.dim() == 2
        and router_logits.shape[0] >= 1
        and router_logits.shape[1] == NUM_EXPERTS
        and router_logits.dtype == torch.bfloat16
        and router_logits.is_cuda
        and router_logits.stride(1) == 1
        and router_logits.stride(0) % 8 == 0
        and router_logits.data_ptr() % 16 == 0
    )


def route(router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, 512] bf16 logits -> ([M, 10] int32 expert ids, [M, 10] bf16 weights), column k the k-th largest."""
    m = router_logits.shape[0]
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device=router_logits.device)
    weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=router_logits.device)
    _jit_module().run(router_logits, ids, weights)
    return ids, weights
