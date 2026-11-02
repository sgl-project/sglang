from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_finalize_module(
    weight_dtype: torch.dtype,
    top_k: int,
    hidden: int,
    has_shared: bool,
    use_pdl: bool,
) -> Module:
    # Widest bf16 vector dividing the row: the producers (trtllm-gen and our MoE kernels) keep
    # every row aligned to it, and the launcher checks the alignment.
    vec = next(v for v in (8, 4, 2) if hidden % v == 0)
    args = make_cpp_args(weight_dtype, top_k, vec, has_shared, use_pdl)
    return load_jit(
        "moe_finalize_fuse_shared",
        *args,
        cuda_files=["moe/moe_finalize_fuse_shared.cuh"],
        cuda_wrappers=[("run", f"MoeFinalizeKernel<{args}>::run")],
    )


def moe_finalize_fuse_shared(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    top_k: int,
    enable_pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    routing_ready: bool = True,
) -> torch.Tensor:
    """out[t] = sum_k expert_weights[t, k] * gemm2_out[permuted_idx[t * top_k + k]]
    (+ shared_output[t]), accumulated in fp32; rows with index -1 are skipped.

    gemm2_out: [num_permuted_rows, padded_hidden] bf16 (trtllm deferred output);
    expanded_idx_to_permuted_idx: [num_tokens * top_k] int32;
    expert_weights: [num_tokens, top_k] fp32 or bf16;
    shared_output: optional [num_tokens, hidden] bf16, hidden <= padded_hidden
    (without it the output keeps the padded width);
    out: optional [num_tokens, hidden] bf16 destination with unit column stride
    (e.g. a symmetric-memory buffer), allocated when None;
    routing_ready: with enable_pdl, the routing tensors and shared_output are
    read before the PDL wait. Valid when they were complete at this kernel's
    launch, i.e. the kernel producing gemm2_out triggers its dependents only
    after its own griddepcontrol.wait, as the trtllm-gen gemm2 does (its routing
    runs in an earlier kernel). Pass False for a producer that triggers before
    it has waited for the routing.
    """
    num_tokens = expert_weights.shape[0]
    hidden_dim = (
        shared_output.shape[1] if shared_output is not None else gemm2_out.shape[1]
    )
    if out is None:
        out = torch.empty(
            num_tokens, hidden_dim, dtype=torch.bfloat16, device=gemm2_out.device
        )
    module = _jit_finalize_module(
        expert_weights.dtype,
        top_k,
        hidden_dim,
        shared_output is not None,
        enable_pdl,
    )
    module.run(
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        routing_ready,
    )
    return out
