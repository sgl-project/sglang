"""DEPRECATED: superseded by ``sglang.kernels.ops.quantization._jit_per_token_group_quant`` (the
default CUDA path). No sglang runtime code may call this kernel; it is kept
only as the perf baseline for the per_token_group_quant benchmarks and its own
bit-parity tests, and will be deleted once those move to torch references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(in_dtype: torch.dtype, out_dtype: torch.dtype, use_pdl: bool) -> Module:
    args = make_cpp_args(in_dtype, out_dtype, use_pdl)
    return load_jit(
        "per_token_group_quant_8bit_v2",
        *args,
        cuda_files=["gemm/per_token_group_quant_8bit_v2.cuh"],
        cuda_wrappers=[
            (
                "per_token_group_quant_8bit_v2",
                f"PerTokenGroupQuant8bitV2Kernel<{args}>::run",
            )
        ],
        # Match the AOT sgl-kernel build (-use_fast_math) so the FP8 scale
        # division/rounding is bit-identical to sgl_per_token_group_quant_8bit_v2.
        extra_cuda_cflags=["--use_fast_math"],
    )


@register_custom_op(
    op_name="per_token_group_quant_8bit_v2",
    mutates_args=["output_q", "output_s"],
)
def _per_token_group_quant_8bit_v2_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    """Opaque custom-op boundary around the JIT v2 kernel.

    Registering this as a custom op (instead of calling the tvm-ffi module
    directly) keeps torch.compile / piecewise-CUDA-graph from tracing into the
    tvm-ffi ``Function.__call__`` (which Dynamo cannot trace). All shape-derived
    scalars are computed here and passed to the kernel.

    Layouts (matching the AOT v2):
      vanilla:                   input (num_tokens, hidden), output_q (num_tokens, hidden)
      fuse_silu_and_mul:         input (num_tokens, hidden*2), output_q (num_tokens, hidden)
      fuse_silu_and_mul+masked:  input (num_experts, tokens_pad, hidden*2),
                                 output_q (num_experts, tokens_pad, hidden), masked_m (num_experts,)
    """
    masked_layout = masked_m is not None
    numel = input.numel()
    num_groups = numel // group_size // (2 if fuse_silu_and_mul else 1)
    if num_groups == 0:  # empty input -> grid 0 -> cudaErrorInvalidConfiguration
        return
    num_local_experts = input.shape[0] if masked_layout else 1
    last = output_q.dim() - 1
    is_column_major = output_s.stride(last - 1) < output_s.stride(last)
    hidden_dim_num_groups = output_q.shape[last] // group_size
    num_tokens_per_expert = output_q.shape[last - 1]
    scale_expert_stride = output_s.stride(0) if masked_layout else 0
    scale_hidden_stride = output_s.stride(last)

    module = _jit_module(input.dtype, output_q.dtype, is_arch_support_pdl())
    module.per_token_group_quant_8bit_v2(
        input,
        output_q,
        output_s,
        masked_m if masked_layout else input,  # unused (nullptr) when not masked
        int(group_size),
        bool(scale_ue8m0),
        bool(fuse_silu_and_mul),
        bool(masked_layout),
        int(num_groups),
        int(num_local_experts),
        bool(is_column_major),
        int(hidden_dim_num_groups),
        int(num_tokens_per_expert),
        int(scale_expert_stride),
        int(scale_hidden_stride),
    )


@debug_kernel_api
def per_token_group_quant_8bit_v2(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    """JIT port of sgl_per_token_group_quant_8bit_v2 (full feature parity).

    Wraps the registered custom op so torch.compile / piecewise CUDA graph treat
    the tvm-ffi kernel call as an opaque boundary.
    """
    _per_token_group_quant_8bit_v2_custom_op(
        input=input,
        output_q=output_q,
        output_s=output_s,
        group_size=group_size,
        eps=eps,
        min_8bit=min_8bit,
        max_8bit=max_8bit,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )
