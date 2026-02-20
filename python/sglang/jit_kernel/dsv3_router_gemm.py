from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dsv3_router_gemm_module(out_dtype: torch.dtype) -> Module:
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported output dtype for dsv3_router_gemm: {out_dtype}")

    # Keep PDL gating aligned with AOT behavior.
    enable_pdl = is_arch_support_pdl() and os.getenv("TRTLLM_ENABLE_PDL") == "1"
    args = make_cpp_args(enable_pdl, out_dtype)

    return load_jit(
        "dsv3_router_gemm",
        *args,
        cuda_files=[
            "gemm/dsv3_router_gemm_entry.cuh",
            "gemm/dsv3_router_gemm_bf16_out.cuh",
            "gemm/dsv3_router_gemm_float_out.cuh",
        ],
        cuda_wrappers=[("dsv3_router_gemm", f"dsv3_router_gemm_kernel<{args}>::run")],
    )


@register_custom_op(
    op_name="jit_dsv3_router_gemm",
    mutates_args=["output"],
)
def jit_dsv3_router_gemm(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
) -> None:
    module = _jit_dsv3_router_gemm_module(output.dtype)
    module.dsv3_router_gemm(output, mat_a, mat_b)


def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty(
            (hidden_states.shape[0], router_weights.shape[0]),
            device=hidden_states.device,
            dtype=out_dtype,
        )
    jit_dsv3_router_gemm(output, hidden_states, router_weights)
    return output
