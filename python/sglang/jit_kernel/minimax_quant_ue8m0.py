from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels._jit import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(group_size: int) -> Module:
    args = make_cpp_args(group_size, is_arch_support_pdl())
    return load_jit(
        "minimax_per_token_quant_ue8m0",
        *args,
        cuda_files=["minimax/per_token_quant_ue8m0.cuh"],
        cuda_wrappers=[
            ("per_token_quant_ue8m0", f"per_token_quant_ue8m0<{args}>"),
        ],
    )


@cache_once
def _jit_scatter_module(group_size: int, topk: int) -> Module:
    # topk is a template arg so the dst-row load/store loops fully unroll.
    args = make_cpp_args(group_size, topk, is_arch_support_pdl())
    return load_jit(
        "minimax_per_token_quant_ue8m0_scatter",
        *args,
        cuda_files=["minimax/per_token_quant_ue8m0.cuh"],
        cuda_wrappers=[
            (
                "per_token_quant_ue8m0_scatter",
                f"per_token_quant_ue8m0_scatter<{args}>",
            ),
        ],
    )


def per_token_quant_fp8_ue8m0(
    x: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token group quant to FP8-e4m3 with a fused UE8M0 (int32-packed) scale.

    Returns ``(x_q, x_sf)`` where ``x_q`` is fp8_e4m3 ``[num_tokens, hidden]`` and
    ``x_sf`` is the int32-packed UE8M0 scale ``[num_tokens, hidden//group_size//4]``
    (row-major). Byte-identical to ``per_token_group_quant_fp8(scale_ue8m0=True)``
    followed by ``transform_sf_into_required_layout`` (both ceil-round the scale),
    but does it in a single kernel -- no separate transpose/pack launch.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
    assert x.is_contiguous()
    num_tokens, hidden = x.shape
    assert hidden % group_size == 0
    num_groups = hidden // group_size
    assert num_groups % 4 == 0, "num_groups must be a multiple of 4 for int32 packing"

    x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty(
        (num_tokens, num_groups // 4), dtype=torch.int32, device=x.device
    )
    _jit_module(group_size).per_token_quant_ue8m0(x, x_q, x_sf)
    return x_q, x_sf


def per_token_quant_fp8_ue8m0_scatter(
    x: torch.Tensor,
    gateup_input: torch.Tensor,
    gateup_input_scale: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    m_max: int,
    group_size: int = 128,
) -> None:
    """Fused per-token FP8/UE8M0 quant **and** scatter into the permuted grouped-GEMM
    input -- a single kernel replacing ``per_token_quant_fp8_ue8m0`` +
    ``fill_gateup_input_triton_kernel``.

    For each source token it computes the fp8 row + int32-packed UE8M0 scale once,
    then writes them to each of the token's ``topk`` destination rows:
      ``gateup_input``       fp8  ``[E, m_max, hidden]``      (row ``src2dst[token, i]``)
      ``gateup_input_scale`` int32 ``[E, hidden//group//4, m_max]`` (MN-major; byte-scattered)
    Slots with ``topk_ids[token, i] < 0`` are skipped. Byte-identical to the
    two-kernel path on every written row.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2
    assert x.is_contiguous()
    assert gateup_input.dtype == torch.float8_e4m3fn and gateup_input.dim() == 3
    assert gateup_input_scale.dtype == torch.int32 and gateup_input_scale.dim() == 3
    num_tokens, hidden = x.shape
    assert hidden % group_size == 0
    num_groups = hidden // group_size
    assert num_groups % 4 == 0, "num_groups must be a multiple of 4 for int32 packing"
    _jit_scatter_module(group_size, int(topk)).per_token_quant_ue8m0_scatter(
        x, gateup_input, gateup_input_scale, src2dst, topk_ids, int(topk), int(m_max)
    )
