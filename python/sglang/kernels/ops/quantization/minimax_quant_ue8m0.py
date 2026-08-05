from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


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
    input -- a single kernel replacing ``per_token_group_quant`` +
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
