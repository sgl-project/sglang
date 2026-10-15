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
    pass

# Single-token fused sigmoid+bias top-k router (see
# csrc/moe/sm120_topk_gate_m1.cuh). The generic Triton router runs ~10us for
# one token over 512 experts; this single-block kernel does the same
# selection (biased scores, lowest-id tie-break, unbiased weights,
# renormalize, routed scaling) in ~2-3us on the decode hot path.


@cache_once
def _jit_topk_gate_m1(n, topk, renorm, apply_scale):
    args = make_cpp_args(n, topk, renorm, apply_scale, is_arch_support_pdl())
    return load_jit(
        "sm120_topk_gate_m1",
        *args,
        cuda_files=["moe/sm120_topk_gate_m1.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120TopkGateM1Kernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_topk_gate_m1(num_tokens, num_experts, topk):
    return num_tokens == 1 and 32 <= num_experts <= 1024 and topk <= num_experts


def topk_gate_m1(
    scores: torch.Tensor,  # [1, N] fp32 router logits
    bias: torch.Tensor,  # [N] fp32
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
):
    n = scores.shape[-1]
    module = _jit_topk_gate_m1(
        n, topk, bool(renormalize), bool(apply_routed_scaling_factor_on_output)
    )
    out_w = torch.empty(1, topk, dtype=torch.float32, device=scores.device)
    out_i = torch.empty(1, topk, dtype=torch.int32, device=scores.device)
    module.run(
        scores.view(-1),
        bias.view(-1),
        out_w.view(-1),
        out_i.view(-1),
        float(routed_scaling_factor if routed_scaling_factor is not None else 1.0),
    )
    return out_w, out_i
