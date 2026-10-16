"""Exact sorted sigmoid top22 for the small-M SM120 Nemotron router."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


@triton.jit
def _sm120_router_top22_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    stride_sm,
    stride_sn,
    scale,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.arange(0, 512)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    scores = tl.load(scores_ptr + row * stride_sm + n * stride_sn).to(tl.float32)
    bias = tl.load(bias_ptr + n).to(tl.float32)
    activated = tl.sigmoid(scores)
    biased = activated + bias
    biased = tl.where(biased == biased, biased, -1e30)
    biased = tl.where(biased == 0.0, 0.0, biased)
    bits = biased.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - n.to(tl.uint32)).to(tl.uint64)
    winners = tl.topk(keys, 32)
    ids = (0xFFFFFFFF - (winners & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    k = tl.arange(0, 32)
    values = tl.gather(activated, ids, axis=0)
    values = tl.where(k < 22, values, 0.0)
    total = tl.sum(values, 0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if RENORMALIZE:
        values = values / tl.where(total > 0.0, total, 1.0)
    if APPLY_SCALE:
        values = values * scale
    tl.store(weights_ptr + row * 22 + k, values, k < 22)
    tl.store(ids_ptr + row * 22 + k, ids, k < 22)


def topk_gate_small_m(
    scores,
    bias,
    topk=22,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
):
    assert scores.is_cuda and bias.device == scores.device
    assert scores.ndim == 2 and scores.shape[1] == 512
    assert scores.dtype == torch.float32
    assert 1 <= scores.shape[0] <= 16
    assert bias.shape == (512,) and bias.dtype == torch.float32 and bias.is_contiguous()
    assert topk == 22 and scoring_func == "sigmoid" and num_fused_shared_experts == 0
    m = scores.shape[0]
    weights = torch.empty((m, 22), device=scores.device, dtype=torch.float32)
    ids = torch.empty((m, 22), device=scores.device, dtype=torch.int32)
    pdl = is_arch_support_pdl()
    _sm120_router_top22_kernel[(m,)](
        scores,
        bias,
        weights,
        ids,
        scores.stride(0),
        scores.stride(1),
        float(routed_scaling_factor),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        USE_PDL=pdl,
        num_warps=1,
        **({"launch_pdl": True} if pdl else {}),
    )
    return weights, ids


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
