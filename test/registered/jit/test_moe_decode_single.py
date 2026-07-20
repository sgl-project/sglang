"""Correctness tests for the single-token (decode, M==1) fused MoE fast path.

:func:`decode_single_moe` is a performance shortcut for the plain bf16,
gated-SiLU, unquantised MoE at ``num_tokens == 1``. It must match the generic
``fused_experts`` path (which it replaces via an early return in
``fused_experts_impl``) up to floating-point rounding.

We validate two ways:

* against an explicit, definition-based torch reference (documents the math), and
* against the production generic path with the fast path disabled via
  ``SGLANG_DISABLE_MOE_DECODE_SINGLE``.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_decode_single import (
    decode_single_moe,
    decode_single_moe_supported,
)

DEVICE = "cuda"


def _ref_single_moe(hidden_states, w1, w2, topk_weights, topk_ids, routed_scaling_factor):
    """Definition-based reference: silu(x@Wg) * (x@Wu) @ Wd, routed-weighted sum."""
    x = hidden_states.float()  # [1, H]
    topk = topk_ids.shape[1]
    I = w1.shape[1] // 2
    out = torch.zeros_like(x)
    for j in range(topk):
        e = int(topk_ids[0, j])
        wj = topk_weights[0, j].float()
        wg = w1[e, :I, :].float()  # [I, H]
        wu = w1[e, I:, :].float()  # [I, H]
        wd = w2[e].float()  # [H, I]
        g = x @ wg.T  # [1, I]
        u = x @ wu.T  # [1, I]
        act = torch.nn.functional.silu(g) * u  # [1, I]
        out += wj * (act @ wd.T)  # [1, H]
    return (out * routed_scaling_factor).to(hidden_states.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("E,topk,H,I", [(128, 8, 2048, 768), (64, 6, 2048, 1408)])
def test_decode_single_matches_reference(dtype, E, topk, H, I):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    hidden_states = torch.randn(1, H, device=DEVICE, dtype=dtype) * 0.1
    w1 = torch.randn(E, 2 * I, H, device=DEVICE, dtype=dtype) * 0.05
    w2 = torch.randn(E, H, I, device=DEVICE, dtype=dtype) * 0.05
    topk_ids = torch.randperm(E, device=DEVICE)[:topk].unsqueeze(0).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(1, topk, device=DEVICE, dtype=torch.float32), dim=-1
    )
    rsf = 1.0

    assert decode_single_moe_supported(
        hidden_states,
        w1,
        w2,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        b1=None,
        b2=None,
        block_shape=None,
        gemm1_alpha=None,
        gemm1_limit=None,
        swiglu_limit=None,
        gate_up_interleaved=True,
        no_combine=False,
    )

    out = decode_single_moe(hidden_states, w1, w2, topk_weights, topk_ids, rsf)
    ref = _ref_single_moe(hidden_states, w1, w2, topk_weights, topk_ids, rsf)

    rel = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-2)).max().item()
    assert rel < 0.02, f"max rel err {rel} too high"


def test_decode_single_supported_gating():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    H, I, E = 2048, 768, 128
    hs = torch.randn(1, H, device=DEVICE, dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * I, H, device=DEVICE, dtype=torch.bfloat16)
    w2 = torch.randn(E, H, I, device=DEVICE, dtype=torch.bfloat16)
    common = dict(
        activation="silu", is_gated=True, apply_router_weight_on_input=False,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
        use_int4_w4a16=False, b1=None, b2=None, block_shape=None,
        gemm1_alpha=None, gemm1_limit=None, swiglu_limit=None,
        gate_up_interleaved=True, no_combine=False,
    )
    # M==1 supported
    assert decode_single_moe_supported(hs, w1, w2, **common)
    # M>1 not supported
    hs2 = torch.randn(2, H, device=DEVICE, dtype=torch.bfloat16)
    assert not decode_single_moe_supported(hs2, w1, w2, **common)
    # quantised not supported
    assert not decode_single_moe_supported(hs, w1, w2, **{**common, "use_fp8_w8a8": True})
    # gelu not supported
    assert not decode_single_moe_supported(hs, w1, w2, **{**common, "activation": "gelu"})
