"""Unit test for the SM90 cutlass MXFP4 path in :class:`Mxfp4MoEMethod`.

Builds a single-layer GPT-OSS-style MoE with random MXFP4 weights, drives the
SGLang plumbing (``_process_weights_for_sm90_cutlass`` + ``_apply_sm90_cutlass``)
and compares against a direct FlashInfer ``cutlass_fused_moe`` call with the
same inputs. Both paths invoke the same SM90 kernel from FlashInfer PR #3084,
so outputs must be bit-exact.

Run on H100/H200:

    cd python && python -m pytest sglang/test/test_mxfp4_sm90_cutlass.py -v
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

flashinfer_fused_moe = pytest.importorskip("flashinfer.fused_moe")

if not hasattr(flashinfer_fused_moe, "interleave_moe_weights_for_sm90_mixed_gemm"):
    pytest.skip(
        "FlashInfer build does not include PR #3084 SM90 mixed-input helpers",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.srt.utils import is_sm90_supported, is_sm100_supported

if not is_sm90_supported() or is_sm100_supported():
    pytest.skip(
        "SM90-only path; require Hopper without SM100 promotion",
        allow_module_level=True,
    )

from flashinfer.fused_moe import (
    cutlass_fused_moe,
    interleave_moe_scales_for_sm90_mixed_gemm,
    interleave_moe_weights_for_sm90_mixed_gemm,
)
from flashinfer.fused_moe.core import ActivationType

GROUP_SIZE = 32  # MXFP4 block size


class _MockLayer:
    """Stand-in for ``FusedMoE`` carrying the attributes the SM90 helpers read.

    We construct one by hand so the test stays out of SGLang's distributed init
    path (``get_tp_group`` etc.).
    """


class _MockTopKOutput:
    def __init__(self, weights, ids):
        self.topk_weights = weights
        self.topk_ids = ids


def _make_random_mxfp4(num_experts, hidden, inter, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * inter, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden, inter // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    # E8M0 scales centered around 127 (= 2^0); narrow band keeps dequant values
    # in a sane range so SwiGLU clamp doesn't dominate.
    w13_s = torch.randint(
        125,
        130,
        (num_experts, 2 * inter, hidden // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2_s = torch.randint(
        125,
        130,
        (num_experts, hidden, inter // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w13_b = (
        torch.randn(
            num_experts, 2 * inter, dtype=torch.float32, device="cuda", generator=g
        ).to(torch.bfloat16)
        * 0.01
    )
    w2_b = (
        torch.randn(
            num_experts, hidden, dtype=torch.float32, device="cuda", generator=g
        ).to(torch.bfloat16)
        * 0.01
    )
    return w13, w2, w13_s, w2_s, w13_b, w2_b


def _make_topk(tokens, num_experts, top_k, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(torch.float32), ids.to(torch.int32)


def _build_mock_layer(num_experts, hidden, inter, w13, w2, w13_s, w2_s, w13_b, w2_b):
    layer = _MockLayer()
    layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_s.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_s.clone(), requires_grad=False)
    layer.w13_weight_bias = torch.nn.Parameter(w13_b.clone(), requires_grad=False)
    layer.w2_weight_bias = torch.nn.Parameter(w2_b.clone(), requires_grad=False)
    layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.moe_ep_size = 1
    layer.moe_ep_rank = 0
    return layer


def _build_method(num_experts, hidden, inter):
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method._fi_kernel = "cutlass_sm90"
    method.num_experts = num_experts
    method.hidden_size = hidden
    method.intermediate_size_per_partition = inter
    method.use_flashinfer = True
    return method


@pytest.mark.parametrize(
    "num_experts,hidden,inter",
    [
        (4, 256, 256),
        (8, 768, 384),
        (8, 1024, 1024),
    ],
)
def test_process_weights_matches_direct_interleave(num_experts, hidden, inter):
    """``_process_weights_for_sm90_cutlass`` should produce exactly the same
    interleaved bytes / scales as direct calls to the FlashInfer helpers."""
    w13, w2, w13_s, w2_s, w13_b, w2_b = _make_random_mxfp4(num_experts, hidden, inter)

    layer = _build_mock_layer(
        num_experts, hidden, inter, w13, w2, w13_s, w2_s, w13_b, w2_b
    )
    method = _build_method(num_experts, hidden, inter)
    method._process_weights_for_sm90_cutlass(layer)

    # Reference: drive the helpers directly on the originals.
    ref_w13 = interleave_moe_weights_for_sm90_mixed_gemm(w13, "fp4")
    ref_w2 = interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4")
    ref_w13_s = interleave_moe_scales_for_sm90_mixed_gemm(w13_s, group_size=GROUP_SIZE)
    ref_w2_s = interleave_moe_scales_for_sm90_mixed_gemm(w2_s, group_size=GROUP_SIZE)

    assert torch.equal(layer.w13_weight.data, ref_w13)
    assert torch.equal(layer.w2_weight.data, ref_w2)
    assert torch.equal(layer.w13_weight_scale.data, ref_w13_s)
    assert torch.equal(layer.w2_weight_scale.data, ref_w2_s)

    # SwiGLU per-expert scalars seeded with GPT-OSS defaults.
    assert torch.allclose(
        layer.swiglu_alpha,
        torch.full((num_experts,), 1.702, dtype=torch.float32, device="cuda"),
    )
    assert torch.allclose(
        layer.swiglu_beta,
        torch.full((num_experts,), 1.0, dtype=torch.float32, device="cuda"),
    )
    assert torch.allclose(
        layer.swiglu_limit,
        torch.full((num_experts,), 7.0, dtype=torch.float32, device="cuda"),
    )

    # Biases are passed through unchanged.
    assert torch.equal(layer.w13_weight_bias.data, w13_b)
    assert torch.equal(layer.w2_weight_bias.data, w2_b)


@pytest.mark.parametrize(
    "tokens,num_experts,hidden,inter,top_k",
    [
        (4, 4, 256, 256, 2),
        (16, 8, 768, 384, 2),
        (32, 8, 1024, 1024, 4),
    ],
)
def test_apply_sm90_cutlass_matches_flashinfer_direct(
    tokens, num_experts, hidden, inter, top_k, monkeypatch
):
    """End-to-end: SGLang's ``_apply_sm90_cutlass`` must produce the same
    output as a direct FlashInfer ``cutlass_fused_moe`` call with the same
    input tensors. (Both call the same SM90 kernel under the hood.)"""
    import sglang.srt.layers.quantization.mxfp4 as mxfp4_mod

    # Bypass the symmetric-memory / TP-group stack: not relevant to numerics
    # and requires distributed init we don't have here.
    monkeypatch.setattr(
        mxfp4_mod, "use_symmetric_memory", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(mxfp4_mod, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(mxfp4_mod, "get_tp_group", lambda: None)

    w13, w2, w13_s, w2_s, w13_b, w2_b = _make_random_mxfp4(num_experts, hidden, inter)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    topk_w, topk_i = _make_topk(tokens, num_experts, top_k)

    # ---- SGLang path ----
    layer = _build_mock_layer(
        num_experts, hidden, inter, w13, w2, w13_s, w2_s, w13_b, w2_b
    )
    method = _build_method(num_experts, hidden, inter)
    method._process_weights_for_sm90_cutlass(layer)

    out_sglang = method._apply_sm90_cutlass(
        layer, x.clone(), _MockTopKOutput(topk_w, topk_i)
    ).hidden_states

    # ---- FlashInfer-direct reference ----
    ref_w13 = interleave_moe_weights_for_sm90_mixed_gemm(w13, "fp4")
    ref_w2 = interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4")
    ref_w13_s = interleave_moe_scales_for_sm90_mixed_gemm(w13_s, group_size=GROUP_SIZE)
    ref_w2_s = interleave_moe_scales_for_sm90_mixed_gemm(w2_s, group_size=GROUP_SIZE)
    swiglu_alpha = torch.full((num_experts,), 1.702, dtype=torch.float32, device="cuda")
    swiglu_beta = torch.full((num_experts,), 1.0, dtype=torch.float32, device="cuda")
    swiglu_limit = torch.full((num_experts,), 7.0, dtype=torch.float32, device="cuda")

    out_ref = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    cutlass_fused_moe(
        input=x.clone(),
        token_selected_experts=topk_i,
        token_final_scales=topk_w,
        fc1_expert_weights=ref_w13,
        fc2_expert_weights=ref_w2,
        output_dtype=torch.bfloat16,
        quant_scales=[ref_w13_s.view(torch.int32), ref_w2_s.view(torch.int32)],
        fc1_expert_biases=w13_b,
        fc2_expert_biases=w2_b,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        output=out_ref,
    )

    # Same kernel, same inputs => bit-exact.
    assert torch.equal(out_sglang, out_ref), (
        f"SGLang vs FlashInfer-direct mismatch; "
        f"max abs diff = {(out_sglang.float() - out_ref.float()).abs().max().item():.4g}"
    )


# =============================================================================
# DeepSeek-V4 path: Mxfp4FlashinferCutlassMoEMethod (sibling of Marlin /
# trtllm-gen). Wired into fp8.py's get_quant_method when SM90 +
# is_flashinfer_mxfp4 + is_fp4_experts.
# =============================================================================


def _make_random_dsv4_mxfp4(num_experts, hidden, inter, seed=0):
    """Mirrors the fp8 base method's allocation for fp4 experts: int8-packed
    4-bit weights, fp32 scales (containing 2**e values, not raw E8M0 bytes)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    # int8 storage (signed) -- matches Fp8MoEMethod.create_weights for fp4_experts.
    w13 = torch.randint(
        -128,
        128,
        (num_experts, 2 * inter, hidden // 2),
        dtype=torch.int8,
        device="cuda",
        generator=g,
    )
    w2 = torch.randint(
        -128,
        128,
        (num_experts, hidden, inter // 2),
        dtype=torch.int8,
        device="cuda",
        generator=g,
    )
    # fp32 scales whose bit pattern after .to(float8_e8m0fnu).view(uint8) lands
    # in a sane E8M0 band -- generate exponents around 0 (= 2**0).
    raw_e = torch.randint(
        125,
        130,
        (num_experts, 2 * inter, hidden // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    raw_e2 = torch.randint(
        125,
        130,
        (num_experts, hidden, inter // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w13_s = raw_e.view(torch.float8_e8m0fnu).to(torch.float32)
    w2_s = raw_e2.view(torch.float8_e8m0fnu).to(torch.float32)
    return w13, w2, w13_s, w2_s


@pytest.mark.parametrize(
    "tokens,num_experts,hidden,inter,top_k",
    [
        (4, 4, 256, 256, 2),
        (16, 8, 768, 384, 2),
        (256, 8, 1024, 1024, 4),
    ],
)
def test_dsv4_apply_matches_flashinfer_direct(
    tokens, num_experts, hidden, inter, top_k, monkeypatch
):
    """End-to-end: SGLang's DSv4 ``Mxfp4FlashinferCutlassMoEMethod.apply``
    output must match a direct FlashInfer ``cutlass_fused_moe`` call with
    the equivalent reorder + scale-cast + interleave applied manually."""
    from types import SimpleNamespace

    import sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe as ds_mod
    from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1

    # Bypass symmetric-memory / TP-group stack -- not relevant to numerics.
    monkeypatch.setattr(ds_mod, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(ds_mod, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(ds_mod, "get_tp_group", lambda: None)

    w13, w2, w13_s, w2_s = _make_random_dsv4_mxfp4(num_experts, hidden, inter)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    topk_w, topk_i = _make_topk(tokens, num_experts, top_k)

    # ---- SGLang DSv4 path ----
    method = ds_mod.Mxfp4FlashinferCutlassMoEMethod.__new__(
        ds_mod.Mxfp4FlashinferCutlassMoEMethod
    )
    method._fp8 = SimpleNamespace(
        process_weights_after_loading=lambda layer: None,
    )
    method.prefix = "test"
    method._swiglu_limit_tensor = None  # plain SiLU * up

    layer = _MockLayer()
    layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(w13_s.clone(), requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(w2_s.clone(), requires_grad=False)
    layer.num_local_experts = num_experts
    layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.moe_ep_size = 1
    layer.moe_ep_rank = 0

    method.process_weights_after_loading(layer)

    out_sglang = method.apply(
        layer, _MockDispatchOutput(x.clone(), topk_w, topk_i)
    ).hidden_states

    # ---- Direct FlashInfer reference ----
    w13_re, w13_s_re = reorder_w1w3_to_w3w1(w13, w13_s)
    w13_s_u8 = w13_s_re.to(torch.float8_e8m0fnu).view(torch.uint8).contiguous()
    w2_s_u8 = w2_s.to(torch.float8_e8m0fnu).view(torch.uint8).contiguous()
    ref_w13 = interleave_moe_weights_for_sm90_mixed_gemm(
        w13_re.view(torch.uint8).contiguous(), "fp4"
    )
    ref_w2 = interleave_moe_weights_for_sm90_mixed_gemm(
        w2.view(torch.uint8).contiguous(), "fp4"
    )
    ref_w13_s = interleave_moe_scales_for_sm90_mixed_gemm(
        w13_s_u8, group_size=GROUP_SIZE
    )
    ref_w2_s = interleave_moe_scales_for_sm90_mixed_gemm(w2_s_u8, group_size=GROUP_SIZE)

    out_ref = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    cutlass_fused_moe(
        input=x.clone(),
        token_selected_experts=topk_i,
        token_final_scales=topk_w,
        fc1_expert_weights=ref_w13,
        fc2_expert_weights=ref_w2,
        output_dtype=torch.bfloat16,
        quant_scales=[ref_w13_s.view(torch.int32), ref_w2_s.view(torch.int32)],
        fc1_expert_biases=None,
        fc2_expert_biases=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=None,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        output=out_ref,
    )

    assert torch.equal(out_sglang, out_ref), (
        f"DSv4 SGLang vs FlashInfer-direct mismatch; "
        f"max abs diff = "
        f"{(out_sglang.float() - out_ref.float()).abs().max().item():.4g}"
    )


class _MockDispatchOutput:
    """Stand-in for StandardDispatchOutput. ``topk_output`` is a real
    ``StandardTopKOutput`` so ``TopKOutputChecker.format_is_standard``
    (an isinstance check) returns True without distributed init."""

    def __init__(self, hidden_states, topk_weights, topk_ids):
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        self.hidden_states = hidden_states
        # router_logits is unused by Mxfp4FlashinferCutlassMoEMethod.apply;
        # supply a placeholder of the right shape to keep the NamedTuple happy.
        router_logits = torch.zeros(
            topk_ids.shape[0],
            int(topk_ids.max().item()) + 1 if topk_ids.numel() else 1,
            dtype=torch.float32,
            device=topk_ids.device,
        )
        self.topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
