"""Unit test for the SM90 cutlass MXFP4 path in :class:`Mxfp4MoEMethod`.

Builds a single-layer GPT-OSS-style MoE with random MXFP4 weights, drives the
SGLang plumbing (``_process_weights_for_sm90_cutlass`` + ``_apply_sm90_cutlass``)
and compares against a direct FlashInfer ``cutlass_fused_moe`` call with the
same inputs. Both paths invoke the same SM90 kernel from FlashInfer PR #3084,
so outputs must be bit-exact.

Run on H100/H200:

    python -m pytest test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py -v
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="stage-b", runner_config="1-gpu-large")

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
    layer.num_local_experts = num_experts  # tests run with EP size = 1
    layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.moe_ep_size = 1
    layer.moe_ep_rank = 0
    return layer


def _round_up(x, base):
    return ((x + base - 1) // base) * base


def _build_method(num_experts, hidden, inter):
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method._fi_kernel = "cutlass_sm90"
    method.num_experts = num_experts
    # The new SM90 cutlass path tracks padded sizes in dedicated attrs;
    # ``hidden_size`` / ``intermediate_size_per_partition`` keep the unpadded
    # values to mirror what ``create_weights`` records.
    method.hidden_size = hidden
    method.intermediate_size_per_partition = inter
    method._padded_hidden = _round_up(hidden, 128)
    method._padded_intermediate = _round_up(inter, 128)
    method.use_flashinfer = True
    return method


def _expected_w13_processed(w13_un, w13_s_un, w13_b_un, N_pad, K_pad, group_size):
    """Replicate ``_process_weights_for_sm90_cutlass`` for w13: de-interleave
    HF's pair-wise ``[g_0, u_0, g_1, u_1, ...]`` layout into halved
    ``[up; gate]``, pad each half along its row dim from ``N_un -> N_pad``
    and last dim from ``K_un -> K_pad`` with zeros, then run the FlashInfer
    SM90 byte / scale interleave helpers."""
    E, two_n_un, last_un_w = w13_un.shape
    N_un = two_n_un // 2
    K_un = last_un_w * 2  # packed 4-bit -> *2 for raw K

    def _split_and_pad(unpadded, last_pad, last_un, dtype):
        gate = unpadded[:, 0::2, :]
        up = unpadded[:, 1::2, :]
        out = torch.zeros(E, 2 * N_pad, last_pad, dtype=dtype, device=unpadded.device)
        out[:, :N_un, :last_un] = up
        out[:, N_pad : N_pad + N_un, :last_un] = gate
        return out

    w13_pad = _split_and_pad(
        w13_un.view(torch.uint8), K_pad // 2, K_un // 2, w13_un.dtype
    )
    w13_s_pad = _split_and_pad(
        w13_s_un, K_pad // group_size, K_un // group_size, w13_s_un.dtype
    )

    gate_b = w13_b_un[:, 0::2]
    up_b = w13_b_un[:, 1::2]
    w13_b_pad = torch.zeros(E, 2 * N_pad, dtype=w13_b_un.dtype, device=w13_b_un.device)
    w13_b_pad[:, :N_un] = up_b
    w13_b_pad[:, N_pad : N_pad + N_un] = gate_b

    w13_il = interleave_moe_weights_for_sm90_mixed_gemm(w13_pad, "fp4")
    w13_s_il = interleave_moe_scales_for_sm90_mixed_gemm(
        w13_s_pad, group_size=group_size
    )
    return w13_il, w13_s_il, w13_b_pad


def _expected_w2_processed(w2_un, w2_s_un, w2_b_un, N_pad, K_pad, group_size):
    """w2 needs padding only (no halving / no de-interleave)."""
    E, K_un, last_un_w = w2_un.shape
    N_un = last_un_w * 2

    def _pad(unpadded, last_pad, last_un):
        out = torch.zeros(
            E, K_pad, last_pad, dtype=unpadded.dtype, device=unpadded.device
        )
        out[:, :K_un, :last_un] = unpadded
        return out

    w2_pad = _pad(w2_un.view(torch.uint8), N_pad // 2, N_un // 2)
    w2_s_pad = _pad(w2_s_un, N_pad // group_size, N_un // group_size)
    w2_b_pad = torch.zeros(E, K_pad, dtype=w2_b_un.dtype, device=w2_b_un.device)
    w2_b_pad[:, :K_un] = w2_b_un

    w2_il = interleave_moe_weights_for_sm90_mixed_gemm(w2_pad, "fp4")
    w2_s_il = interleave_moe_scales_for_sm90_mixed_gemm(w2_s_pad, group_size=group_size)
    return w2_il, w2_s_il, w2_b_pad


@pytest.mark.parametrize(
    "num_experts,hidden,inter",
    [
        # Aligned shapes (no padding needed).
        (4, 256, 256),
        (8, 768, 384),
        (8, 1024, 1024),
        # Non-aligned shapes (exercise the de-interleave + pad path).
        # 192 % 128 = 64, so N_pad = K_pad = 256 (round_up(192, 128)).
        (4, 192, 192),
        # GPT-OSS-20B-like: hidden=2880, inter=2880 -> padded to 2944.
        # Use smaller E to keep memory bounded.
        (4, 2880, 2880),
    ],
)
def test_process_weights_matches_direct_interleave(num_experts, hidden, inter):
    """``_process_weights_for_sm90_cutlass`` must produce the same bytes as
    a manual de-interleave + pad + halved-swap + interleave reference."""
    w13, w2, w13_s, w2_s, w13_b, w2_b = _make_random_mxfp4(num_experts, hidden, inter)

    layer = _build_mock_layer(
        num_experts, hidden, inter, w13, w2, w13_s, w2_s, w13_b, w2_b
    )
    method = _build_method(num_experts, hidden, inter)
    method._process_weights_for_sm90_cutlass(layer)

    N_pad = _round_up(inter, 128)
    K_pad = _round_up(hidden, 128)
    ref_w13, ref_w13_s, ref_w13_b = _expected_w13_processed(
        w13, w13_s, w13_b, N_pad, K_pad, GROUP_SIZE
    )
    ref_w2, ref_w2_s, ref_w2_b = _expected_w2_processed(
        w2, w2_s, w2_b, N_pad, K_pad, GROUP_SIZE
    )

    assert torch.equal(layer.w13_weight.data, ref_w13)
    assert torch.equal(layer.w2_weight.data, ref_w2)
    assert torch.equal(layer.w13_weight_scale.data, ref_w13_s)
    assert torch.equal(layer.w2_weight_scale.data, ref_w2_s)
    assert torch.equal(layer.w13_weight_bias.data, ref_w13_b)
    assert torch.equal(layer.w2_weight_bias.data, ref_w2_b)

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


@pytest.mark.parametrize(
    "tokens,num_experts,hidden,inter,top_k",
    [
        # Aligned shapes (no padding).
        (4, 4, 256, 256, 2),
        (16, 8, 768, 384, 2),
        (32, 8, 1024, 1024, 4),
        # Non-aligned (exercises pad x + trim output).
        (8, 4, 192, 192, 2),
    ],
)
def test_apply_sm90_cutlass_matches_flashinfer_direct(
    tokens, num_experts, hidden, inter, top_k, monkeypatch
):
    """End-to-end: SGLang's ``_apply_sm90_cutlass`` must produce the same
    output as a direct FlashInfer ``cutlass_fused_moe`` call fed with the
    same processed weights / scales / biases. The processing pipeline is
    covered separately by ``test_process_weights_matches_direct_interleave``;
    here we just verify that ``apply`` calls the kernel with the right
    arguments (incl. input padding + output trim)."""
    import sglang.srt.layers.quantization.mxfp4 as mxfp4_mod

    # Bypass symmetric-memory / TP-group: not relevant to numerics.
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

    # ---- FlashInfer-direct reference using the same processed weights ----
    K_pad = method._padded_hidden
    if K_pad != hidden:
        x_padded = torch.nn.functional.pad(
            x.clone(), (0, K_pad - hidden), mode="constant", value=0.0
        )
    else:
        x_padded = x.clone()

    out_ref_padded = torch.empty(tokens, K_pad, dtype=torch.bfloat16, device="cuda")
    cutlass_fused_moe(
        input=x_padded,
        token_selected_experts=topk_i.to(torch.int),
        token_final_scales=topk_w,
        fc1_expert_weights=layer.w13_weight,
        fc2_expert_weights=layer.w2_weight,
        output_dtype=torch.bfloat16,
        quant_scales=[
            layer.w13_weight_scale.view(torch.int32),
            layer.w2_weight_scale.view(torch.int32),
        ],
        fc1_expert_biases=layer.w13_weight_bias,
        fc2_expert_biases=layer.w2_weight_bias,
        swiglu_alpha=layer.swiglu_alpha,
        swiglu_beta=layer.swiglu_beta,
        swiglu_limit=layer.swiglu_limit,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        output=out_ref_padded,
    )
    out_ref = (
        out_ref_padded[:, :hidden].contiguous() if K_pad != hidden else out_ref_padded
    )

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
    # plain SiLU * up — all three SwiGLU scalars None (no clamp configured).
    method._swiglu_alpha_tensor = None
    method._swiglu_beta_tensor = None
    method._swiglu_limit_tensor = None

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
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
