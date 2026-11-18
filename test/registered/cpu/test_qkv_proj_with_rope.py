import unittest
from types import SimpleNamespace

import pytest
import sgl_kernel  # noqa: F401
import torch
from torch._inductor import config as inductor_config

from sglang.srt.layers import amx_utils
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import input_to_float8
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.model_executor import cpu_graph_runner
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_fused_rope_cpu as cpu_mla,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import (
    convert_weight,
    native_w8a8_per_token_matmul,
    per_token_quant_int8,
    precision,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="extra-test-cpu-intel")

convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
qkv_proj_with_rope = torch.ops.sgl_kernel.qkv_proj_with_rope
qkv_proj_with_rope_fused_weight = torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight
torch.manual_seed(1234)
# constants
kv_lora_rank = 512
qk_head_dim = 192
qk_nope_head_dim = 128
qk_rope_head_dim = 64
rotary_dim = qk_rope_head_dim
num_heads = 22
q_lora_rank = 1536
hidden_size = 7168
B = 1
eps = 1e-6


@pytest.fixture
def fp8_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def layernorm(x, weight, variance_epsilon=1e-6, residual=None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return (x * weight).to(orig_dtype)


def rotary_emb(q_pe, k_pe, pos, cos_sin_cache):
    orig_dtype = q_pe.dtype
    q_pe = q_pe.float()
    k_pe = k_pe.float()
    cos_sin_cache = cos_sin_cache.float()

    query_rot = q_pe[..., :rotary_dim]
    key_rot = k_pe[..., :rotary_dim]
    cos_sin = cos_sin_cache[pos]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query_rot = apply_rotary_emb(query_rot, cos, sin, False)
    key_rot = apply_rotary_emb(key_rot, cos, sin, False)
    return query_rot.to(orig_dtype), key_rot.to(orig_dtype)


def native_torch(
    q_input,
    hidden_states,
    q_a_proj_weight,
    norm_weight1,
    q_b_proj_weight,
    w_kc,
    kv_a_proj_weight,
    norm_weight2,
    pos,
    cos_sin_cache,
):

    q = torch.matmul(hidden_states, q_a_proj_weight.t())
    q = layernorm(q, norm_weight1)
    q = torch.matmul(q, q_b_proj_weight.t()).view(-1, num_heads, qk_head_dim)

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., :kv_lora_rank] = q_nope_out.transpose(0, 1)
    latent_cache = torch.matmul(hidden_states, kv_a_proj_weight.t())
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = layernorm(v_input.contiguous(), norm_weight2).unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank:]

    q_pe, k_pe = rotary_emb(q_pe, k_pe, pos, cos_sin_cache)
    q_input[..., kv_lora_rank:] = q_pe
    k_input[..., kv_lora_rank:] = k_pe

    return q_input, k_input, v_input


def native_torch_int8(
    q_input,
    hidden_states,
    w1_q,
    w1_s,
    norm_weight1,
    w2_q,
    w2_s,
    w_kc,
    w3_q,
    w3_s,
    norm_weight2,
    pos,
    cos_sin_cache,
):

    a_q, a_s = per_token_quant_int8(hidden_states)
    q = native_w8a8_per_token_matmul(a_q, w1_q, a_s, w1_s, None, torch.bfloat16)
    q = layernorm(q, norm_weight1)

    a_q, a_s = per_token_quant_int8(q)
    q = native_w8a8_per_token_matmul(a_q, w2_q, a_s, w2_s, None, torch.bfloat16).view(
        -1, num_heads, qk_head_dim
    )

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., :kv_lora_rank] = q_nope_out.transpose(0, 1)
    a_q, a_s = per_token_quant_int8(hidden_states)
    latent_cache = native_w8a8_per_token_matmul(
        a_q, w3_q, a_s, w3_s, None, torch.bfloat16
    )
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = layernorm(v_input.contiguous(), norm_weight2).unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank:]

    q_pe, k_pe = rotary_emb(q_pe, k_pe, pos, cos_sin_cache)
    q_input[..., kv_lora_rank:] = q_pe
    k_input[..., kv_lora_rank:] = k_pe

    return q_input, k_input, v_input


def fp8_rows_reference(x):
    x = x.float()
    scales = (x.abs().amax(dim=-1) / 448).clamp_min(torch.finfo(torch.float32).eps)
    return (x * scales.reciprocal().unsqueeze(-1)).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ), scales


def fp8_linear_reference(x, weight, scales):
    xq, xs = fp8_rows_reference(x)
    result = torch.zeros(x.shape[0], weight.shape[0])
    scales = scales.repeat_interleave(128, 0)[: weight.shape[0]]
    for group in range(weight.shape[1] // 128):
        begin = group * 128
        result += (
            xq.float()[:, begin : begin + 128]
            @ weight.float()[:, begin : begin + 128].T
        ) * scales[:, group]
    return (result * xs.unsqueeze(-1)).bfloat16()


def make_fp8_w8a8_qkv_inputs(m, monkeypatch, hidden=256, q_rank=256, heads=2):
    monkeypatch.setattr(fp8_module, "_is_cpu", True)
    monkeypatch.setattr(fp8_module, "_is_cpu_amx_available", True)
    monkeypatch.setattr(fp8_module, "SGLANG_DEEPSEEK_FP8A8", True)
    monkeypatch.setattr(fp8_module, "SGLANG_BRGEMM_REF_FP8A8", False)
    generator = torch.Generator().manual_seed(1234)
    config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    x = torch.randn(m, hidden, dtype=torch.bfloat16, generator=generator)
    methods = []
    layers = []
    raw_weights = []
    for prefix, n, k in (
        ("fused_qkv_a_proj_with_mqa", q_rank + 576, hidden),
        ("q_b_proj", heads * 192, q_rank),
    ):
        layer = torch.nn.Module()
        weight = (torch.randn(n, k, generator=generator) * 8).to(torch.float8_e4m3fn)
        scales = (
            torch.rand((n + 127) // 128, k // 128, generator=generator) + 0.5
        ) * 0.02
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
        method = Fp8LinearMethod(config, prefix)
        method.process_weights_after_loading(layer)
        assert layer.use_f8f8 and layer.weight.ndim == 4
        methods.append(method)
        layers.append(layer)
        raw_weights.append((weight, scales))
    attention = torch.nn.Module()
    wkc = (torch.randn(heads, 512, 128, generator=generator) * 8).to(
        torch.float8_e4m3fn
    )
    attention.w_kc = torch.nn.Parameter(wkc.transpose(1, 2), requires_grad=False)
    attention.w_vc = torch.nn.Parameter(
        torch.randn(heads, 512, 128, generator=generator).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    attention.w_scale = torch.tensor(0.02)
    amx_utils.PackWeightMethod(
        ["w_kc", "w_vc"], [[1, 2], [1, 2]], use_fp8_w8a8=True
    ).process_weights_after_loading(attention)
    norm_q = torch.randn(q_rank, dtype=torch.bfloat16, generator=generator)
    norm_k = torch.randn(512, dtype=torch.bfloat16, generator=generator)
    pos = torch.arange(m) % 32
    cache = torch.randn(32, 64, dtype=torch.bfloat16, generator=generator)
    latent = fp8_linear_reference(x, *raw_weights[0])
    qa = layernorm(latent[:, :q_rank], norm_q)
    k = latent[:, q_rank:].clone().unsqueeze(1)
    k[..., :512] = layernorm(k[..., :512], norm_k)
    qb = fp8_linear_reference(qa, *raw_weights[1]).reshape(m, heads, 192)
    qn, qs = fp8_rows_reference(qb[..., :128].transpose(0, 1))
    q = torch.empty(m, heads, 576, dtype=torch.bfloat16)
    q[..., :512] = (
        (
            torch.bmm(qn.float(), wkc.float().transpose(1, 2))
            * attention.w_scale
            * qs.unsqueeze(-1)
        )
        .bfloat16()
        .transpose(0, 1)
    )
    q[..., 512:], k[..., 512:] = rotary_emb(qb[..., 128:], k[..., 512:], pos, cache)
    a, b = layers
    fused_args = (
        x,
        a.weight,
        b.weight,
        attention.w_kc,
        norm_q,
        norm_k,
        pos,
        cache,
        eps,
        False,
        True,
        a.weight_scale_inv,
        b.weight_scale_inv,
        attention.w_kc_scale,
        True,
        [128, 128],
        q_rank,
        512,
        64,
    )
    qa_weight, kva_weight = a.weight.split([q_rank // 32, 576 // 32])
    qa_scales, kva_scales = a.weight_scale_inv.split([q_rank // 32, 576 // 32])
    separate_args = (
        x,
        qa_weight,
        b.weight,
        kva_weight,
        attention.w_kc,
        norm_q,
        norm_k,
        pos,
        cache,
        eps,
        False,
        True,
        qa_scales,
        b.weight_scale_inv,
        kva_scales,
        attention.w_kc_scale,
        True,
        [128, 128],
    )
    return fused_args, separate_args, (q, k, k[..., :512]), methods, layers, raw_weights


@pytest.mark.parametrize(
    "m,hidden,q_rank,heads",
    [
        (0, 256, 256, 2),
        (1, 256, 256, 2),
        (35, 256, 256, 2),
        (96, 256, 256, 2),
        (1, 7168, 1536, 32),
        (96, 7168, 1536, 32),
    ],
)
def test_fp8_w8a8_qkv(m, hidden, q_rank, heads, monkeypatch, fp8_rng):
    fused, separate, expected, methods, layers, weights = make_fp8_w8a8_qkv_inputs(
        m, monkeypatch, hidden, q_rank, heads
    )
    for op, args in (
        (qkv_proj_with_rope_fused_weight, fused),
        (qkv_proj_with_rope, separate),
    ):
        result = op(*args)
        for index, (actual, ref) in enumerate(zip(result, expected)):
            assert actual.shape == ref.shape and actual.dtype == ref.dtype
            if index == 0:
                # Upstream BF16 rounding ties can be amplified by requantization.
                relative_error = (
                    actual.float() - ref.float()
                ).norm() / ref.float().norm().clamp_min(1e-12)
                assert relative_error <= 0.001
            else:
                torch.testing.assert_close(actual, ref, atol=0.003, rtol=0.01)
        if m:
            assert (
                result[1].untyped_storage().data_ptr()
                == result[2].untyped_storage().data_ptr()
            )
    # Prefill uses the same packed modules through Fp8LinearMethod.apply.
    for index, (method, layer, raw) in enumerate(zip(methods, layers, weights)):
        x = fused[0] if index == 0 else torch.randn(m, q_rank, dtype=torch.bfloat16)
        result = method.apply(layer, x)
        torch.testing.assert_close(
            result, fp8_linear_reference(x, *raw), atol=0.003, rtol=0.01
        )
    if m and hidden == 7168:
        latent = methods[0].apply(layers[0], fused[0])
        qa = torch.ops.sgl_kernel.rmsnorm_cpu(
            latent[:, :q_rank].contiguous(), fused[4], eps
        )
        qb = methods[1].apply(layers[1], qa).reshape(m, heads, 192)
        q_nope = torch.empty(heads, m, 512, dtype=torch.bfloat16)
        torch.ops.sgl_kernel.bmm_cpu(
            q_nope, qb[..., :128].transpose(0, 1), fused[3], True, fused[13]
        )
        torch.testing.assert_close(
            qkv_proj_with_rope_fused_weight(*fused)[0][..., :512].transpose(0, 1),
            q_nope,
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("cpp_wrapper", [False, True])
def test_fp8_w8a8_qkv_compile(m, cpp_wrapper, monkeypatch, fp8_rng):
    fused, separate, _, _, _, _ = make_fp8_w8a8_qkv_inputs(m, monkeypatch)
    with inductor_config.patch(cpp_wrapper=True):
        cpu_graph_runner.register_fake_ops(tp_size=1)
    for op, args in (
        (qkv_proj_with_rope_fused_weight, fused),
        (qkv_proj_with_rope, separate),
    ):

        def forward(x):
            return op(x, *args[1:])

        with inductor_config.patch(cpp_wrapper=cpp_wrapper):
            compiled = torch.compile(forward, backend="inductor", fullgraph=True)
            result = compiled(args[0])
        for actual, expected in zip(result, op(*args)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fp8_w8a8_cpu_mla_wiring(monkeypatch, fp8_rng):
    fused, _, _, methods, layers, _ = make_fp8_w8a8_qkv_inputs(3, monkeypatch)
    monkeypatch.setattr(cpu_mla, "SGLANG_DEEPSEEK_FP8A8", True)
    for layer, method in zip(layers, methods):
        layer.quant_method = method
    vc = torch.randn(2, 128, 512).to(torch.float8_e4m3fn)
    packed = [
        torch.ops.sgl_kernel.float8_linear_prepack_cpu(
            head, torch.full((128, 4), 0.03125)
        )
        for head in vc
    ]
    attn_output = torch.randn(3, 2, 512, dtype=torch.bfloat16)
    attention = SimpleNamespace(
        has_fused_proj=True,
        is_packed_weight=False,
        use_intel_amx_backend=True,
        fused_qkv_a_proj_with_mqa=layers[0],
        q_b_proj=layers[1],
        w_kc=fused[3],
        w_kc_scale=fused[13],
        w_vc=torch.stack([p[0] for p in packed]),
        w_vc_scale=torch.stack([p[1] for p in packed]),
        w_scale=torch.tensor(9.0),
        q_a_layernorm=SimpleNamespace(weight=fused[4]),
        kv_a_layernorm=SimpleNamespace(weight=fused[5], variance_epsilon=eps),
        rotary_emb=SimpleNamespace(cos_sin_cache=fused[7]),
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_local_heads=2,
        attn_mqa=lambda q, k, v, batch: attn_output,
        o_proj=lambda x: (x, None),
    )
    mixin = cpu_mla.DeepseekMLACpuForwardMixin
    mixin.init_mla_fused_rope_cpu_forward(attention)
    assert attention.quant_method.use_fp8_w8a8
    prepared = mixin.forward_absorb_fused_mla_rope_cpu_prepare(
        attention, fused[6], fused[0], None, None
    )
    for actual, expected in zip(prepared[:3], qkv_proj_with_rope_fused_weight(*fused)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual = mixin.forward_absorb_fused_mla_rope_cpu_core(attention, *prepared)
    aq, scales = fp8_rows_reference(attn_output.transpose(0, 1))
    expected = (
        (
            torch.bmm(aq.float(), vc.float().transpose(1, 2))
            * 0.03125
            * scales[..., None]
        )
        .bfloat16()
        .transpose(0, 1)
        .flatten(1, 2)
    )
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.01)


@pytest.mark.parametrize(
    "prefix", ["fused_qkv_a_proj_with_mqa", "q_b_proj", "shared_experts.gate_up_proj"]
)
@pytest.mark.parametrize(
    "deepseek,reference", [(False, False), (False, True), (True, False)]
)
def test_fp8_projection_legacy_packing(
    prefix, deepseek, reference, monkeypatch, fp8_rng
):
    monkeypatch.setattr(fp8_module, "_is_cpu", True)
    monkeypatch.setattr(fp8_module, "_is_cpu_amx_available", True)
    monkeypatch.setattr(fp8_module, "SGLANG_DEEPSEEK_FP8A8", deepseek)
    monkeypatch.setattr(fp8_module, "SGLANG_BRGEMM_REF_FP8A8", reference)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(256, 256).to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]),
        prefix,
    )
    method.process_weights_after_loading(layer)
    expected_a8 = deepseek
    assert layer.use_f8f8 == expected_a8
    assert layer.weight.ndim == (4 if expected_a8 else 2)


class TestQKVProjWithROPE(CustomTestCase):
    def test_bf16_qkv_proj_with_rope(self):
        dtype = torch.bfloat16
        hidden_states = torch.randn(B, hidden_size, dtype=dtype) / hidden_size
        q_input = torch.empty(
            B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype
        )
        q_a_proj_weight = torch.randn(q_lora_rank, hidden_size, dtype=dtype) * 0.1
        norm_weight1 = torch.randn(q_lora_rank, dtype=dtype)
        q_b_proj_weight = (
            torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=dtype) * 0.1
        )
        w_kc = torch.randn(num_heads, kv_lora_rank, qk_nope_head_dim, dtype=dtype) * 0.1
        kv_a_proj_weight = (
            torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size, dtype=dtype) * 0.1
        )
        fused_weight = torch.cat([q_a_proj_weight, kv_a_proj_weight], dim=0)
        norm_weight2 = torch.randn(kv_lora_rank, dtype=dtype)
        pos = torch.randint(10, 100, (B,))
        cos_sin_cache = torch.randn(100, rotary_dim, dtype=dtype)
        q_ref, k_ref, v_ref = native_torch(
            q_input,
            hidden_states,
            q_a_proj_weight,
            norm_weight1,
            q_b_proj_weight,
            w_kc.transpose(1, 2),
            kv_a_proj_weight,
            norm_weight2,
            pos,
            cos_sin_cache,
        )
        qa_packed = convert_weight_packed(q_a_proj_weight)
        qb_packed = convert_weight_packed(q_b_proj_weight)
        kva_packed = convert_weight_packed(kv_a_proj_weight)
        wkc_packed = convert_weight_packed(w_kc)
        fused_weight_packed = convert_weight_packed(fused_weight)

        q_out, k_out, v_out = qkv_proj_with_rope(
            hidden_states,
            qa_packed,
            qb_packed,
            kva_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            False,
            None,
            None,
            None,
            None,
            True,
            None,
        )
        fused_q_out, fused_k_out, fused_v_out = qkv_proj_with_rope_fused_weight(
            hidden_states,
            fused_weight_packed,
            qb_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            False,
            None,
            None,
            None,
            True,
            None,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        atol = rtol = precision[q_ref.dtype]
        torch.testing.assert_close(q_ref, q_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_ref, k_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_ref, v_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(fused_q_out, q_out)
        torch.testing.assert_close(fused_k_out, k_out)
        torch.testing.assert_close(fused_v_out, v_out)

    def test_int8_qkv_proj_with_rope(self):
        dtype = torch.bfloat16
        hidden_states = torch.randn(B, hidden_size, dtype=dtype) / hidden_size
        q_input = torch.empty(
            B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype
        )
        q_a_proj_weight = torch.randn(q_lora_rank, hidden_size, dtype=dtype) * 0.1
        norm_weight1 = torch.randn(q_lora_rank, dtype=dtype)
        q_b_proj_weight = (
            torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=dtype) * 0.1
        )
        w_kc = torch.randn(num_heads, kv_lora_rank, qk_nope_head_dim, dtype=dtype) * 0.1
        kv_a_proj_weight = (
            torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size, dtype=dtype) * 0.1
        )
        norm_weight2 = torch.randn(kv_lora_rank, dtype=dtype)
        pos = torch.randint(10, 100, (B,))
        cos_sin_cache = torch.randn(100, rotary_dim, dtype=dtype)

        w1_q, w1_s = per_token_quant_int8(q_a_proj_weight)
        w2_q, w2_s = per_token_quant_int8(q_b_proj_weight)
        w3_q, w3_s = per_token_quant_int8(kv_a_proj_weight)
        q_ref, k_ref, v_ref = native_torch_int8(
            q_input,
            hidden_states,
            w1_q,
            w1_s,
            norm_weight1,
            w2_q,
            w2_s,
            w_kc.transpose(1, 2),
            w3_q,
            w3_s,
            norm_weight2,
            pos,
            cos_sin_cache,
        )
        w1_q_packed = convert_weight_packed(w1_q)
        w2_q_packed = convert_weight_packed(w2_q)
        w3_q_packed = convert_weight_packed(w3_q)
        wkc_packed = convert_weight_packed(w_kc)
        q_out, k_out, v_out = qkv_proj_with_rope(
            hidden_states,
            w1_q_packed,
            w2_q_packed,
            w3_q_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            True,
            False,
            w1_s,
            w2_s,
            w3_s,
            None,
            True,
            None,
        )
        fused_weight = torch.cat([w1_q, w3_q], dim=0)
        fused_weight_s = torch.cat([w1_s, w3_s], dim=0)
        w_fused_q_packed = convert_weight_packed(fused_weight)
        fused_q_out, fused_k_out, fused_v_out = qkv_proj_with_rope_fused_weight(
            hidden_states,
            w_fused_q_packed,
            w2_q_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            True,
            False,
            fused_weight_s,
            w2_s,
            None,
            True,
            None,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        atol = rtol = precision[q_ref.dtype]
        torch.testing.assert_close(q_ref, q_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_ref, k_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_ref, v_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(fused_q_out, q_out)
        torch.testing.assert_close(fused_k_out, k_out)
        torch.testing.assert_close(fused_v_out, v_out)

    def test_fp8_qkv_proj_with_rope(self):
        dtype = torch.bfloat16
        hidden_states = torch.randn(B, hidden_size, dtype=dtype) / hidden_size
        q_input = torch.empty(
            B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype
        )
        q_a_proj_weight = torch.randn(q_lora_rank, hidden_size, dtype=dtype) * 0.1
        norm_weight1 = torch.randn(q_lora_rank, dtype=dtype)
        q_b_proj_weight = (
            torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=dtype) * 0.1
        )
        w_kc = torch.randn(num_heads, kv_lora_rank, qk_nope_head_dim, dtype=dtype) * 0.1
        w_kc_q, w_kc_s = input_to_float8(w_kc)
        kv_a_proj_weight = (
            torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size, dtype=dtype) * 0.1
        )
        norm_weight2 = torch.randn(kv_lora_rank, dtype=dtype)
        pos = torch.randint(10, 100, (B,))
        cos_sin_cache = torch.randn(100, rotary_dim, dtype=dtype)

        scale_block_size_N = 128
        scale_block_size_K = 128
        fp8_q_a_proj_weight, q_a_proj_weight_scale_inv, q_a_proj_weight_dq = (
            convert_weight(
                q_a_proj_weight,
                [scale_block_size_N, scale_block_size_K],
                torch.bfloat16,
            )
        )
        fp8_q_b_proj_weight, q_b_proj_weight_scale_inv, q_b_proj_weight_dq = (
            convert_weight(
                q_b_proj_weight,
                [scale_block_size_N, scale_block_size_K],
                torch.bfloat16,
            )
        )
        (
            fp8_kv_a_proj_with_mqa_weight,
            kv_a_proj_with_mqa_weight_scale_inv,
            kv_a_proj_with_mqa_weight_dq,
        ) = convert_weight(
            kv_a_proj_weight, [scale_block_size_N, scale_block_size_K], torch.bfloat16
        )
        w_kc_dq = w_kc_q.to(torch.bfloat16) * w_kc_s
        q_ref, k_ref, v_ref = native_torch(
            q_input,
            hidden_states,
            q_a_proj_weight_dq,
            norm_weight1,
            q_b_proj_weight_dq,
            w_kc_dq.transpose(1, 2),
            kv_a_proj_with_mqa_weight_dq,
            norm_weight2,
            pos,
            cos_sin_cache,
        )
        fp8_q_a_proj_weight_packed = convert_weight_packed(fp8_q_a_proj_weight)
        fp8_q_b_proj_weight_packed = convert_weight_packed(fp8_q_b_proj_weight)
        fp8_kv_a_proj_with_mqa_weight_packed = convert_weight_packed(
            fp8_kv_a_proj_with_mqa_weight
        )
        w_kc_q = convert_weight_packed(w_kc_q)
        q_out, k_out, v_out = qkv_proj_with_rope(
            hidden_states,
            fp8_q_a_proj_weight_packed,
            fp8_q_b_proj_weight_packed,
            fp8_kv_a_proj_with_mqa_weight_packed,
            w_kc_q,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            True,
            q_a_proj_weight_scale_inv.float(),
            q_b_proj_weight_scale_inv.float(),
            kv_a_proj_with_mqa_weight_scale_inv.float(),
            w_kc_s,
            True,
            [scale_block_size_N, scale_block_size_K],
        )

        fused_weight = torch.cat(
            [fp8_q_a_proj_weight, fp8_kv_a_proj_with_mqa_weight], dim=0
        )
        fused_weight_s = torch.cat(
            [q_a_proj_weight_scale_inv, kv_a_proj_with_mqa_weight_scale_inv], dim=0
        )
        fused_weight_packed = convert_weight_packed(fused_weight)
        fused_q_out, fused_k_out, fused_v_out = qkv_proj_with_rope_fused_weight(
            hidden_states,
            fused_weight_packed,
            fp8_q_b_proj_weight_packed,
            w_kc_q,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            True,
            fused_weight_s.float(),
            q_b_proj_weight_scale_inv.float(),
            w_kc_s,
            True,
            [scale_block_size_N, scale_block_size_K],
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        atol = rtol = precision[q_ref.dtype]
        # Due to the change in multiplication order, the error is amplified.
        # In the model, with fewer layers, this doesn't cause issues, but in
        # tests with more layers, we need to enlarge the tolerance to pass the tests.
        torch.testing.assert_close(q_ref, q_out, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(k_ref, k_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_ref, v_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(fused_q_out, q_out)
        torch.testing.assert_close(fused_k_out, k_out)
        torch.testing.assert_close(fused_v_out, v_out)


if __name__ == "__main__":
    unittest.main()
