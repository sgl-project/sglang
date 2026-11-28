# SPDX-License-Identifier: Apache-2.0
"""gfx1201 AITER dispatch: production diffusion shapes and attention semantics."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

# AITER op_tests/op_benchmarks/flydsl/bench_fmha_gfx1201.py, at
# 342aa1dc1971480955ac6ebd504c237e668d75b4: FLUX, Wan, and SD3.5.
DIFFUSION_SHAPES = [
    (1, 1536, 24, 128),
    (1, 4096, 24, 128),
    (1, 8192, 12, 128),
    (1, 32768, 12, 128),
    (1, 2048, 38, 64),
]


@pytest.fixture
def backend(monkeypatch):
    if not (
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.split(":")[0] == "gfx1201"
    ):
        pytest.skip("Requires gfx1201")
    pytest.importorskip("aiter.jit.core")
    pytest.importorskip("aiter.ops.flydsl")
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter

    if not aiter.USE_AITER_GFX1201:
        pytest.skip("Requires SGLANG_USE_AITER=1")
    if aiter._aiter_flash_attn_gfx1201 is None:
        pytest.skip("Requires the gfx1201 FlyDSL FMHA API")
    monkeypatch.setattr(aiter, "_use_fp8_attn", False)
    return aiter


@pytest.fixture(params=[False, True], ids=["bf16", "fp8"])
def fp8(backend, monkeypatch, request):
    enabled = request.param
    if enabled and backend._aiter_fp8_quant_gfx1201 is None:
        pytest.skip("Requires the gfx1201 FlyDSL FP8 quantizer")
    monkeypatch.setattr(backend, "_use_fp8_attn", enabled)
    return enabled


def _reference(q, k, v, scale, causal=False):
    mask = None
    if causal and q.shape[1] != k.shape[1]:
        # Flash Attention aligns unequal causal sequences at the bottom right.
        rows = torch.arange(q.shape[1], device=q.device)[:, None]
        cols = torch.arange(k.shape[1], device=k.device)[None, :]
        mask = cols <= rows + k.shape[1] - q.shape[1]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        scale=scale,
        attn_mask=mask,
        is_causal=causal and mask is None,
        enable_gqa=q.shape[2] != k.shape[2],
    ).transpose(1, 2)


@pytest.mark.parametrize("shape", DIFFUSION_SHAPES)
@torch.no_grad()
def test_diffusion_shapes(backend, monkeypatch, shape, fp8):
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    scale = shape[-1] ** -0.5
    impl = backend.AITerImpl(
        num_heads=shape[2], head_size=shape[3], softmax_scale=scale
    )
    kernel = backend._aiter_flash_attn_gfx1201
    calls = []

    def traced_kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return kernel(*args, **kwargs)

    monkeypatch.setattr(backend, "_aiter_flash_attn_gfx1201", traced_kernel)
    out = impl.forward(q, k, v)
    assert len(calls) == 1
    inputs, kwargs = calls[0]
    assert inputs[0].dtype == (torch.float8_e4m3fn if fp8 else torch.bfloat16)
    assert out.shape == q.shape and out.dtype == q.dtype
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
    ).transpose(1, 2)
    if fp8:
        # Match AITER's benchmark gate against the original BF16 attention.
        torch.testing.assert_close(out, reference, atol=0.2, rtol=0.2)
        cosine = F.cosine_similarity(out.float(), reference.float(), dim=-1)
        assert cosine.min().item() > 0.97
        assert cosine.mean().item() > 0.994
    else:
        torch.testing.assert_close(out, reference, atol=0.008, rtol=0.02)
    # FP32 reference across the entire KV window without allocating an S x S
    # FP32 score matrix for the 32768-token case. Include both sequence ends.
    rows = torch.linspace(0, shape[1] - 1, 64, device=q.device).long()
    if fp8:
        q, k, v = (
            tensor.float() * kwargs[name]
            for tensor, name in zip(inputs, ("q_descale", "k_descale", "v_descale"))
        )
    torch.testing.assert_close(
        out[:, rows].float(), _reference(q[:, rows], k, v, scale), atol=0.008, rtol=0.02
    )


@pytest.mark.parametrize(
    "head_dim,query_len,key_len,nondefault_stream",
    [(64, 63, 77, False), (128, 128, 128, True), (128, 129, 63, True)],
)
@torch.no_grad()
def test_scale_layout_and_stream(
    backend, monkeypatch, head_dim, query_len, key_len, nondefault_stream, fp8
):
    torch.manual_seed(42)
    q = torch.randn(2, query_len, 4, head_dim * 2, device="cuda", dtype=torch.bfloat16)[
        ..., ::2
    ]
    k = torch.randn(2, key_len, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 0.7 * head_dim**-0.5
    impl = backend.AITerImpl(num_heads=4, head_size=head_dim, softmax_scale=scale)
    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = impl.forward(q, k, v).clone()
    torch.cuda.current_stream().wait_stream(stream)
    if fp8:
        q8, k8, v8, sq, sk, sv = backend._aiter_fp8_quant_gfx1201(
            q.contiguous(), k.contiguous(), v.contiguous()
        )
        direct = backend._aiter_flash_attn_gfx1201(
            q8,
            k8,
            v8,
            softmax_scale=scale,
            q_descale=sq,
            k_descale=sk,
            v_descale=sv,
        )
        torch.testing.assert_close(out, direct, atol=0, rtol=0)
        q, k, v = q8.float() * sq, k8.float() * sk, v8.float() * sv
    # The FP8 kernel also rounds intermediate attention probabilities. The
    # observed maximum error against dequantized-input FP32 SDPA is 0.0206.
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, scale), atol=0.03 if fp8 else 0.008, rtol=0.02
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "precision,head_dim",
    [(precision, d) for precision in ("bf16", "fp16") for d in (64, 96, 128, 480)]
    + [("fp8", d) for d in (64, 96, 128, 512, 960)],
)
def test_supported_dtypes_dimensions_and_causal(
    backend, monkeypatch, precision, head_dim, causal
):
    # Cover minimum/maximum, padded and common dimensions; FP8 also covers
    # its largest power-of-two dimension within the gfx1201 LDS limit.
    # Keep gradient mode enabled with non-differentiable inputs: inference must
    # still select FlyDSL, just as AITER's own generic dispatcher does.
    if precision == "fp8" and backend._aiter_fp8_quant_gfx1201 is None:
        pytest.skip("Requires the gfx1201 FlyDSL FP8 quantizer")
    torch.manual_seed(42)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    q = torch.randn(1, 129, 2, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(1, 129 if causal else 77, 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    scale = 0.7 * head_dim**-0.5
    impl = backend.AITerImpl(2, head_dim, scale, causal=causal)

    def unexpected_generic(*args, **kwargs):
        pytest.fail("Supported gfx1201 input bypassed FlyDSL")

    monkeypatch.setattr(backend.aiter, "flash_attn_func", unexpected_generic)
    monkeypatch.setattr(backend, "_use_fp8_attn", precision == "fp8")
    out = impl.forward(q, k, v)
    assert out.dtype == dtype and out.shape == q.shape
    reference = _reference(q, k, v, scale, causal)
    if precision == "fp8":
        # Integration quality is checked against the original BF16 inputs.
        # The consumer's probability-rounding oracle belongs to AITER.
        torch.testing.assert_close(out.float(), reference, atol=0.2, rtol=0.2)
        cosine = F.cosine_similarity(out.float(), reference, dim=-1)
        assert cosine.min().item() > 0.97
        assert cosine.mean().item() > 0.994
    else:
        torch.testing.assert_close(out.float(), reference, atol=0.008, rtol=0.02)


@pytest.mark.parametrize(
    "case", ["d32", "gqa", "causal_cross", "missing_api", "zero_scale"]
)
@torch.no_grad()
def test_generic_aiter_remains_available(backend, monkeypatch, case, fp8):
    torch.manual_seed(42)
    head_dim = 32 if case == "d32" else 128
    dtype = torch.bfloat16
    kv_heads = 2 if case == "gqa" else 4
    causal = case == "causal_cross"
    q = torch.randn(1, 33, 4, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(
        1, 37 if causal else 33, kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v = torch.randn_like(k)
    scale = 0.0 if case == "zero_scale" else 0.7 * head_dim**-0.5
    impl = backend.AITerImpl(4, head_dim, scale, causal=causal, num_kv_heads=kv_heads)

    def unsupported_launch(*args, **kwargs):
        pytest.fail("Generic AITER input reached the gfx1201 FlyDSL path")

    monkeypatch.setattr(
        backend,
        "_aiter_flash_attn_gfx1201",
        None if case == "missing_api" else unsupported_launch,
    )
    out = impl.forward(q, k, v)
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, scale, causal), atol=0.008, rtol=0.02
    )


@torch.no_grad()
def test_bf16_lds_limit_delegates_to_generic_aiter(backend, monkeypatch, fp8):
    # D512 FP16 is outside the gfx1201 FlyDSL LDS limit and cannot use the
    # BF16-only FP8 producer. Generic AITER owns whether it can serve this case.
    torch.manual_seed(42)
    q, k, v = (
        torch.randn(1, 33, 4, 512, device="cuda", dtype=torch.float16) for _ in range(3)
    )
    scale = 0.07
    impl = backend.AITerImpl(4, 512, scale)

    def unsupported_launch(*args, **kwargs):
        pytest.fail("D512 FP16 input reached gfx1201 FlyDSL")

    def generic_attention(q, k, v, **kwargs):
        # Replace only the vendor boundary; check dispatch and scale wiring
        # through an actual attention result, independent of Triton tile policy.
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=kwargs["softmax_scale"],
            dropout_p=kwargs["dropout_p"],
            is_causal=kwargs["causal"],
        ).transpose(1, 2)

    monkeypatch.setattr(backend, "_aiter_flash_attn_gfx1201", unsupported_launch)
    monkeypatch.setattr(backend.aiter, "flash_attn_func", generic_attention)
    out = impl.forward(q, k, v)
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, scale), atol=0.008, rtol=0.02
    )


def test_differentiable_inputs_remain_on_generic_aiter(backend, monkeypatch, fp8):
    q = torch.randn(
        1, 33, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k, v = torch.randn_like(q), torch.randn_like(q)
    impl = backend.AITerImpl(2, 128, 0.07)

    def unsupported_launch(*args, **kwargs):
        pytest.fail("Differentiable input reached the forward-only FlyDSL kernel")

    monkeypatch.setattr(backend, "_aiter_flash_attn_gfx1201", unsupported_launch)
    out = impl.forward(q, k, v)
    assert out.requires_grad
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, 0.07), atol=0.008, rtol=0.02
    )


@torch.no_grad()
def test_flydsl_runs_outside_dynamo(backend, monkeypatch, fp8):
    torch.manual_seed(42)
    q, k, v = (
        torch.randn(1, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    impl = backend.AITerImpl(2, 128, 0.07)
    reference = impl.forward(q, k, v)
    kernel = backend._aiter_flash_attn_gfx1201
    traced_calls = []

    def traced_kernel(*args, **kwargs):
        traced_calls.append(torch.compiler.is_compiling())
        return kernel(*args, **kwargs)

    monkeypatch.setattr(backend, "_aiter_flash_attn_gfx1201", traced_kernel)
    torch._dynamo.reset()
    try:
        out = torch.compile(impl.forward, backend="eager")(q, k, v)
        assert traced_calls == [False]
        torch.testing.assert_close(out, reference, atol=0, rtol=0)
    finally:
        torch._dynamo.reset()


@torch.no_grad()
def test_packed_aiter_preserves_segments(backend, monkeypatch):
    torch.manual_seed(42)
    q, k, v = (
        torch.randn(33, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    scale = 0.07
    impl = backend.AITerImpl(4, 128, scale)

    def unsupported_launch(*args, **kwargs):
        pytest.fail("Packed input reached the dense gfx1201 FlyDSL path")

    monkeypatch.setattr(backend, "_aiter_flash_attn_gfx1201", unsupported_launch)
    bounds = (0, 13, 33)
    out = impl.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=torch.tensor(bounds, device="cuda", dtype=torch.int32),
        max_seqlen=20,
        cu_seqlens_host=bounds,
    )
    reference = torch.cat(
        [
            _reference(
                q[a:b].unsqueeze(0), k[a:b].unsqueeze(0), v[a:b].unsqueeze(0), scale
            ).squeeze(0)
            for a, b in zip(bounds, bounds[1:])
        ]
    )
    torch.testing.assert_close(out.float(), reference, atol=0.008, rtol=0.02)


@torch.no_grad()
def test_fp8_without_quantizer_uses_bf16(backend, monkeypatch):
    torch.manual_seed(42)
    q, k, v = (
        torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    impl = backend.AITerImpl(4, 128, 0.07)
    monkeypatch.setattr(backend, "_use_fp8_attn", True)
    monkeypatch.setattr(backend, "_aiter_fp8_quant_gfx1201", None)
    out = impl.forward(q, k, v)
    torch.testing.assert_close(
        out.float(), _reference(q, k, v, 0.07), atol=0.008, rtol=0.02
    )


@pytest.mark.parametrize("causal", [False, True])
@torch.no_grad()
def test_packed_projection_views(backend, monkeypatch, fp8, causal):
    # FLUX.2 packs QKV together with its MLP projection: S stride is 9*H*D.
    # Exercise batch gaps and a partial tile without hiding them in the caller.
    torch.manual_seed(42)
    storage = torch.randn(2, 129, 18 * 128, device="cuda", dtype=torch.bfloat16)
    q, k, v = (
        storage[..., start : start + 256].unflatten(-1, (2, 128))
        for start in (0, 256, 512)
    )
    impl = backend.AITerImpl(2, 128, 0.07, causal=causal)
    expected = impl.forward(*(x.contiguous() for x in (q, k, v)))
    out = impl.forward(q, k, v)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    if not fp8:
        torch.testing.assert_close(
            out.float(), _reference(q, k, v, 0.07, causal=causal), atol=0.008, rtol=0.02
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
