# SPDX-License-Identifier: Apache-2.0
"""ROCm LN/modulation: rounding, strided modulation, graphs and fallback."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_layernorm_modulate_rocm,
    layernorm_modulate_rocm,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(
    not torch.version.hip
    or not torch.cuda.is_available()
    or getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":", 1)[0]
    != "gfx90a",
    reason="validated gfx90a ROCm GPU required",
)


def reference(x, scale, shift, eps):
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return normalized * (1 + scale.reshape(x.shape[0], 1, -1)) + shift.reshape(
        x.shape[0], 1, -1
    )


@pytest.mark.parametrize("width", [3072])
@pytest.mark.parametrize("batch,seq", [(1, 1), (1, 512), (2, 17), (2, 4096)])
@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-3])
@pytest.mark.parametrize("kind", ["random", "constant", "near_constant"])
@pytest.mark.parametrize("variance_fma", [False, True])
@torch.inference_mode()
def test_reference(width, batch, seq, eps, kind, variance_fma):
    torch.manual_seed(42)
    x = torch.randn(batch, seq, width, device="cuda", dtype=torch.bfloat16)
    if kind == "constant":
        x.fill_(3)
    elif kind == "near_constant":
        x = (x.float() * 0.01 + 3).bfloat16()
    # FLUX modulation is chunked from a projection; its batch stride is 6D.
    projection = torch.randn(batch, width * 6, device="cuda", dtype=x.dtype)
    scale, shift = projection[:, :width], projection[:, width : 2 * width]
    assert can_use_layernorm_modulate_rocm(x, scale, shift)
    actual = layernorm_modulate_rocm(x, scale, shift, eps, variance_fma=variance_fma)
    expected = reference(x, scale, shift, eps)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    # Broad standalone error bound; the model-dispatch tests require equality.
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    if kind == "constant":
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_guard_and_eps():
    x = torch.randn(2, 17, 3072, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(2, 3072, device="cuda", dtype=x.dtype)
    assert can_use_layernorm_modulate_rocm(x, s[:, None], s[:, None])
    assert not can_use_layernorm_modulate_rocm(x.float(), s, s)
    assert not can_use_layernorm_modulate_rocm(x.transpose(0, 1), s, s)
    assert not can_use_layernorm_modulate_rocm(x[..., :2048], s, s)
    wide = torch.zeros(1, 1, 4096, device="cuda", dtype=x.dtype)
    modulation = torch.zeros(1, 4096, device="cuda", dtype=x.dtype)
    assert not can_use_layernorm_modulate_rocm(wide, modulation, modulation)
    assert not can_use_layernorm_modulate_rocm(x, s[:1], s)
    assert not can_use_layernorm_modulate_rocm(x[:, :0], s, s)
    assert not can_use_layernorm_modulate_rocm(x, s.cpu(), s)
    with pytest.raises(ValueError, match="unsupported input"):
        layernorm_modulate_rocm(x.float(), s, s, 1e-6)
    for eps in (0.0, -1e-6, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="eps"):
            layernorm_modulate_rocm(x, s, s, eps)


@pytest.mark.parametrize("variance_fma", [False, True])
@torch.inference_mode()
def test_compile_graph_and_stream(variance_fma):
    x = torch.randn(2, 17, 3072, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(2, 3072, device="cuda", dtype=x.dtype)
    fn = lambda a, b: layernorm_modulate_rocm(a, b, b, 1e-6, variance_fma=variance_fma)
    eager = fn(x, s)
    compiled = torch.compile(fn, fullgraph=True)
    torch.testing.assert_close(compiled(x, s), eager, atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(x, s)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn(x, s)
    x.add_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, fn(x, s), atol=0, rtol=0)


@torch.inference_mode()
def test_flux_exactness_dispatch(monkeypatch):
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.runtime.models.dits import flux

    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", False)

    x = torch.randn(2, 17, 3072, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(2, 3072, device="cuda", dtype=x.dtype)
    site = torch.nn.Module()
    norm = torch.nn.LayerNorm(3072, eps=1e-6, elementwise_affine=False).cuda()
    flux.mark_fused_ln_modulate_site(site)
    gate = diffusion.BitExactFusionGate("test ROCm LN", per_signature=True)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)
    calls = []
    real_kernel = diffusion.layernorm_modulate_rocm

    def tracked(*args, **kwargs):
        calls.append(True)
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(diffusion, "layernorm_modulate_rocm", tracked)
    baseline = reference(x, s, s, norm.eps)
    # CUDA tensor predicates are also true on ROCm; its CUDA-only modulation
    # backend must not silently replace the eager BF16 rounding contract.
    torch.testing.assert_close(
        diffusion.modulate_scale_shift(norm(x), s, s), baseline, atol=0, rtol=0
    )
    fused = flux._flux_norm_modulate(site, norm, x, s, s)
    first_calls = len(calls)
    assert first_calls in (1, 2)
    assert not gate.disabled and gate.verified_sigs, "exactness test silently fell back"
    torch.testing.assert_close(fused, baseline, atol=0, rtol=0)
    restored = flux._flux_norm_modulate(site, norm, x, s, s)
    assert len(calls) == first_calls + (0 if gate.disabled else 1)
    torch.testing.assert_close(restored, baseline, atol=0, rtol=0)

    # A backend mismatch returns the reference and disables further attempts.
    gate = diffusion.BitExactFusionGate("test ROCm mismatch", per_signature=True)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)
    monkeypatch.setattr(
        diffusion,
        "layernorm_modulate_rocm",
        lambda *args, **kwargs: real_kernel(*args, **kwargs) + 0.25,
    )
    fallback = flux._flux_norm_modulate(site, norm, x, s, s)
    assert gate.disabled
    torch.testing.assert_close(fallback, baseline, atol=0, rtol=0)
    monkeypatch.setattr(diffusion, "layernorm_modulate_rocm", tracked)
    count = len(calls)
    flux._flux_norm_modulate(site, norm, x, s, s)
    assert len(calls) == count

    # The operational kill switch bypasses even a previously verified gate.
    gate = diffusion.BitExactFusionGate("rollback ROCm LN", per_signature=True)
    gate.mark_verified(
        (
            x.device,
            x.dtype,
            x.shape,
            x.stride(),
            s.shape,
            s.stride(),
            s.shape,
            s.stride(),
            norm.eps,
        )
    )
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)
    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", True)
    assert flux._flux_fused_ln_modulate(norm, x, s, s) is None
    assert len(calls) == count and not gate.disabled


@torch.inference_mode()
def test_independent_modulation_strides():
    x = torch.randn(2, 17, 3072, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 3 * 3072, device="cuda", dtype=x.dtype)[:, :3072]
    shift = torch.randn(2, 5 * 3072, device="cuda", dtype=x.dtype)[:, 3072:6144]
    actual = layernorm_modulate_rocm(x, scale[:, None], shift, 1e-5)
    torch.testing.assert_close(
        actual, reference(x, scale, shift, 1e-5), atol=0.016, rtol=0.016
    )


@torch.inference_mode()
def test_flux_cold_capture_and_exception(monkeypatch):
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.runtime.models.dits import flux

    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", False)
    gate = diffusion.BitExactFusionGate("cold ROCm LN", per_signature=True)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)
    x = torch.randn(1, 17, 3072, device="cuda", dtype=torch.bfloat16)
    s = torch.zeros(1, 3072, device="cuda", dtype=x.dtype)
    norm = torch.nn.LayerNorm(3072, eps=1e-6, elementwise_affine=False).cuda()
    calls = []

    def failure(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected backend failure")

    monkeypatch.setattr(diffusion, "layernorm_modulate_rocm", failure)
    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert flux._flux_fused_ln_modulate(norm, x, s, s) is None
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        assert flux._flux_fused_ln_modulate(norm, x, s, s) is None
    assert not calls and not gate.disabled
    assert flux._flux_fused_ln_modulate(norm, x, s, s) is None
    assert len(calls) == 1 and gate.disabled
    assert flux._flux_fused_ln_modulate(norm, x, s, s) is None
    assert len(calls) == 1


@pytest.mark.parametrize("offset", [1, 2, 3])
@torch.inference_mode()
def test_misaligned_input_cannot_reuse_verified_signature(monkeypatch, offset):
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.runtime.models.dits import flux

    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", False)
    x = torch.randn(17 * 3072 + offset, device="cuda", dtype=torch.bfloat16)
    x = x[offset:].view(1, 17, 3072)
    s = torch.zeros(1, 3072, device="cuda", dtype=x.dtype)
    norm = torch.nn.LayerNorm(3072, eps=1e-6, elementwise_affine=False).cuda()
    assert x.is_contiguous() and x.data_ptr() % 8 != 0
    gate = diffusion.BitExactFusionGate("misaligned ROCm LN", per_signature=True)
    gate.mark_verified(
        (
            x.device,
            x.dtype,
            x.shape,
            x.stride(),
            s.shape,
            s.stride(),
            s.shape,
            s.stride(),
            norm.eps,
        )
    )
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)

    def forbidden(*args, **kwargs):
        pytest.fail("misaligned input reached the vectorized backend")

    monkeypatch.setattr(diffusion, "layernorm_modulate_rocm", forbidden)
    assert not can_use_layernorm_modulate_rocm(x, s, s)
    with pytest.raises(ValueError, match="unsupported input"):
        layernorm_modulate_rocm(x, s, s, norm.eps)
    actual = flux._flux_norm_modulate(torch.nn.Module(), norm, x, s, s)
    assert torch.equal(actual, reference(x, s, s, norm.eps))
    assert not gate.disabled


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-3])
@torch.inference_mode()
def test_verified_signature_with_changing_inputs(monkeypatch, eps):
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.runtime.models.dits import flux

    gate = diffusion.BitExactFusionGate("changing inputs", per_signature=True)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)
    monkeypatch.setattr(flux, "_FLUX_ROCM_VARIANCE_FMA", {})
    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", False)
    norm = torch.nn.LayerNorm(3072, eps=eps, elementwise_affine=False).cuda()
    calls = []
    kernel = diffusion.layernorm_modulate_rocm

    def tracked(*args, **kwargs):
        calls.append(True)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(diffusion, "layernorm_modulate_rocm", tracked)
    for seed in range(32):
        for spread, center in ((1, 0), (0.01, 3), (10, -100), (1000, 0)):
            torch.manual_seed(seed)
            x = (torch.randn(2, 17, 3072, device="cuda") * spread + center).bfloat16()
            scale, shift = torch.randn(2, 6 * 3072, device="cuda", dtype=x.dtype).chunk(
                6, dim=1
            )[:2]
            previous_calls = len(calls)
            was_verified = bool(gate.verified_sigs)
            actual = flux._flux_fused_ln_modulate(norm, x, scale, shift)
            assert actual is not None and not gate.disabled and gate.verified_sigs
            assert len(calls) - previous_calls in ((1,) if was_verified else (1, 2))
            assert torch.equal(actual, reference(x, scale, shift, eps))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
