# SPDX-License-Identifier: Apache-2.0
"""Weight-free ROCm guards for #34351; never launch CUDA PTX to test rejection."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="ROCm GPU required"
)


@pytest.mark.parametrize(
    "arch,accepted",
    [
        ("gfx90a:sramecc+:xnack-", True),
        ("gfx942", False),
        ("gfx950", False),
        ("gfx90a_future", False),
        ("", False),
    ],
)
def test_architecture_allowlist(monkeypatch, arch, accepted):
    backend = importlib.import_module(
        "sglang.kernels.ops.diffusion.norm.layernorm_modulate_rocm_triton"
    )
    backend._is_validated_rocm_device.cache_clear()
    try:
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda index: SimpleNamespace(gcnArchName=arch),
        )
        assert backend._is_validated_rocm_device(0) is accepted
    finally:
        backend._is_validated_rocm_device.cache_clear()


@torch.inference_mode()
def test_all_ptx_norm_guard_families_reject_rocm():
    from sglang.kernels.ops import diffusion

    # The LN/QK family contains _rcp4, div_rn_f32 and cuda_rsqrtf;
    # the RMS family contains mul_rn_f32 and rsqrt_approx_f32.
    # Check public predicates before any JIT call: LLVM failures may abort,
    # so catching an exception from an intentionally launched PTX kernel is unsafe.
    x = torch.randn(1, 17, 3072, device="cuda", dtype=torch.bfloat16)
    row = torch.zeros(1, 3072, device="cuda", dtype=x.dtype)
    assert not diffusion.can_use_fused_layernorm_modulate(x, row, row)
    q = torch.randn(1, 17, 24, 128, device="cuda", dtype=x.dtype)
    assert not diffusion.can_use_fused_qk_head_layernorm(q, q)
    rms = importlib.import_module(
        "sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_bitexact"
    )
    x = x[..., :2048].contiguous()
    weight = torch.ones(2048, device="cuda", dtype=x.dtype)
    mod = torch.zeros(1, 1, 2048, device="cuda", dtype=x.dtype)
    assert not rms.can_use_fused_rmsnorm_scale_shift(x, weight, mod, mod)
    assert not rms.can_use_fused_scale_residual_rmsnorm_scale_shift(
        x, x, mod, weight, mod, mod
    )


@torch.inference_mode()
def test_unvalidated_architecture_cannot_bypass_verified_signature(monkeypatch):
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.runtime.models.dits import flux

    backend = importlib.import_module(
        "sglang.kernels.ops.diffusion.norm.layernorm_modulate_rocm_triton"
    )
    monkeypatch.setattr(backend, "_is_validated_rocm_device", lambda index: False)
    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", False)
    x = torch.randn(1, 17, 3072, device="cuda", dtype=torch.bfloat16)
    row = torch.zeros(1, 3072, device="cuda", dtype=x.dtype)
    norm = torch.nn.LayerNorm(3072, eps=1e-6, elementwise_affine=False).cuda()
    gate = diffusion.BitExactFusionGate("unvalidated architecture", per_signature=True)
    sig = (
        x.device,
        x.dtype,
        x.shape,
        x.stride(),
        row.shape,
        row.stride(),
        row.shape,
        row.stride(),
        norm.eps,
    )
    gate.mark_verified(sig)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported architecture reached a fused backend")

    monkeypatch.setattr(diffusion, "layernorm_modulate_rocm", forbidden)
    monkeypatch.setattr(flux, "fused_layernorm_modulate", forbidden)
    assert not backend.can_use_layernorm_modulate_rocm(x, row, row)
    assert flux._flux_fused_ln_modulate(norm, x, row, row) is None
    out = flux._flux_norm_modulate(torch.nn.Module(), norm, x, row, row)
    assert torch.equal(out, norm(x) * (1 + row[:, None]) + row[:, None])
    assert not gate.disabled


@torch.inference_mode()
def test_eager_verified_signature_stays_out_of_compiled_model(monkeypatch):
    from sglang.kernels.ops import diffusion
    from sglang.multimodal_gen.runtime.models.dits import flux

    x = torch.randn(1, 17, 3072, device="cuda", dtype=torch.bfloat16)
    row = torch.zeros(1, 3072, device="cuda", dtype=x.dtype)
    norm = torch.nn.LayerNorm(3072, eps=1e-6, elementwise_affine=False).cuda()
    gate = diffusion.BitExactFusionGate("compiled model", per_signature=True)
    gate.mark_verified(
        (
            x.device,
            x.dtype,
            x.shape,
            x.stride(),
            row.shape,
            row.stride(),
            row.shape,
            row.stride(),
            norm.eps,
        )
    )
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)
    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def forbidden(*args, **kwargs):
        pytest.fail("eager-only backend reached compiled model")

    monkeypatch.setattr(diffusion, "layernorm_modulate_rocm", forbidden)
    assert flux._flux_fused_ln_modulate(norm, x, row, row) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
