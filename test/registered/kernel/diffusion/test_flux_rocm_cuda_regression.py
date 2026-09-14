# SPDX-License-Identifier: Apache-2.0
"""CUDA regressions for shared code touched by the ROCm FLUX integration."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required; a skip is not CUDA validation",
)


@torch.inference_mode()
def test_cuda_modulation_keeps_exact_backend():
    from sglang.kernels.ops.diffusion import (
        can_use_modulate_scale_shift_cuda,
        modulate_scale_shift_cuda,
    )

    torch.manual_seed(17)
    x = torch.randn(1, 512, 3072, device="cuda", dtype=torch.bfloat16)
    shift, scale = torch.randn(1, 6 * 3072, device="cuda", dtype=x.dtype).chunk(6, 1)[
        :2
    ]
    assert can_use_modulate_scale_shift_cuda(x, scale, shift)
    assert torch.equal(
        modulate_scale_shift_cuda(x, scale, shift),
        x * (1 + scale[:, None]) + shift[:, None],
    )


@torch.inference_mode()
def test_rocm_rollback_does_not_change_cuda_dispatch(monkeypatch):
    from sglang.kernels.ops.diffusion import BitExactFusionGate
    from sglang.multimodal_gen.runtime.models.dits import flux

    torch.manual_seed(42)
    gate = BitExactFusionGate("CUDA regression", per_signature=True)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD", gate)
    monkeypatch.setattr(flux, "_FLUX_LN_MOD_SIGS", gate.verified_sigs)
    monkeypatch.setattr(flux, "_DISABLE_ROCM_LN_MODULATE", True)
    x = torch.randn(1, 512, 3072, device="cuda", dtype=torch.bfloat16)
    scale, shift = torch.randn(2, 1, 3072, device="cuda", dtype=x.dtype)
    norm = torch.nn.LayerNorm(3072, eps=1e-6, elementwise_affine=False).cuda()
    result = flux._flux_fused_ln_modulate(norm, x, scale, shift)
    assert result is not None
    assert torch.equal(result, flux.modulate_scale_shift(norm(x), scale, shift))
    assert gate.verified_sigs and not gate.disabled


@torch.inference_mode()
def test_cuda_swiglu_import_and_execution():
    from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul

    torch.manual_seed(42)
    x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
    layer = SiluAndMul()
    torch.testing.assert_close(
        layer.forward_cuda(x), layer.forward_native(x), atol=0.016, rtol=0.016
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
