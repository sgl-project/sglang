"""Cake Sage-FP8 Q/K/V quantization through sglang.kernels.

Checks for the Cake adapter distributed by FlashInfer: the registry resolves
the explicit FlashInfer backend; the facade result is bitwise identical to
calling FlashInfer's ``sage_fp8_quantize_sm100`` directly; the per-token /
per-16-token / per-channel scales are bit-exact with the torch recipe
``amax.clamp_min(1e-12) / 448``; and dequantizing the e4m3 outputs with those
scales reproduces the BF16 inputs within the FP8 tolerance (atol 0.1,
rtol 0.1). Skips (with the reason) when the installed FlashInfer lacks the
Cake module or the GPU is outside sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import quantization as cake_quant
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.quantization.cake import cake_sage_fp8_quantize
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "quantization.sage_fp8_quantize"
HEAD_DIM = 128
K_GROUP = 16
FP8_MAX = 448.0


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.quantization:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_quant.SAGE_FI_MODULE, cake_quant.SAGE_FI_JIT_MODULE
    ):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake_quant.ARCHS:
        pytest.skip(
            f"Cake Sage-FP8 quantizer is built for sm_100a/103a, device is {cc}"
        )


def _scale(amax: torch.Tensor) -> torch.Tensor:
    return amax.clamp_min(1e-12) / FP8_MAX


def _reference_scales(q, k, v):
    qf, kf, vf = q.float(), k.float(), v.float()
    batch, seqlen_k, kv_heads, _ = kf.shape
    groups = (seqlen_k + K_GROUP - 1) // K_GROUP
    pad = groups * K_GROUP - seqlen_k
    kf_pad = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, pad))  # pad tokens with 0
    q_scale = _scale(qf.abs().amax(dim=-1)).permute(0, 2, 1)  # [B, H, Sq]
    k_scale = _scale(
        kf_pad.view(batch, groups, K_GROUP, kv_heads, HEAD_DIM).abs().amax(dim=(2, 4))
    ).permute(0, 2, 1)  # [B, Hkv, groups]
    v_scale = _scale(vf.abs().amax(dim=1))  # [B, Hkv, 128]
    return q_scale.contiguous(), k_scale.contiguous(), v_scale.contiguous()


@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(64, 64), (96, 200)])
def test_matches_flashinfer_and_reference(heads, kv_heads, seqlen_q, seqlen_k):
    _skip_unless_supported()
    device = torch.device("cuda")
    batch = 2
    g = torch.Generator(device=device).manual_seed(300 + heads + seqlen_q + seqlen_k)
    q = torch.randn(
        batch,
        seqlen_q,
        heads,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    )
    k = torch.randn(
        batch,
        seqlen_k,
        kv_heads,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    )
    v = torch.randn(
        batch,
        seqlen_k,
        kv_heads,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    )
    assert cake_quant.supports_sage_fp8_quantize(q, k, v)

    q8, k8, v8, q_scale, k_scale, v_scale = cake_sage_fp8_quantize(q, k, v)
    from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import sage_fp8_quantize_sm100

    outs_fi = sage_fp8_quantize_sm100(q, k, v)
    torch.cuda.synchronize()
    for got, ref in zip((q8, k8, v8, q_scale, k_scale, v_scale), outs_fi):
        assert torch.equal(got, ref)

    groups = (seqlen_k + K_GROUP - 1) // K_GROUP
    assert q8.dtype == k8.dtype == v8.dtype == torch.float8_e4m3fn
    assert tuple(q_scale.shape) == (batch, heads, seqlen_q)
    assert tuple(k_scale.shape) == (batch, kv_heads, groups)
    assert tuple(v_scale.shape) == (batch, kv_heads, HEAD_DIM)

    q_scale_ref, k_scale_ref, v_scale_ref = _reference_scales(q, k, v)
    assert torch.equal(q_scale, q_scale_ref)
    assert torch.equal(k_scale, k_scale_ref)
    assert torch.equal(v_scale, v_scale_ref)

    # Dequantize with the kernel scales and compare with the BF16 inputs.
    q_hat = q8.float() * q_scale.permute(0, 2, 1)[:, :, :, None]
    torch.testing.assert_close(q_hat, q.float(), atol=0.1, rtol=0.1)
    token_group = torch.arange(seqlen_k, device=device) // K_GROUP
    k_tok_scale = k_scale[:, :, token_group].permute(0, 2, 1)  # [B, Sk, Hkv]
    k_hat = k8.float() * k_tok_scale[:, :, :, None]
    torch.testing.assert_close(k_hat, k.float(), atol=0.1, rtol=0.1)
    v_hat = v8.float() * v_scale[:, None, :, :]
    torch.testing.assert_close(v_hat, v.float(), atol=0.1, rtol=0.1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
