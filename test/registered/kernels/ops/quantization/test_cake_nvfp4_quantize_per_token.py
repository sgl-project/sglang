"""Cake NVFP4 per-token activation quantization through sglang.kernels.

Checks for the Cake adapter distributed by FlashInfer: the registry resolves
the explicit FlashInfer backend; the facade result is bitwise identical to
calling FlashInfer's public ``nvfp4_quantize(backend="cake",
per_token_activation=True)`` directly; the per-token scales match the FP32
recipe; and dequantizing the FP4 codes with the kernel's own block scales
reproduces the input within the FP4 block-scaled tolerance (atol 1.0,
rtol 0.1). Skips (with the reason) when the installed FlashInfer lacks the
Cake module, the GPU is outside sm_100a / sm_103a, or no generated program is
registered for this GPU.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import quantization as cake_quant
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.quantization.cake import cake_nvfp4_quantize_per_token
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "quantization.nvfp4_quantize_per_token"
GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.quantization:")


def _skip_unless_supported(x: torch.Tensor):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_quant.NVFP4_FI_MODULE, cake_quant.NVFP4_FI_BACKEND_MODULE
    ):
        pytest.skip("installed FlashInfer lacks the cake_nvfp4_per_token backend")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_quant.ARCHS:
        pytest.skip(
            f"Cake NVFP4 per-token quantizer is built for sm_100a/103a, device is {cc}"
        )
    if not cake_quant.supports_nvfp4_quantize_per_token(x):
        pytest.skip("no generated per-token NVFP4 program registered for this GPU")


def _unswizzle_sf_128x4(sf: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """Logical ``[m, k // 16]`` E4M3 scale bytes of the swizzled 128x4 tensor."""
    rows = (m + 127) // 128 * 128
    cols = (k // 16 + 3) // 4 * 4
    r = torch.arange(rows, device=sf.device)[:, None]
    c = torch.arange(cols, device=sf.device)[None, :]
    offsets = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * cols)
    )
    return sf.reshape(-1)[offsets][:m, : k // 16]


@pytest.mark.parametrize(
    "m,k,dtype",
    [
        (1, 7168, torch.bfloat16),
        (130, 7168, torch.bfloat16),
        (17, 16384, torch.bfloat16),
        (8, 7168, torch.float16),
    ],
)
def test_matches_flashinfer_and_reference(m, k, dtype):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type != "cuda":
        pytest.skip("CUDA required")
    g = torch.Generator(device=device).manual_seed(1000 + m + k)
    x = torch.randn(m, k, device=device, dtype=dtype, generator=g)
    _skip_unless_supported(x)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)

    fp4, sf, scale = cake_nvfp4_quantize_per_token(x, gs_inv)
    from flashinfer.quantization.fp4_quantization import SfLayout, nvfp4_quantize

    fp4_fi, sf_fi, scale_fi = nvfp4_quantize(
        x,
        gs_inv,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cake",
        per_token_activation=True,
    )
    torch.cuda.synchronize()
    assert fp4.dtype == torch.uint8 and tuple(fp4.shape) == (m, k // 2)
    assert sf.dtype == torch.uint8
    assert scale.dtype == torch.float32 and tuple(scale.shape) == (m,)
    assert torch.equal(fp4, fp4_fi)
    assert torch.equal(sf, sf_fi)
    assert torch.equal(scale, scale_fi)

    # Per-token scale recipe: amax(row) * global_scale_inv.
    xf = x.float()
    row_amax = xf.abs().amax(dim=1)
    scale_ref = row_amax * GLOBAL_SCALE_INV
    torch.testing.assert_close(scale, scale_ref, atol=1e-2, rtol=1e-2)

    # Dequantize with the kernel's block scales: x_hat = code * sf * token_scale.
    sf_dec = _unswizzle_sf_128x4(sf, m, k).view(torch.float8_e4m3fn).float()
    nib = torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).reshape(m, k).long()
    codes = torch.tensor(E2M1, device=device)[nib]
    x_hat = (codes.view(m, k // 16, 16) * sf_dec[:, :, None]).view(m, k) * scale[
        :, None
    ]
    torch.testing.assert_close(x_hat, xf, atol=1.0, rtol=0.1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
