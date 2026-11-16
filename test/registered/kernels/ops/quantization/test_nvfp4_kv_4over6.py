"""NVFP4 KV cache: 4-over-6 block-scale search."""

import pytest
import torch

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)

pytestmark = pytest.mark.skipif(
    not is_sm100_supported(), reason="NVFP4 KV cache requires SM100"
)

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize(packed: torch.Tensor, scales: torch.Tensor, global_scale: float):
    lut = torch.tensor(_E2M1, device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(
        packed.shape[0], -1
    )
    codes = codes.long()
    values = torch.where((codes & 8) != 0, -lut[codes & 7], lut[codes & 7])
    block_scales = scales.view(torch.float8_e4m3fn).float()
    return values * block_scales.repeat_interleave(16, dim=1) * global_scale


@torch.inference_mode()
@pytest.mark.parametrize("global_scale", [1.0, 0.05])
def test_default_search_matches_flashinfer(global_scale):
    """Without the search, the Triton kernel reproduces nvfp4_kv_quantize."""
    from flashinfer import nvfp4_kv_quantize

    from sglang.kernels.ops.quantization.nvfp4_kv_4over6 import (
        nvfp4_kv_quantize_triton,
    )

    torch.manual_seed(0)
    x = torch.randn(1024, 128, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([global_scale], device="cuda")
    ref_data, ref_scales = nvfp4_kv_quantize(x, scale)
    data, scales = nvfp4_kv_quantize_triton(x, scale, four_over_six=False)
    assert torch.equal(scales, ref_scales.view(torch.uint8))
    assert torch.equal(data, ref_data.view(torch.uint8))


@torch.inference_mode()
def test_4over6_never_increases_block_error():
    """Each block keeps the lower-error candidate, so no block gets worse."""
    from sglang.kernels.ops.quantization.nvfp4_kv_4over6 import (
        nvfp4_kv_quantize_triton,
    )

    torch.manual_seed(1)
    x = torch.randn(1024, 128, device="cuda") * torch.rand(1024, 1, device="cuda") * 30
    x = x.to(torch.bfloat16)
    scale = torch.tensor([1.0], device="cuda")
    ref = x.float()
    err6 = (_dequantize(*nvfp4_kv_quantize_triton(x, scale, False), 1.0) - ref) ** 2
    err4 = (_dequantize(*nvfp4_kv_quantize_triton(x, scale, True), 1.0) - ref) ** 2
    err6, err4 = err6.reshape(-1, 16).sum(1), err4.reshape(-1, 16).sum(1)
    assert bool((err4 <= err6 * (1 + 1e-6) + 1e-12).all())
    assert err4.mean() < 0.9 * err6.mean()
