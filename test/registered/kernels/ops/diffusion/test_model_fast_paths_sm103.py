"""SM103-only model fast paths split out of ``test_model_fast_paths.py``.

These run on the B300 lane; the other model wrappers there run on H100.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import try_flux2_token_cat_nvfp4
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-c", runner_config="8-gpu-b300")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="FLUX.2 token-cat NVFP4 requires Blackwell SM103",
)


def test_flux2_token_cat_nvfp4_matches_flashinfer():
    import flashinfer

    torch.manual_seed(20260830)
    attention = torch.randn(1, 17, 6144, device="cuda", dtype=torch.bfloat16)
    mlp = torch.randn(1, 17, 18432, device="cuda", dtype=torch.bfloat16)
    global_scale = torch.tensor(0.625, device="cuda", dtype=torch.float32)

    expected_fp4, expected_scales = flashinfer.fp4_quantize(
        torch.cat([attention, mlp], dim=-1).view(-1, 24576), global_scale
    )
    actual = try_flux2_token_cat_nvfp4(attention, mlp, global_scale)

    assert actual is not None
    actual_fp4, actual_scales = actual
    assert torch.equal(actual_fp4, expected_fp4)
    assert torch.equal(
        actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
