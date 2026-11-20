import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.layernorm import LayerNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Requires CUDA FlashInfer LayerNorm",
)
@torch.inference_mode()
def test_layernorm_strided_projection_slice(monkeypatch):
    """A fused projection's trailing gates must not enter the next key's norm."""
    flashinfer_norm = pytest.importorskip("flashinfer.norm")
    monkeypatch.setattr(flashinfer_norm, "_USE_CUDA_NORM", True)
    torch.manual_seed(0)
    projection = torch.randn(37, 160, device="cuda", dtype=torch.bfloat16)
    key = projection[:, :128]
    norm = LayerNorm(128).cuda()
    norm.weight.copy_(torch.randn_like(norm.weight))
    norm.bias.copy_(torch.randn_like(norm.bias))

    expected = F.layer_norm(
        key.float(),
        (128,),
        norm.weight,
        norm.bias,
        norm.variance_epsilon,
    ).to(key.dtype)
    torch.testing.assert_close(norm.forward_cuda(key), expected, rtol=2**-7, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
