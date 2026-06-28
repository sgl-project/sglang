import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.layernorm import RMSNormNoWeight
from sglang.multimodal_gen.runtime.models.dits import ltx_2 as ltx2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    ltx2._LTX2_FUSED_RMS_ADALN_RUNTIME_DISABLED = False


def _reference(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float):
    normed = F.rms_norm(x, (x.shape[-1],), eps=eps)
    return normed * (1 + scale) + shift


@torch.no_grad()
@pytest.mark.parametrize(
    ("batch", "seq_len", "hidden_size", "param_seq"),
    [
        (1, 529, 4096, 529),
        (1, 161, 4096, 1),
        (2, 5, 2048, 1),
    ],
)
def test_ltx2_try_fused_rms_adaln_broadcast_scale(
    batch, seq_len, hidden_size, param_seq
):
    eps = 1e-6
    x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(
        batch, param_seq, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    shift = torch.randn(
        batch, param_seq, hidden_size, device="cuda", dtype=torch.bfloat16
    )

    actual = ltx2._ltx2_try_fused_rms_adaln(x, scale, shift, eps)
    assert actual is not None

    expected = _reference(x, scale, shift, eps)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.05)


@torch.no_grad()
def test_ltx2_rms_adaln_fallback_for_unsupported_hidden_size():
    eps = 1e-6
    x = torch.randn(1, 3, 384, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 384, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(1, 1, 384, device="cuda", dtype=torch.bfloat16)

    assert ltx2._ltx2_try_fused_rms_adaln(x, scale, shift, eps) is None

    rms_norm = RMSNormNoWeight()
    actual = ltx2._ltx2_rms_adaln(rms_norm, x, scale, shift, eps)
    expected = _reference(x, scale, shift, eps)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.05)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
