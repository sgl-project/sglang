import torch
import torch_npu  # noqa: F401

from sglang.srt.layers import attn_residual
from sglang.srt.models.kimi_k3 import _add3
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")


def test_attn_residual_uses_portable_fallback_without_cuda_capability():
    attn_residual._FAST_SUPPORTED = None
    assert attn_residual._use_fast(7168) is False


def test_kimi_k3_add3_uses_portable_fallback_for_non_cuda_tensors():
    a = torch.randn(2, 16, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.randn_like(a)

    torch.testing.assert_close(_add3(a, b, c), (a + b) + c)
