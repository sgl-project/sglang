"""kimi_k3_tiny_gemm on ROCm: the tiny kernel up to 8 tokens, a16w16_linear above."""

from unittest import mock

import pytest
import torch

import sglang.kernels.ops.gemm.kimi_k3 as k3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")


@pytest.mark.parametrize("n,k", [(144, 7168), (1536, 128)])
@pytest.mark.parametrize("m,want", [(1, "tiny"), (8, "tiny"), (9, "a16w16")])
def test_rocm_dispatch(n, k, m, want):
    x = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(n, k, dtype=torch.bfloat16)
    calls = []

    def tiny(x, w, out=None, *, out_dtype=None):
        calls.append("tiny")
        return x @ w.T

    def a16w16(x, w):
        calls.append("a16w16")
        return x @ w.T

    with (
        mock.patch.object(k3, "_is_hip", True),
        mock.patch("sglang.kernels.ops.gemm.tiny_gemm.tiny_gemm_bf16", tiny),
        mock.patch("sglang.srt.layers.quantization.unquant_hip.a16w16_linear", a16w16),
    ):
        out = k3.kimi_k3_tiny_gemm(x, w)
    assert calls == [want]
    assert out.shape == (m, n)
