import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the MXFP8 reference quantizer is Blackwell-only",
)

HIDDEN = 5120
STREAMS = 4


@pytest.mark.parametrize("m", [1, 2, 5, 6, 8])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_bitwise_identical_to_norm_then_quantize(m: int, scale: float, backend: str):
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    from sglang.kernels.ops.layernorm.mxfp8_epilogue import hc_combine_norm_mxfp8
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    g = torch.Generator(device="cuda").manual_seed(m * 31 + int(scale * 1000))
    x = (
        torch.randn(
            (m, STREAMS * HIDDEN), device="cuda", dtype=torch.bfloat16, generator=g
        )
        * scale
    )
    pre = torch.randn(
        (m, STREAMS), device="cuda", dtype=torch.bfloat16, generator=g
    ).contiguous()
    w = torch.randn((HIDDEN,), device="cuda", dtype=torch.bfloat16, generator=g)
    eps = 1e-6

    y_ref = hc_combine_norm(x, pre, w, eps)
    q_ref, sf_ref = flashinfer_mxfp8_quantize(y_ref, True, 32, backend)
    y, q, sf = hc_combine_norm_mxfp8(x, pre, w, eps)

    assert torch.equal(y, y_ref)
    assert torch.equal(
        q.reshape(-1).view(torch.uint8), q_ref.reshape(-1).view(torch.uint8)
    )
    assert sf.shape == sf_ref.reshape(-1).shape
    assert torch.equal(sf, sf_ref.reshape(-1))


@pytest.mark.parametrize("m", [6, 4096])
def test_model_dispatch_preserves_prefill_and_small_row_paths(m):
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    torch.manual_seed(42)
    x = torch.randn((m, STREAMS, HIDDEN), device="cuda", dtype=torch.bfloat16)
    pre = torch.randn((m, STREAMS), device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(HIDDEN, eps=1e-6).cuda().bfloat16()
    fn = torch.empty((24, STREAMS * HIDDEN), device="cuda")
    layer = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v41"),
        hc_mult=STREAMS,
        hc_attn_fn=fn,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    quantized = []
    # Isolate the combine/quantize integration from coefficient computation.
    with patch(
        "sglang.kernels.ops.layernorm.mhc.hc_mix_stats_sinkhorn",
        return_value=(pre, pre, pre),
    ):
        y, *_ = DeepseekV4DecoderLayer._hc_mix_and_combine(
            layer, x, fn, None, None, pre, norm, quantized=quantized
        )
    expected = hc_combine_norm(x.flatten(1), pre, norm.weight, 1e-6)
    assert torch.equal(y, expected)
    if m == 6:
        assert len(quantized) == 1
        q, sf = flashinfer_mxfp8_quantize(expected, True, 32, "cuda")
        actual_q, actual_sf = quantized[0]
        assert torch.equal(actual_q.view(torch.uint8), q.view(torch.uint8))
        assert torch.equal(actual_sf.reshape(-1), sf.reshape(-1))
    else:
        assert quantized == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
