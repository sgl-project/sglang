import sys

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    dequant_group_fp8_to_bf16,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    triton_w8a8_block_fp8_linear,
)
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not is_sm90_supported(), reason="Hopper group32 path")
@pytest.mark.parametrize(
    "m,k", [(0, 288), (1, 1024), (63, 288), (64, 288), (65, 1024), (257, 5120)]
)
def test_cached_weight_linear(m, k):
    torch.manual_seed(103 + m)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(576, k, device="cuda") * 32).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-12, -5, (18, k // 32), device="cuda").float())
    cached = block_quant_dequant(w, scale, [32, 32], torch.bfloat16)
    bias = torch.randn(576, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        weight=w,
        block_size=[32, 32],
        weight_scale=scale,
        act_scale_ue8m0=True,
        bias=bias,
    )
    if not m:
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        sf = torch.empty(0, k // 32, device="cuda")
        assert dequant_group_fp8_to_bf16(q, sf).shape == x.shape
        return
    q, sf = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
    ref_x = q.double() * sf.double().repeat_interleave(32, 1)
    torch.testing.assert_close(
        dequant_group_fp8_to_bf16(q, sf).double(), ref_x, rtol=0, atol=0
    )
    actual = triton_w8a8_block_fp8_linear(x, weight_bf16=cached, **kwargs)
    if m < 64:
        torch.testing.assert_close(
            actual, triton_w8a8_block_fp8_linear(x, **kwargs), rtol=0, atol=0
        )
    else:
        ref = ref_x @ cached.double().T
        raw = triton_w8a8_block_fp8_linear(
            x, weight_bf16=cached, **{**kwargs, "bias": None}
        )
        # Check GEMM rounding before a potentially cancelling bias addition.
        torch.testing.assert_close(
            raw.double(),
            ref,
            rtol=1 / 256,
            atol=ref.square().mean().sqrt().item() * 1e-5,
        )
        torch.testing.assert_close(actual, raw + bias, rtol=0, atol=0)
        # Capture the complete quantize/dequantize/GEMM path, then change inputs.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = triton_w8a8_block_fp8_linear(x, weight_bf16=cached, **kwargs)
        x.neg_()
        graph.replay()
        torch.testing.assert_close(
            replay,
            triton_w8a8_block_fp8_linear(x, weight_bf16=cached, **kwargs),
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not is_sm90_supported(), reason="Hopper group32 path")
def test_cached_weight_refresh():
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    method = Fp8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[32, 32],
            scale_fmt="ue8m0",
        )
    )
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(
        torch.ones(64, 288, device="cuda").to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.ones(2, 9, device="cuda"), requires_grad=False
    )
    for scale in (1, 2):
        layer.weight_scale_inv.data.fill_(scale)
        method.process_weights_after_loading_block_quant(layer)
        torch.testing.assert_close(
            layer._block_fp8_bf16_weight,
            torch.full((64, 288), scale, device="cuda", dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        assert "_block_fp8_bf16_weight" not in layer.state_dict()
        assert "Hopper FP8" in _unsupported_derived_weight_cache_error(layer)
    with envs.SGLANG_OPT_HOPPER_BLOCK_FP8_BF16.override(False):
        method.process_weights_after_loading_block_quant(layer)
        assert layer._block_fp8_bf16_weight is None
        assert layer._derived_weight_cache_error is None
    for scale in (2.0**-118, 2.0**120, 1.5, float("nan")):
        layer.weight_scale_inv.data.fill_(scale)
        method.process_weights_after_loading_block_quant(layer)
        assert layer._block_fp8_bf16_weight is None
        assert layer._derived_weight_cache_error is None
    for scale in (2.0**-117, 2.0**119):
        layer.weight_scale_inv.data.fill_(scale)
        method.process_weights_after_loading_block_quant(layer)
        assert layer._block_fp8_bf16_weight is not None
    layer.weight_scale_inv.data.fill_(1)
    layer.keep_plain_weight_layout = True
    method.process_weights_after_loading_block_quant(layer)
    assert layer._block_fp8_bf16_weight is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
