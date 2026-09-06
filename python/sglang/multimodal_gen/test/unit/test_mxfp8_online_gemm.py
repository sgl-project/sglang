# SPDX-License-Identifier: Apache-2.0
"""Online MXFP8 GEMM path (``MXFP8Config()`` on a bf16 checkpoint): load-time
block quant, the prequantized (e4m3, swizzled scales) input from the fused
SwiGLU kernel, and the per-layer fallback to the per-channel fp8 path."""

import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _init_parallel() -> None:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _layer(in_f: int, out_f: int, bias: bool):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config

    return RowParallelLinear(
        in_f,
        out_f,
        bias=bias,
        params_dtype=torch.bfloat16,
        quant_config=MXFP8Config(),
        prefix="mlp.fc2",
    ).to("cuda")


def test_mxfp8_linear_matches_bf16_and_accepts_prequantized() -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuBLASLt MXFP8 block scaling requires Blackwell or newer")
    _init_parallel()
    from sglang.kernels.ops.diffusion import silu_mul_mxfp8

    in_f, out_f, rows = 512, 384, 200
    layer = _layer(in_f, out_f, bias=False)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.mxfp8 and layer.quant_method.accepts_mxfp8_input(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float8_e8m0fnu

    x = torch.randn(rows, in_f, generator=g).to("cuda", torch.bfloat16)
    out, _ = layer(x)
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel

    hidden = torch.randn(rows, 2 * in_f, generator=g).to("cuda", torch.bfloat16)
    act = torch.nn.functional.silu(hidden[:, :in_f]) * hidden[:, in_f:]
    out_tensor, _ = layer(act)
    out_tuple, _ = layer(silu_mul_mxfp8(hidden))
    assert torch.equal(out_tensor, out_tuple)


def test_pre_blackwell_aligned_layer_falls_back_to_channelwise() -> None:
    if torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("requires a pre-Blackwell GPU")
    _init_parallel()
    layer = _layer(512, 384, bias=False)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(384, 512, generator=g) * 0.02).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.mxfp8
    assert not layer.quant_method.accepts_mxfp8_input(layer)
    x = torch.randn(200, 512, generator=g).to("cuda", torch.bfloat16)
    out, _ = layer(x)
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


def test_unaligned_layer_falls_back_to_channelwise() -> None:
    """The block-scaled GEMM needs K % 32 == 0; such a layer keeps the
    per-channel fp8 path and still answers a forward."""
    _init_parallel()
    layer = _layer(8, 128, bias=True)
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.mxfp8 and not layer.quant_method.accepts_mxfp8_input(layer)
    out, _ = layer(torch.randn(4, 8, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (4, 128)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
