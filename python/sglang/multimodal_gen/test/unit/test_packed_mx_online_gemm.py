# SPDX-License-Identifier: Apache-2.0
"""Online packed-MX GEMM paths (``Mxfp6Config()`` / ``A6W4Config()`` on a bf16
checkpoint): load-time quantization into AITER's packed blob, and the fallback
to the plain linear for a layer whose parameters are not bf16."""

import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx95_supported(),
    reason="AITER A6W6 requires ROCm gfx95x",
)


def _config(fmt: str):
    from sglang.multimodal_gen.runtime.layers.quantization.a6w4 import A6W4Config
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp6 import Mxfp6Config

    return {"mxfp6": Mxfp6Config, "a6w4": A6W4Config}[fmt]()


def _layer(
    in_f: int,
    out_f: int,
    bias: bool,
    params_dtype: torch.dtype = torch.bfloat16,
    fmt: str = "mxfp6",
):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear

    return RowParallelLinear(
        in_f,
        out_f,
        bias=bias,
        params_dtype=params_dtype,
        quant_config=_config(fmt),
        prefix="mlp.fc2",
    ).to("cuda")


@pytest.mark.parametrize("fmt", ["mxfp6", "a6w4"])
def test_packed_mx_linear_matches_bf16(request, fmt) -> None:
    """The packed weight is a flat uint8 blob, so N and K reach the gemm as
    ints rather than tensor shapes; a wrong one, or swapped A/B operands,
    shows up here and nowhere else."""
    request.getfixturevalue("single_process_model_parallel")

    in_f, out_f, rows = 512, 384, 200
    layer = _layer(in_f, out_f, bias=False, fmt=fmt)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.packed_mx
    assert layer.weight.dtype == torch.uint8 and layer.weight.dim() == 1

    x = torch.randn(rows, in_f, generator=g).to("cuda", torch.bfloat16)
    out, _ = layer(x)
    # out_f is not a multiple of the gemm's 256-wide tile, so the kernel hands
    # back a strided slice of its padded output. The custom op must return the
    # dense layout its fake implementation reports, or a compiled graph plans
    # the wrong strides for everything downstream.
    assert out.is_contiguous()
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    # a6w4's 4-bit weights carry more error than mxfp6's 6-bit ones; both
    # bounds stay tight enough that a silent drop to a coarser format fails.
    assert rel < {"mxfp6": 0.08, "a6w4": 0.14}[fmt], rel

    x3d = torch.randn(2, rows, in_f, generator=g).to("cuda", torch.bfloat16)
    out3d, _ = layer(x3d)
    assert out3d.shape == (2, rows, out_f)


def test_fp16_layer_falls_back_to_plain_linear(request) -> None:
    """The A6W6 gemm emits bf16 only; an fp16 layer must keep the plain linear
    rather than failing at load or returning a bf16 result."""
    request.getfixturevalue("single_process_model_parallel")

    in_f, out_f = 512, 384
    layer = _layer(in_f, out_f, bias=False, params_dtype=torch.float16)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("cuda", torch.float16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.packed_mx

    x = torch.randn(200, in_f, generator=g).to("cuda", torch.float16)
    out, _ = layer(x)
    assert out.dtype == torch.float16
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
