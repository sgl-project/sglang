# SPDX-License-Identifier: Apache-2.0
"""Numeric coverage for the weight-only MXFP4 compressed-tensors linear scheme.

The MXFP4 path differs from NVFP4 in what the translation has to get right:
scales arrive as raw E8M0 exponent bytes that must be reinterpreted rather than
converted, they are permuted and interleaved for Marlin's layout, and there is no
global scale to invert. BF16 is the only supported activation dtype, because the
E8M0 scale decoder is instantiated for bf16 alone.
"""

import sys
from collections.abc import Callable

import pytest
import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Mxfp4,
)
from sglang.srt.utils.common import (
    is_sm80_supported,
    is_sm90_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_mxfp4_weight_and_ref

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-small")

SIZE_M = 17
SIZE_K = 256
SIZE_N = 192

requires_fp4_marlin = pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="Weight-only MXFP4 Marlin requires CUDA SM8X/SM9X/SM120",
)


def _build_layer(
    scheme: CompressedTensorsW4A16Mxfp4, dtype: torch.dtype
) -> torch.nn.Module:
    layer = torch.nn.Module()
    weight_loader: Callable = lambda *args, **kwargs: None
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[SIZE_N],
        input_size_per_partition=SIZE_K,
        params_dtype=dtype,
        weight_loader=weight_loader,
    )
    layer.to("cuda")
    return layer


@requires_fp4_marlin
@pytest.mark.parametrize("has_input_activations", [False, True])
def test_scheme_matches_dequant_reference(has_input_activations):
    """A w4a4 checkpoint served weight-only must produce the same numbers as an
    a16 one; the declared activation quantization is ignored, not applied."""
    torch.manual_seed(0)
    dtype = torch.bfloat16

    scheme = CompressedTensorsW4A16Mxfp4(has_input_activations=has_input_activations)
    layer = _build_layer(scheme, dtype)

    fp4_weight, scales, weight_ref = make_mxfp4_weight_and_ref(
        SIZE_N, SIZE_K, dtype, group_size=32
    )
    layer.weight_packed.data.copy_(fp4_weight)
    layer.weight_scale.data.copy_(scales)

    scheme.process_weights_after_loading(layer)

    a_input = torch.randn((SIZE_M, SIZE_K), dtype=dtype, device="cuda") / 10
    output = scheme.apply_weights(layer, a_input)
    output_ref = torch.matmul(a_input, weight_ref.T)
    torch.cuda.synchronize()

    assert torch.isfinite(output).all(), "MXFP4 Marlin produced non-finite output"
    # Relative, not absolute: the error scales with the output magnitude, which
    # depends on K and on the random inputs, so an atol picked for one shape
    # does not transfer. One bf16 eps is 2**-7 = 0.0078.
    rel = (output.float() - output_ref.float()).norm() / output_ref.float().norm()
    assert rel < 0.02, f"relative error {rel:.4f} too large"


@requires_fp4_marlin
def test_fp16_activations_rejected():
    """FP16 must fail loudly.

    dequant_fp8_scales<half2, kFE8M0fnu> is declared but never defined, so no
    fp16 MXFP4 kernel is instantiated and the dispatch would fall through to the
    no-op MarlinDefault, returning silent garbage instead of an error.
    """
    scheme = CompressedTensorsW4A16Mxfp4()
    layer = _build_layer(scheme, torch.float16)

    fp4_weight, scales, _ = make_mxfp4_weight_and_ref(
        SIZE_N, SIZE_K, torch.bfloat16, group_size=32
    )
    layer.weight_packed.data.copy_(fp4_weight)
    layer.weight_scale.data.copy_(scales)

    with pytest.raises(RuntimeError, match="BF16"):
        scheme.process_weights_after_loading(layer)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
