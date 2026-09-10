# SPDX-License-Identifier: Apache-2.0
"""Numeric coverage for the weight-only NVFP4 compressed-tensors linear scheme.

test/registered/kernels/ops/quantization/test_nvfp4_marlin.py already covers the
Marlin kernel itself, driving it from an already-Marlin-shaped layer. What is
only reachable through the scheme is the checkpoint-to-Marlin translation in
process_weights_after_loading: compressed-tensors stores the global scale as a
divisor and names the packed weight `weight_packed`, and both must be converted
before the kernel sees them.
"""

import sys
from collections.abc import Callable

import pytest
import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Fp4,
)
from sglang.srt.utils.common import (
    is_sm80_supported,
    is_sm90_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-small")

SIZE_M = 17
SIZE_K = 256
SIZE_N = 192


def _load_checkpoint_weights(
    layer: torch.nn.Module,
    fp4_weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    """Fill the registered parameters the way a checkpoint would.

    global_scale is stored inverted: that divisor convention is what
    process_weights_after_loading has to undo.
    """
    layer.weight_packed.data.copy_(fp4_weight)
    layer.weight_scale.data.copy_(scales)
    layer.weight_global_scale.data.fill_((1 / global_scale).item())


def _build_layer(
    scheme: CompressedTensorsW4A16Fp4, dtype: torch.dtype
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


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="Weight-only NVFP4 Marlin requires CUDA SM8X/SM9X/SM120",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_input_global_scale", [False, True])
def test_scheme_matches_dequant_reference(dtype, has_input_global_scale):
    """A w4a4 checkpoint served weight-only must produce the same numbers as an
    a16 one; the extra input_global_scale is dropped, not applied."""
    torch.manual_seed(0)

    scheme = CompressedTensorsW4A16Fp4(has_input_global_scale=has_input_global_scale)
    layer = _build_layer(scheme, dtype)

    fp4_weight, scales, global_scale, weight_ref = make_nvfp4_weight_and_ref(
        SIZE_N, SIZE_K, dtype, group_size=16
    )
    _load_checkpoint_weights(layer, fp4_weight, scales, global_scale)
    if has_input_global_scale:
        layer.input_global_scale.data.fill_(1 / 448.0)

    scheme.process_weights_after_loading(layer)

    a_input = torch.randn((SIZE_M, SIZE_K), dtype=dtype, device="cuda") / 10
    output = scheme.apply_weights(layer, a_input)
    output_ref = torch.matmul(a_input, weight_ref.T)
    torch.cuda.synchronize()

    # Relative, not absolute: the error scales with the output magnitude, which
    # depends on K and on the random inputs, so an atol picked for one shape
    # does not transfer. One bf16 eps is 2**-7 = 0.0078.
    rel = (output.float() - output_ref.float()).norm() / output_ref.float().norm()
    assert rel < 0.02, f"relative error {rel:.4f} too large"


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported() or is_sm120_supported()),
    reason="Weight-only NVFP4 Marlin requires CUDA SM8X/SM9X/SM120",
)
def test_uninverted_global_scale_would_overflow():
    """Guards the divisor-vs-scale conversion specifically.

    Feeding the stored divisor straight through overflows the bf16 exponent-bias
    multiply (2**119) to inf, which silently zeroes every logit rather than
    failing. The scheme's own output must stay finite on the same input.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16

    scheme = CompressedTensorsW4A16Fp4()
    layer = _build_layer(scheme, dtype)
    fp4_weight, scales, global_scale, _ = make_nvfp4_weight_and_ref(
        SIZE_N, SIZE_K, dtype, group_size=16
    )
    _load_checkpoint_weights(layer, fp4_weight, scales, global_scale)

    stored = layer.weight_global_scale.max().clone()
    scheme.process_weights_after_loading(layer)

    assert torch.isfinite(layer.weight_global_scale).all()
    # The un-inverted value is what a missing reciprocal would have passed on.
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
    )

    unconverted = nvfp4_marlin_process_global_scale(stored.to(dtype))
    assert not torch.isfinite(unconverted).all(), (
        "expected the un-inverted divisor to overflow; if this now stays finite "
        "the reciprocal guard above no longer proves anything"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
