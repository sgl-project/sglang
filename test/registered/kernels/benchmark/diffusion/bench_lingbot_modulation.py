# SPDX-License-Identifier: Apache-2.0
"""Whole native LingBot FP32 normalization and elementwise sites."""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    modulate_scale_shift_cuda,
    try_fused_fp32_layernorm_bf16,
    try_fused_scaled_residual_bf16,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    ScaleResidualLayerNormScaleShift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="standalone benchmark",
)


@marker.parametrize("tokens", [4680, 9360])
@marker.parametrize("site", ["norm1", "cross_norm", "camera", "residual"])
@marker.benchmark("provider", ["native", "sglang"])
@torch.inference_mode()
def benchmark(tokens, site, provider):
    channels = 5120
    x = torch.randn(1, tokens, channels, device="cuda", dtype=torch.bfloat16)
    if site in ("norm1", "cross_norm"):
        scale = torch.randn(1, channels, device="cuda")
        shift = torch.randn_like(scale)
        affine = site == "cross_norm"

        def native(x, scale, shift):
            if affine:
                return F.layer_norm(
                    x.float(), (channels,), scale[0], shift[0], 1e-6
                ).bfloat16()
            return (
                F.layer_norm(x.float(), (channels,), eps=1e-6) * (1 + scale[:, None])
                + shift[:, None]
            ).bfloat16()

        def fused(x, scale, shift):
            return try_fused_fp32_layernorm_bf16(x, scale, shift, 1e-6, affine=affine)
    elif site == "camera":
        scale = torch.randn_like(x)
        shift = torch.randn_like(x)

        def native(x, scale, shift):
            return (1 + scale) * x + shift

        def fused(x, scale, shift):
            return modulate_scale_shift_cuda(
                x.view(-1, 1, channels),
                scale.view(-1, channels),
                shift.view(-1, channels),
            ).view_as(x)

    else:
        scale = torch.randn(1, 1, 1, channels, device="cuda")
        shift = torch.randn_like(x)
        norm = ScaleResidualLayerNormScaleShift(
            channels, elementwise_affine=True
        ).cuda()
        zero = x.new_zeros((1,))

        def native(x, scale, shift):
            # The caller discards the first (normalized) output.
            return norm(shift, x, scale, zero, zero)[1]

        def fused(x, scale, shift):
            return try_fused_scaled_residual_bf16(shift, x, scale)

    expected, actual = native(x, scale, shift), fused(x, scale, shift)
    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
    return marker.do_bench(
        native if provider == "native" else fused,
        input_args=(x, scale, shift),
        use_cuda_graph=False,
        replay_iters=100,
        memory_args=(x, scale, shift),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
