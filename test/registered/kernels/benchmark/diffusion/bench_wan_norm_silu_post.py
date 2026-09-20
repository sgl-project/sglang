# SPDX-License-Identifier: Apache-2.0
"""Whole native Wan norm+SiLU versus the lossless post-op fusion."""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import wan_norm_silu_post
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="standalone benchmark",
)

SHAPES = {
    "c1024_t1_h30_w52": (1, 1024, 1, 30, 52),
    "c1024_t2_h60_w104": (1, 1024, 2, 60, 104),
    "c256_t4_h240_w416": (1, 256, 4, 240, 416),
}


@torch.inference_mode()
def native(x, gamma):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return F.silu(F.normalize(x, dim=1) * x.shape[1] ** 0.5 * gamma + 0.0)


@torch.inference_mode()
def fused(x, gamma):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        denominator = x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return wan_norm_silu_post(x, denominator, gamma)


@marker.parametrize("case", list(SHAPES))
@marker.parametrize("channels_last", [False, True])
@marker.benchmark("provider", ["torch", "sglang"])
def benchmark(case, channels_last, provider):
    shape = SHAPES[case]
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    gamma = torch.randn(shape[1], 1, 1, 1, device="cuda", dtype=x.dtype)
    expected, actual = native(x, gamma), fused(x, gamma)
    assert torch.equal(expected.view(torch.int32), actual.view(torch.int32))
    assert actual.stride() == expected.stride()
    return marker.do_bench(
        native if provider == "torch" else fused,
        input_args=(x, gamma),
        use_cuda_graph=False,
        replay_iters=100,
        memory_args=(x, gamma),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
