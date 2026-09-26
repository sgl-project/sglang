"""Qwen-Image 2.1 VAE decoder channel RMSNorm + SiLU on the production shapes.

decoder_base_dim 144, dim_mult 1/2/4/8/8, latent 64x64 for a 1024x1024 image:
each up block runs its residual blocks at its own resolution and then
upsamples, so the last block's six norms run at 1024x1024 with 144 (first block
also 288) channels. `cuda` is the lossless path (aten fp32 reduction kept, fused
finish + SiLU); `cuda_nhwc` is the quality-gated channels_last kernel.
"""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    channel_rmsnorm_finish_silu,
    channel_rmsnorm_silu_nhwc,
)
from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_preserve_reduction import (
    channel_rmsnorm_preserve_reduction,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)


@marker.parametrize(
    "channels,size",
    [(144, 1024), (288, 1024), (288, 512), (576, 256), (1152, 128)],
    [(144, 256)],
)
@marker.benchmark("impl", ["eager", "triton", "cuda", "cuda_nhwc"])
def benchmark_vae_norm_silu(channels: int, size: int, impl: str):
    generator = torch.Generator(device="cuda").manual_seed(20260925)
    x = torch.randn(
        1,
        channels,
        1,
        size,
        size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gamma = torch.randn(
        channels, 1, 1, 1, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    scale = channels**0.5
    x_nhwc = x.contiguous(memory_format=torch.channels_last_3d)

    def eager():
        normalized = F.normalize(x.float(), dim=1).to(x.dtype)
        return F.silu(normalized * scale * gamma + 0.0)

    fns = {
        "eager": eager,
        "triton": lambda: F.silu(channel_rmsnorm_preserve_reduction(x, gamma, scale)),
        "cuda": lambda: channel_rmsnorm_finish_silu(
            x, x.float().norm(p=2, dim=1, keepdim=True), gamma, scale
        ),
        "cuda_nhwc": lambda: channel_rmsnorm_silu_nhwc(x_nhwc, gamma, scale),
    }
    return marker.do_bench(
        fns[impl],
        memory_args=(x, gamma),
        memory_output=None,
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark_vae_norm_silu.run()
