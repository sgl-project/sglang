# Benchmarks the fused Triton GroupNorm+SiLU kernel against the eager
# silu(group_norm(x)) reference across a set of diffusion-shaped cases.
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.diffusion.triton.group_norm_silu import triton_group_norm_silu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=45,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="standalone benchmark",
)

DEVICE = "cuda"
EPS = 1e-5


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    num_groups: int


CASES = [
    Case("token_2d", (4, 128), 32),
    Case("image_2d", (2, 64, 32, 32), 32),
    Case("video_3d_small", (1, 64, 4, 16, 16), 32),
    Case("threshold_3d", (1, 128, 1, 256, 256), 32),
    Case("hunyuan_video_large", (1, 128, 20, 256, 256), 32),
    # LTX-2 latent upsampler (`LatentUpsampler` + `ResBlock`) operates on
    # `[B, mid_channels=512, F, H, W]` tensors with num_groups=32. The
    # `small` and `pre_720p` cases stay in the default set; the larger
    # `post_720p` case is opt-in via LARGE_CASES below.
    Case("ltx2_upsampler_small", (1, 512, 8, 45, 80), 32),
    Case("ltx2_upsampler_pre_720p", (1, 512, 16, 90, 160), 32),
]

# Cases too large to fit comfortably alongside the native-path intermediates
# on consumer GPUs (e.g. 24 GB L4). `ltx2_upsampler_post_720p` is ~471M bf16
# elements (~940 MB tensor); the eager reference materializes mean / variance
# / normalized / silu intermediates -- working set lands around 5 GB. On
# H100 / H200 this is fine and surfaces the asymptotic ~14x kernel speedup;
# on a 24 GB GPU it can OOM, so it is gated out of the default sweep.
LARGE_CASES = [
    Case("ltx2_upsampler_post_720p", (1, 512, 16, 180, 320), 32),
]

CASE_BY_NAME = {case.name: case for case in CASES + LARGE_CASES}

DTYPE_BY_NAME = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def native_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    return F.silu(F.group_norm(x, num_groups, weight=weight, bias=bias, eps=EPS))


def make_inputs(case: Case, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(len(case.shape) * 1009 + case.shape[1] * 17 + case.num_groups)
    x = torch.randn(case.shape, device=DEVICE, dtype=dtype, generator=generator)
    weight = torch.randn(case.shape[1], device=DEVICE, dtype=dtype, generator=generator)
    bias = torch.randn(case.shape[1], device=DEVICE, dtype=dtype, generator=generator)
    return x, weight, bias


CASE_NAMES = [case.name for case in CASES]


@marker.parametrize("dtype_name", ["bf16", "fp16"], ["bf16"])
@marker.parametrize("case_name", CASE_NAMES, [CASE_NAMES[0]])
@marker.benchmark("provider", ["native", "fused"])
def benchmark(case_name: str, dtype_name: str, provider: str):
    case = CASE_BY_NAME[case_name]
    dtype = DTYPE_BY_NAME[dtype_name]
    x, weight, bias = make_inputs(case, dtype)

    if provider == "native":

        def fn(x, weight, bias):
            return native_group_norm_silu(x, weight, bias, case.num_groups)

    else:

        def fn(x, weight, bias):
            return triton_group_norm_silu(
                x, weight, bias, num_groups=case.num_groups, eps=EPS
            )

    # Pass the read tensors as input_args so do_bench rotates (clones) them per
    # iteration; a zero-arg closure reuses the same buffers and reports L2-hot
    # (wrongly fast) numbers.
    return marker.do_bench(fn, input_args=(x, weight, bias))


if __name__ == "__main__":
    benchmark.run()
