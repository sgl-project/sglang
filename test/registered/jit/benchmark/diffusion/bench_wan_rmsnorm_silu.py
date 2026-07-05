import math
import random
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.triton.wan_rmsnorm_silu import (
    triton_wan_rmsnorm_silu,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)


@dataclass(frozen=True)
class Workload:
    name: str
    shape: tuple[int, int, int, int, int]


FULL_WORKLOADS = [
    Workload("fastwan_decode_c384_t21_90x160", (1, 384, 21, 90, 160)),
    Workload("fastwan_decode_c192_t41_180x320", (1, 192, 41, 180, 320)),
    Workload("fastwan_decode_c96_t81_360x640", (1, 96, 81, 360, 640)),
]
CI_WORKLOADS = [
    Workload("fastwan_decode_c384_t5_45x80", (1, 384, 5, 45, 80)),
    Workload("fastwan_decode_c192_t5_90x160", (1, 192, 5, 90, 160)),
]


def native_wan_rmsnorm_silu(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return F.silu(F.normalize(x, dim=1) * math.sqrt(x.shape[1]) * gamma)


def cuda_event_us(fn, warmups: int, repeats: int, rounds: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark() -> None:
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    if not hasattr(torch, "channels_last_3d"):
        print("channels_last_3d required")
        return

    torch.manual_seed(20260705)
    random.seed(20260705)
    torch.cuda.set_device(0)

    workloads = CI_WORKLOADS if is_in_ci() else FULL_WORKLOADS
    warmups = 5 if is_in_ci() else 20
    repeats = 5 if is_in_ci() else 20
    rounds = 5 if is_in_ci() else 13

    print("| workload | native us | triton us | speedup |")
    print("|---|---:|---:|---:|")

    with torch.inference_mode():
        for workload in workloads:
            x = torch.randn(
                workload.shape, device="cuda", dtype=torch.bfloat16
            ).contiguous(memory_format=torch.channels_last_3d)
            gamma = torch.randn(
                (workload.shape[1], 1, 1, 1), device="cuda", dtype=torch.bfloat16
            )

            actual = triton_wan_rmsnorm_silu(x, gamma)
            expected = native_wan_rmsnorm_silu(x, gamma)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=1.5e-1, rtol=3e-2)

            fns = {
                "native": lambda: native_wan_rmsnorm_silu(x, gamma),
                "triton": lambda: triton_wan_rmsnorm_silu(x, gamma),
            }
            order = ["native", "triton"]
            random.shuffle(order)
            times = {
                name: cuda_event_us(fns[name], warmups, repeats, rounds)
                for name in order
            }

            print(
                f"| {workload.name} | {times['native']:.2f} | "
                f"{times['triton']:.2f} | {times['native'] / times['triton']:.3f}x |"
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark()
    sys.exit(0)
