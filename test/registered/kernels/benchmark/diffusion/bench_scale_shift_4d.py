import random
import sys
import time
from dataclasses import dataclass

import torch

from sglang.kernels.ops.diffusion import fuse_scale_shift_kernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=25, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@dataclass(frozen=True)
class Workload:
    name: str
    shape: tuple[int, int, int]
    num_frames: int


FULL_WORKLOADS = [
    Workload("wan_s24960_c1536", (1, 24960, 1536), 5),
    Workload("sana_video_s7800_c2240", (1, 7800, 2240), 5),
    Workload("longlive_s1560_c3072", (1, 1560, 3072), 3),
    Workload("lingbot_world_s4680_c5120", (1, 4680, 5120), 1),
]
CI_WORKLOADS = [
    Workload("ci_s1024_c1536", (1, 1024, 1536), 4),
    Workload("ci_s512_c5120", (1, 512, 5120), 2),
]


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

    torch.manual_seed(20260826)
    random.seed(20260826)
    torch.cuda.set_device(0)

    workloads = CI_WORKLOADS if is_in_ci() else FULL_WORKLOADS
    warmups = 5 if is_in_ci() else 20
    repeats = 5 if is_in_ci() else 20
    rounds = 5 if is_in_ci() else 13

    print("| workload | cold ms | torch us | triton us | speedup |")
    print("|---|---:|---:|---:|---:|")

    for workload in workloads:
        batch, seq_len, hidden = workload.shape
        x = torch.randn(workload.shape, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(
            (batch, workload.num_frames, 1, hidden),
            device="cuda",
            dtype=torch.bfloat16,
        )
        shift = torch.randn_like(x)
        frame_seqlen = seq_len // workload.num_frames

        torch_fn = lambda: (
            x.unflatten(1, (workload.num_frames, frame_seqlen)) * (1 + scale)
            + shift.unflatten(1, (workload.num_frames, frame_seqlen))
        ).flatten(1, 2)
        triton_fn = lambda: fuse_scale_shift_kernel(x, scale, shift)

        torch.cuda.synchronize()
        start = time.perf_counter()
        triton_out = triton_fn()
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - start) * 1000.0
        torch.testing.assert_close(triton_out, torch_fn(), atol=5e-2, rtol=5e-2)

        providers = ["torch", "triton"]
        random.shuffle(providers)
        fns = {"torch": torch_fn, "triton": triton_fn}
        times = {
            provider: cuda_event_us(
                fns[provider], warmups=warmups, repeats=repeats, rounds=rounds
            )
            for provider in providers
        }
        print(
            f"| {workload.name} | {cold_ms:.2f} | {times['torch']:.2f} | "
            f"{times['triton']:.2f} | {times['torch'] / times['triton']:.2f}x |"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark()
    sys.exit(0)
