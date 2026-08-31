import random
import sys
from dataclasses import dataclass

import torch

from sglang.kernels.ops.diffusion import fuse_scale_shift_kernel, residual_gate_add_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@dataclass(frozen=True)
class Workload:
    name: str
    residual_shape: tuple[int, ...]
    gate_shape: tuple[int, ...]
    transposed_residual: bool = False


FULL_WORKLOADS = [
    Workload("ltx2_bcast_s32640_c4096", (1, 32640, 4096), (1, 1, 4096)),
    Workload("ltx2_full_s8160_c4096", (1, 8160, 4096), (1, 8160, 4096)),
    Workload("ideogram4_bcast_s4096_c4608", (1, 4096, 4608), (1, 1, 4608)),
    Workload("flux2_bcast_s4608_c3072", (1, 4608, 3072), (1, 1, 3072)),
    Workload("flux2_bcast_s4096_c3072", (1, 4096, 3072), (1, 1, 3072)),
    Workload("flux2_bcast_s512_c3072", (1, 512, 3072), (1, 1, 3072)),
    Workload("ltx2_full_s126_c2048", (1, 126, 2048), (1, 126, 2048)),
    Workload(
        "sana_video_bcast_s7800_c2240_transposed",
        (1, 7800, 2240),
        (1, 1, 2240),
        transposed_residual=True,
    ),
]
CI_WORKLOADS = [
    Workload("ltx2_bcast_s1024_c4096", (1, 1024, 4096), (1, 1, 4096)),
    Workload("ltx2_full_s512_c4096", (1, 512, 4096), (1, 512, 4096)),
    Workload(
        "sana_video_bcast_s512_c2240_transposed",
        (1, 512, 2240),
        (1, 1, 2240),
        transposed_residual=True,
    ),
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

    torch.manual_seed(20260625)
    random.seed(20260625)
    torch.cuda.set_device(0)

    workloads = CI_WORKLOADS if is_in_ci() else FULL_WORKLOADS
    warmups = 5 if is_in_ci() else 20
    repeats = 5 if is_in_ci() else 20
    rounds = 5 if is_in_ci() else 13

    print(
        "| workload | gate | torch us | triton us | cuda us | reference | " "ref/cuda |"
    )
    print("|---|---|---:|---:|---:|---|---:|")

    for workload in workloads:
        if workload.transposed_residual:
            batch, tokens, hidden_size = workload.residual_shape
            residual = torch.randn(
                (batch, hidden_size, tokens), device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
        else:
            residual = torch.randn(
                workload.residual_shape, device="cuda", dtype=torch.bfloat16
            )
        update = torch.randn(
            workload.residual_shape, device="cuda", dtype=torch.bfloat16
        )
        gate = torch.randn(workload.gate_shape, device="cuda", dtype=torch.bfloat16)

        ref = residual + update * gate
        cuda_out = residual_gate_add_cuda(residual, update, gate)
        torch.cuda.synchronize()
        torch.testing.assert_close(cuda_out, ref, atol=5e-2, rtol=5e-2)

        fns = {
            "torch": lambda: residual + update * gate,
            "cuda": lambda: residual_gate_add_cuda(residual, update, gate),
        }
        if not workload.transposed_residual:
            triton_out = fuse_scale_shift_kernel(
                update, gate, residual, scale_constant=0
            )
            torch.testing.assert_close(triton_out, ref, atol=5e-2, rtol=5e-2)
            fns["triton"] = lambda: fuse_scale_shift_kernel(
                update, gate, residual, scale_constant=0
            )
        order = list(fns)
        random.shuffle(order)
        times = {
            name: cuda_event_us(fns[name], warmups, repeats, rounds) for name in order
        }

        gate_kind = (
            "bcast" if workload.gate_shape != workload.residual_shape else "full"
        )
        reference = "torch" if workload.transposed_residual else "triton"
        triton_us = f"{times['triton']:.2f}" if "triton" in times else "n/a"
        print(
            f"| {workload.name} | {gate_kind} | {times['torch']:.2f} | "
            f"{triton_us} | {times['cuda']:.2f} | {reference} | "
            f"{times[reference] / times['cuda']:.3f}x |"
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark()
    sys.exit(0)
