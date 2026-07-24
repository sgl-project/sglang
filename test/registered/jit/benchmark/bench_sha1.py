"""Benchmark CUDA JIT SHA-1 vs host hashlib.sha1 for weight-sized tensors."""

from __future__ import annotations

import hashlib

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.sha1 import sha1_prefix_data_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _cpu_sha1_prefix_tensor(prefix: bytes, t: torch.Tensor) -> bytes:
    flat = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    h = hashlib.sha1()
    h.update(prefix)
    h.update(memoryview(flat))
    return h.digest()


def _gpu_sha1_prefix_tensor(prefix: bytes, t: torch.Tensor) -> bytes:
    flat = t.detach().contiguous().reshape(-1).view(torch.uint8)
    return sha1_prefix_data_cuda(prefix, flat)


FN_MAP = {
    "jit_cuda": _gpu_sha1_prefix_tensor,
    "cpu_hashlib": _cpu_sha1_prefix_tensor,
}


# 1 MiB … 256 MiB (element count of fp16 → *2 bytes)
@marker.parametrize(
    "num_fp16",
    [1 << 19, 1 << 22, 1 << 24, 1 << 26],  # 1MiB, 8MiB, 32MiB, 128MiB of fp16
    [1 << 20, 1 << 24],  # CI: 2MiB, 32MiB
)
@marker.benchmark("impl", ["jit_cuda", "cpu_hashlib"], unit="ms")
def benchmark(num_fp16: int, impl: str):
    prefix = b"(1024, 1024)torch.float16"
    t = torch.randn(num_fp16, dtype=torch.float16, device="cuda")
    # Warm JIT once outside do_bench for jit path
    if impl == "jit_cuda":
        _gpu_sha1_prefix_tensor(prefix, t)
        torch.cuda.synchronize()
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(prefix, t),
        # CPU path does D2H; graph capture not meaningful for host hashlib.
        use_cuda_graph=False,
        memory_args=(1,),  # tensor only
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
