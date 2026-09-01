"""CUDA vs eager-diffusers timing for the sinusoidal timestep embedding.

Moved out of ``ops/diffusion/test_timestep_embedding.py``: it asserted nothing
and was skipped unless ``SGLANG_RUN_JIT_KERNEL_PERF_TESTS=1``, so it belonged
with the other benchmarks rather than in the correctness suite.
"""

import sys

import torch

from sglang.kernels.ops.diffusion import timestep_embedding
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

BATCHES = [1, 2, 8, 63, 256, 512, 613, 1024, 1536]
DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def _reference(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    exponent = -torch.log(
        torch.tensor(max_period, dtype=torch.float32, device=timesteps.device)
    ) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
    emb = torch.exp(exponent / (half_dim - 1))
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def _time_ms(fn, *args, warmup=4, repeat=20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 0
    print(f"{'batch':>7} {'dim':>6} {'torch(ms)':>11} {'cuda(ms)':>10} {'speedup':>8}")
    speedups = []
    for batch in BATCHES:
        for dim in DIMS:
            t = torch.linspace(
                0, max(100000, batch), steps=batch, device="cuda", dtype=torch.float32
            )
            torch_ms = _time_ms(_reference, t, dim)
            cuda_ms = _time_ms(timestep_embedding, t, dim)
            speedups.append(torch_ms / cuda_ms)
            print(
                f"{batch:>7} {dim:>6} {torch_ms:>11.6f} {cuda_ms:>10.6f} "
                f"{speedups[-1]:>8.3f}"
            )
    print(f"average speedup: {sum(speedups) / len(speedups):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
