"""Compare the router boundary against FP32 cuBLAS with 49 rotating weights.

Run with the candidate Python package on CUDA SM90+:
    python test/manual/kernels/bench_fp32_router.py

Times are microseconds per layer under CUDA graph replay. The baseline includes
BF16-to-FP32 activation conversion; weights stay FP32 for both implementations.
The 245 MiB weight working set exceeds H200/B300 L2. Compilation is untimed.
"""

import json
import statistics

import torch

from sglang.kernels.ops.gemm.fp32_router import fp32_router_gemm

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
print(
    json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}),
    flush=True,
)
weights = [
    torch.randn(256, 5120, device="cuda", dtype=torch.float32) / 5120**0.5
    for _ in range(49)
]


def bench(fn):
    for _ in range(2):
        for w in weights:
            fn(w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in weights:
            fn(w)
    for _ in range(5):
        g.replay()
    samples = []
    for _ in range(7):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        for _ in range(20):
            g.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1000 / 20 / 49)
    return statistics.median(samples)


for m in (*range(1, 17), 24, 32, 48, 64):
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    base = bench(lambda w: torch.nn.functional.linear(x.float(), w))
    ref = x.double() @ weights[0].double().T
    fn = lambda w: fp32_router_gemm(x, w)
    out = fn(weights[0])
    torch.cuda.synchronize()
    err = (out.double() - ref).abs().max().item()
    us = bench(fn)
    print(
        json.dumps(dict(m=m, base_us=base, us=us, speedup=base / us, max_error=err)),
        flush=True,
    )
