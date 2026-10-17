"""Compare cuBLAS and mixed-weight tiny_gemm.

Run with the candidate Python package on CUDA SM90+:
    python test/manual/kernels/bench_fp32_router.py

Times are microseconds per layer under CUDA graph replay. The baseline includes
BF16-to-FP32 activation conversion; weights stay FP32 for all implementations.
The 245 MiB weight working set exceeds H200/B300 L2. Compilation is untimed.
"""

import json

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16


def bench(fn, weights):
    def run_layers():
        for weight in weights:
            fn(weight)

    result = marker.do_bench(
        run_layers,
        warmup_iters=2,
        metrics=(0.5,),
        memory_output=None,
    )
    return result.times[0] * 1e6 / len(weights)


def main():
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
    for m in range(1, 17):
        x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
        base = bench(lambda w: torch.nn.functional.linear(x.float(), w), weights)
        ref = x.double() @ weights[0].double().T
        out = tiny_gemm_bf16(x, weights[0], out_dtype=torch.float32)
        err = (out.double() - ref).abs().max().item()
        us = bench(lambda w: tiny_gemm_bf16(x, w, out_dtype=torch.float32), weights)
        print(
            json.dumps(
                dict(
                    m=m,
                    variant="tiny",
                    base_us=base,
                    us=us,
                    speedup=base / us,
                    max_error=err,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
