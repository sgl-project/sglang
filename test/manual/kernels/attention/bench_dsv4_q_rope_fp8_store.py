"""Compare FP8 Q preparation with fused_q_norm_rope followed by FP8 conversion.

Run on the target GPU before deciding whether to retain the specialization:
python test/manual/kernels/attention/bench_dsv4_q_rope_fp8_store.py
"""

import torch
import triton

from sglang.kernels.ops.attention.dsv4.elementwise import fused_q_norm_rope
from sglang.kernels.ops.attention.dsv4.q_rope_fp8_store import q_rope_fp8_store


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("rows,heads,fused_bf16_then_fp8_us,direct_fp8_us,speedup")
    for rows, heads in [(6, 16), (384, 16), (4097, 16), (32769, 128)]:
        q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
        bf16 = torch.empty_like(q)
        fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        freqs = torch.polar(
            torch.ones(8192, 32, device="cuda"), torch.randn(8192, 32, device="cuda")
        )
        positions = torch.arange(rows, device="cuda") % 8192
        # do_bench_cudagraph warms up on a new stream without waiting for the
        # current stream. Finish initializing inputs before it reads positions.
        torch.cuda.synchronize()

        def baseline():
            fused_q_norm_rope(q, bf16, None, freqs, positions)
            fp8.copy_(bf16)

        def candidate():
            q_rope_fp8_store(q, fp8, freqs, positions)

        baseline_ms = triton.testing.do_bench_cudagraph(baseline)
        candidate_ms = triton.testing.do_bench_cudagraph(candidate)
        print(
            f"{rows},{heads},{baseline_ms * 1000:.3f},"
            f"{candidate_ms * 1000:.3f},{baseline_ms / candidate_ms:.3f}"
        )


if __name__ == "__main__":
    main()
