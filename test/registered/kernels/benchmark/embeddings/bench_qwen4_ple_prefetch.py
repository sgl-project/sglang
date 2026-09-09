"""Single-rank, repeated-key PLE lookup benchmark; excludes TP communication."""

import argparse

import torch
import triton.testing

from sglang.kernels.ops.qwen4_ple import (
    fused_qwen4_ngram_gather,
    fused_qwen4_ngram_hash,
)
from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--rows-per-head", type=int, default=65537)
    parser.add_argument("--dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 64, 256, 4096])
    args = parser.parse_args()
    if args.rows_per_head <= 0 or any(n <= 0 for n in args.tokens):
        parser.error("row and token counts must be positive")
    dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    local_rows = (16 * args.rows_per_head + args.tp_size - 1) // args.tp_size
    weight = torch.empty((local_rows, 160), dtype=dtype, pin_memory=True)
    weight.zero_()
    sizes = torch.full((16,), args.rows_per_head, dtype=torch.long, device="cuda")
    offsets = torch.arange(16, dtype=torch.long, device="cuda") * args.rows_per_head
    multipliers = torch.tensor(
        [190734863281251, 953674316406251, 4768371582031251],
        dtype=torch.long,
        device="cuda",
    )
    torch.manual_seed(42)
    print("Single rank, repeated keys, CUDA Graph timing; no TP all-reduce.")
    print("tokens,separate_us,fused_us")
    for tokens in args.tokens:
        contexts = torch.randint(
            0, 151000, (tokens, 3), dtype=torch.long, device="cuda"
        )
        out = torch.empty((tokens, 16, 160), dtype=torch.bfloat16, device="cuda")

        def separate():
            ids = fused_qwen4_ngram_hash(contexts, multipliers, sizes, offsets, 0)
            _gather_ple_embedding_from_pinned_kernel[(tokens * 16,)](
                weight.data_ptr(),
                ids,
                out,
                embedding_dim=160,
                tp_vocab_start=0,
                tp_vocab_end=local_rows,
                is_fp8=dtype == torch.float8_e4m3fn,
                BLOCK_D=256,
            )
            return out

        def fused():
            return fused_qwen4_ngram_gather(
                contexts, multipliers, sizes, offsets, 0, weight, 0, local_rows, out
            )

        expected = separate().clone()
        torch.testing.assert_close(fused(), expected, rtol=0, atol=0)
        separate_ms = triton.testing.do_bench_cudagraph(separate)
        fused_ms = triton.testing.do_bench_cudagraph(fused)
        print(f"{tokens},{separate_ms * 1000:.3f},{fused_ms * 1000:.3f}")


if __name__ == "__main__":
    main()
