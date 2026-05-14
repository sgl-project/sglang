import argparse
from dataclasses import dataclass

import torch

from sglang.jit_kernel.deepseek_v4 import compress_forward


@dataclass(frozen=True)
class BenchCase:
    name: str
    buffer_dtype: torch.dtype
    input_dtype: torch.dtype


CASES = [
    BenchCase("fp32_buffer_fp32_input", torch.float32, torch.float32),
    BenchCase("bf16_buffer_fp32_input", torch.bfloat16, torch.float32),
    BenchCase("bf16_buffer_bf16_input", torch.bfloat16, torch.bfloat16),
]


def _make_inputs(
    case: BenchCase,
    batch_size: int,
    head_dim: int,
    seq_len: int,
    device: str,
):
    torch.manual_seed(42)
    kv_score_buffer = torch.randn(
        (batch_size, 8, head_dim * 4),
        dtype=case.buffer_dtype,
        device=device,
    )
    kv_score_input = torch.randn(
        (batch_size, head_dim * 4),
        dtype=case.input_dtype,
        device=device,
    )
    ape = torch.randn(
        (8, head_dim),
        dtype=case.input_dtype,
        device=device,
    )
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    return kv_score_buffer, kv_score_input, ape, indices, seq_lens


def _run_once(
    case: BenchCase,
    batch_size: int,
    head_dim: int,
    seq_len: int,
    device: str,
):
    kv_score_buffer, kv_score_input, ape, indices, seq_lens = _make_inputs(
        case, batch_size, head_dim, seq_len, device
    )
    return compress_forward(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        indices=indices,
        head_dim=head_dim,
        compress_ratio=4,
        seq_lens=seq_lens,
    )


def _benchmark_case(
    case: BenchCase,
    batch_size: int,
    head_dim: int,
    seq_len: int,
    warmup: int,
    iters: int,
    device: str,
) -> float:
    for _ in range(warmup):
        _run_once(case, batch_size, head_dim, seq_len, device)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        _run_once(case, batch_size, head_dim, seq_len, device)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek V4 C4 compress decode kernel."
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Must be a multiple of 4 to trigger C4 compress compute.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.head_dim % 128 != 0:
        raise ValueError("--head-dim must be a multiple of 128.")
    if args.seq_len % 4 != 0:
        raise ValueError("--seq-len must be a multiple of 4.")

    print(
        f"Benchmark config: batch_size={args.batch_size}, "
        f"head_dim={args.head_dim}, seq_len={args.seq_len}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )
    print("Note: bf16_input cases also use bf16 APE, because the current C4 kernel matches input and APE dtypes.")

    results = {}
    for case in CASES:
        mean_ms = _benchmark_case(
            case,
            batch_size=args.batch_size,
            head_dim=args.head_dim,
            seq_len=args.seq_len,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        results[case.name] = mean_ms

    baseline_ms = results["fp32_buffer_fp32_input"]
    print("\nResults:")
    for case in CASES:
        mean_ms = results[case.name]
        speedup = baseline_ms / mean_ms
        print(
            f"{case.name:24s}  {mean_ms:8.4f} ms/iter  "
            f"speedup_vs_fp32={speedup:6.3f}x"
        )


if __name__ == "__main__":
    main()
