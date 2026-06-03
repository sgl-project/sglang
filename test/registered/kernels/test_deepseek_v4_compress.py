import argparse
import sys
import unittest
from dataclasses import dataclass

import torch

from sglang.jit_kernel.dsv4 import CompressorDecodePlan, compress_forward
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import is_in_ci

register_cuda_ci(est_time=7, suite="base-b-kernel-unit-1-gpu-large")


@dataclass(frozen=True)
class C4BenchCase:
    name: str
    buffer_dtype: torch.dtype
    input_dtype: torch.dtype


C4_BENCH_CASES = [
    C4BenchCase("fp32_buffer_fp32_input", torch.float32, torch.float32),
    C4BenchCase("bf16_buffer_fp32_input", torch.bfloat16, torch.float32),
    C4BenchCase("bf16_buffer_bf16_input", torch.bfloat16, torch.bfloat16),
]


def _make_c4_inputs(
    case: C4BenchCase,
    batch_size: int,
    head_dim: int,
    seq_len: int,
    device: str,
):
    torch.manual_seed(42)
    kv_score_buffer = torch.randn(
        (batch_size * 2, 4, head_dim * 4),
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
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio=4,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
    )
    return kv_score_buffer, kv_score_input, ape, plan


def _run_c4_once(
    case: C4BenchCase,
    batch_size: int,
    head_dim: int,
    seq_len: int,
    device: str,
):
    kv_score_buffer, kv_score_input, ape, plan = _make_c4_inputs(
        case, batch_size, head_dim, seq_len, device
    )
    return compress_forward(
        kv_score_buffer=kv_score_buffer,
        kv_score_input=kv_score_input,
        ape=ape,
        plan=plan,
        head_dim=head_dim,
        compress_ratio=4,
    )


def _benchmark_c4_case(
    case: C4BenchCase,
    batch_size: int,
    head_dim: int,
    seq_len: int,
    warmup: int,
    iters: int,
    device: str,
) -> float:
    for _ in range(warmup):
        _run_c4_once(case, batch_size, head_dim, seq_len, device)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        _run_c4_once(case, batch_size, head_dim, seq_len, device)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def _cpu_c128_decode_reference(
    kv_score_buffer: torch.Tensor,
    ape: torch.Tensor,
    seq_len: int,
    head_dim: int,
) -> torch.Tensor:
    scores = []
    kvs = []
    for i in range(128):
        j = (seq_len + i) % 128
        src = kv_score_buffer[0, j]
        kv = src[:head_dim]
        score = src[head_dim:]
        kvs.append(kv.float())
        scores.append(score.float())

    kv_tensor = torch.stack(kvs, dim=0)
    score_tensor = torch.stack(scores, dim=0) + ape.float()
    weights = torch.softmax(score_tensor, dim=0)
    return (kv_tensor * weights).sum(dim=0)


class TestDeepseekV4C4Compress(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_c4_decode_dtype_combinations(self):
        batch_size = 16
        head_dim = 128
        seq_len = 8
        for case in C4_BENCH_CASES:
            with self.subTest(case=case.name):
                out = _run_c4_once(
                    case,
                    batch_size=batch_size,
                    head_dim=head_dim,
                    seq_len=seq_len,
                    device="cuda",
                )
                self.assertEqual(out.shape, (batch_size, head_dim))
                self.assertEqual(out.dtype, case.input_dtype)
                self.assertTrue(torch.isfinite(out).all().item())


class TestDeepseekV4C128Compress(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _run_decode_case(self, buffer_dtype: torch.dtype):
        device = "cuda"
        head_dim = 128
        torch.manual_seed(1)

        kv_score_buffer = torch.zeros(
            (1, 128, head_dim * 2), dtype=buffer_dtype, device=device
        )
        ape = torch.randn((128, head_dim), dtype=torch.float32, device=device)
        req_pool_indices = torch.zeros((1,), dtype=torch.int64, device=device)

        ref_buffer = torch.zeros((1, 128, head_dim * 2), dtype=buffer_dtype)

        for seq_len in range(1, 257):
            kv_score_input = torch.randn(
                (1, head_dim * 2), dtype=torch.float32, device=device
            )
            write_pos = (seq_len + 127) % 128
            ref_buffer[0, write_pos] = kv_score_input[0].cpu().to(buffer_dtype)

            plan = CompressorDecodePlan.generate_legacy(
                compress_ratio=128,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            )
            out = compress_forward(
                kv_score_buffer=kv_score_buffer,
                kv_score_input=kv_score_input,
                ape=ape,
                plan=plan,
                head_dim=head_dim,
                compress_ratio=128,
            )

            self.assertTrue(
                torch.equal(kv_score_buffer.cpu(), ref_buffer),
                msg=f"buffer mismatch for seq_len={seq_len}, dtype={buffer_dtype}",
            )

            if seq_len % 128 != 0:
                continue

            ref_out = _cpu_c128_decode_reference(
                ref_buffer,
                ape.cpu(),
                seq_len=seq_len,
                head_dim=head_dim,
            )
            atol = 2e-4 if buffer_dtype == torch.float32 else 2e-2
            rtol = 2e-4 if buffer_dtype == torch.float32 else 2e-2
            self.assertTrue(
                torch.allclose(out[0].cpu().float(), ref_out, atol=atol, rtol=rtol),
                msg=(
                    f"output mismatch for seq_len={seq_len}, "
                    f"dtype={buffer_dtype}, "
                    f"max_diff={(out[0].cpu().float() - ref_out).abs().max().item():.6f}"
                ),
            )

    def test_c128_decode_fp32_buffer_fp32_input(self):
        self._run_decode_case(torch.float32)

    def test_c128_decode_bf16_buffer_fp32_input(self):
        self._run_decode_case(torch.bfloat16)


def _run_c4_benchmark(argv: list[str]) -> int:
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
    argv = [arg for arg in argv if arg not in ("-f", "--failfast", "--benchmark-c4")]
    args = parser.parse_args(argv)

    if is_in_ci():
        args.batch_size = min(args.batch_size, 16)
        args.warmup = min(args.warmup, 5)
        args.iters = min(args.iters, 20)

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
    print(
        "Note: bf16_input cases also use bf16 APE, because the current C4 "
        "kernel matches input and APE dtypes."
    )

    results = {}
    for case in C4_BENCH_CASES:
        mean_ms = _benchmark_c4_case(
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
    for case in C4_BENCH_CASES:
        mean_ms = results[case.name]
        speedup = baseline_ms / mean_ms
        print(
            f"{case.name:24s}  {mean_ms:8.4f} ms/iter  "
            f"speedup_vs_fp32={speedup:6.3f}x"
        )
    return 0


if __name__ == "__main__":
    if "--benchmark-c4" in sys.argv:
        sys.exit(_run_c4_benchmark(sys.argv[1:]))
    unittest.main()
