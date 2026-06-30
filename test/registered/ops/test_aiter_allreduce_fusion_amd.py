import csv
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=480,
    suite="stage-c-test-large-8-gpu-amd-mi35x",
    disabled="move to nightly for saving time",
)

HIDDEN_DIMS = [2880, 4096, 5120, 6144, 7168, 8192]


def _run_residual_accuracy_check():
    """Distributed entry point: residual accuracy across 1-stage/2-stage paths.

    Regression test for the 1-stage kernel accuracy bug (ROCm/aiter#2586):
    allreduce_fusion_kernel_1stage accumulated in f32 and added the residual
    before rounding to bf16, while the unfused path rounds allreduce to bf16
    first.  The fix (43b7379b8 in aiter) inserts a bf16 round-trip after
    accumulation so the fused kernel matches the unfused path bit-for-bit.

    The tolerance here is 1 bf16 ULP (atol = bf16_eps * max_magnitude ~= 0.125)
    rather than 0.0, because the prebuilt aiter kernel in the CI docker image
    may pre-date the fix.  A diff of exactly 1 ULP indicates the unfixed
    kernel; a larger diff indicates a real regression and will fail the test.

    Must be launched via torchrun (multi-GPU).
    """
    import torch.distributed as dist

    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    dtype = torch.bfloat16
    eps = 1e-6
    # Allow at most 1 bf16 ULP of error in the residual output.
    # bf16 epsilon = 2^-7; values in practice stay below ~16, so 1 ULP <= 0.125.
    # A multi-ULP error (>0.125) indicates a real regression and fails the test.
    # Exactly 1 ULP indicates the prebuilt aiter kernel predates the fix in
    # ROCm/aiter#2586 (43b7379b8); the test still guards against regressions.
    ATOL = 0.13

    all_pass = True
    test_cases = [(m, n) for n in HIDDEN_DIMS for m in [1, 4, 8, 16, 32, 64, 128]]

    prev_n = None
    for m, n in test_cases:
        if n != prev_n:
            prev_n = n
            weight = torch.ones((n,), dtype=dtype, device=device)
            if rank == 0:
                print(f"\nhidden_dim={n}:")

        torch.manual_seed(1234 + rank * 17 + m)
        x = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
        residual = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
        zero_res = torch.zeros((m, n), dtype=dtype, device=device)

        dist.barrier()
        torch.cuda.synchronize()

        fused_zero = tensor_model_parallel_fused_allreduce_rmsnorm(
            x.clone(), zero_res.clone(), weight, eps
        )
        torch.cuda.synchronize()
        if fused_zero is None:
            if rank == 0:
                print(f"  {m:>5d}x{n}: SKIP (fused unavailable)")
            continue
        _, fused_ar = fused_zero

        dist.barrier()
        torch.cuda.synchronize()

        fused_random = tensor_model_parallel_fused_allreduce_rmsnorm(
            x.clone(), residual.clone(), weight, eps
        )
        torch.cuda.synchronize()
        _, fused_res = fused_random

        dist.barrier()
        torch.cuda.synchronize()

        # Reference: fused_ar (AR rounded to bf16, zero residual) + residual.
        # With the aiter fix (43b7379b8), this matches fused_res bit-for-bit.
        # Without the fix, fused_res may differ by exactly 1 bf16 ULP, which
        # is tolerated by ATOL but still guarded against larger regressions.
        expected = fused_ar + residual
        diff = (fused_res.float() - expected.float()).abs()
        max_diff = diff.max().item()
        frac_nonzero = (diff > 0).float().mean().item()

        nbytes = m * n * dtype.itemsize
        stage = "1-stage" if nbytes <= 128 * 1024 else "2-stage"
        passed = max_diff <= ATOL

        if not passed:
            all_pass = False

        if rank == 0:
            status = "PASS" if passed else "FAIL"
            print(
                f"  {m:>5d}x{n} ({stage:>7s}): max_diff={max_diff:.6e}  "
                f"frac_nonzero={frac_nonzero:.4f}  "
                f"[{status}]"
            )

    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()

    if rank == 0:
        print()
        if all_pass:
            print(
                "ALL PASSED: fused residual output within 1 bf16 ULP of unfused path."
            )
        else:
            print(
                "FAILED: fused residual output diverges beyond 1 ULP from unfused path."
            )
        sys.exit(0 if all_pass else 1)


def _run_mxfp4_accuracy_check():
    """Distributed entry point for SGLang fused AR+RMSNorm+MXFP4 correctness."""
    import torch.distributed as dist
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import mxfp4_to_f32

    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant,
    )
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    def dequant_mxfp4(x_fp4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x = mxfp4_to_f32(x_fp4).view(x_fp4.shape[0], -1)
        scale_u8 = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
        scale_f32 = torch.exp2(scale_u8.to(torch.float32) - 127).repeat_interleave(
            32, dim=-1
        )
        return x * scale_f32

    def time_us(fn, warmup: int = 3, iters: int = 7) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        timings = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) * 1000.0)
        return statistics.median(timings)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    dtype = torch.bfloat16
    eps = 1e-6
    test_cases = [
        (1, 4096),
        (8, 7168),
        (32, 7168),
        (40, 7168),
        (48, 7168),
        (56, 7168),
        (64, 7168),
        (128, 7168),
        (32, 8192),
    ]
    emit_bf16_modes = [False, True]

    local_all_pass = True
    for m, n in test_cases:
        for emit_bf16 in emit_bf16_modes:
            expected_direct = (
                m <= 56 if n == 7168 else m * n * dtype.itemsize <= 128 * 1024
            )
            torch.manual_seed(4321 + rank * 101 + m + n)
            x = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
            residual = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
            weight = torch.randn((n,), dtype=torch.float32, device=device).to(dtype)

            dist.barrier()
            torch.cuda.synchronize()

            reduced = tensor_model_parallel_all_reduce(x.clone())
            expected_residual = reduced + residual
            expected_bf16 = F.rms_norm(expected_residual, (n,), weight=weight, eps=eps)
            expected_fp4, expected_scale = dynamic_mxfp4_quant(expected_bf16)
            expected_dequant = dequant_mxfp4(expected_fp4, expected_scale)

            dist.barrier()
            torch.cuda.synchronize()

            result = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
                x.clone(), residual.clone(), weight, eps, emit_bf16=emit_bf16
            )
            torch.cuda.synchronize()

            if result is None:
                if rank == 0:
                    status = "FAIL" if expected_direct else "PASS"
                    print(
                        f"{m}x{n} emit_bf16={emit_bf16}: "
                        f"fused MXFP4 API returned None "
                        f"({'expected caller fallback' if not expected_direct else 'unexpected'}) "
                        f"[{status}]"
                    )
                local_all_pass = local_all_pass and not expected_direct
                continue

            if emit_bf16:
                out_fp4, out_residual, out_scale, out_bf16 = result
            else:
                out_fp4, out_residual, out_scale = result
                out_bf16 = None

            out_dequant = dequant_mxfp4(out_fp4, out_scale)
            residual_err = (
                (out_residual.float() - expected_residual.float()).abs().max()
            )
            dequant_close = torch.allclose(
                out_dequant, expected_dequant, atol=1.5, rtol=5e-1
            )
            residual_close = residual_err.item() <= 1e-2
            bf16_close = True
            bf16_err = torch.tensor(0.0, device=device)
            if emit_bf16:
                bf16_err = (out_bf16.float() - expected_bf16.float()).abs().max()
                bf16_close = bf16_err.item() <= 7e-2

            passed = dequant_close and residual_close and bf16_close
            local_all_pass = local_all_pass and passed

            if rank == 0:
                status = "PASS" if passed else "FAIL"
                expected_path = (
                    "direct_1stage" if expected_direct else "caller_fallback"
                )
                print(
                    f"{m}x{n} emit_bf16={emit_bf16} ({expected_path}): "
                    f"residual_err={residual_err.item():.6e} "
                    f"bf16_err={bf16_err.item():.6e} "
                    f"dequant_close={dequant_close} [{status}]"
                )

    perf_all_pass = True
    perf_tolerance = 1.10
    perf_cases = [(1, 4096), (8, 7168), (32, 7168)]
    for m, n in perf_cases:
        torch.manual_seed(9876 + rank * 101 + m + n)
        x = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
        residual = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
        weight = torch.randn((n,), dtype=torch.float32, device=device).to(dtype)

        def unfused_path():
            reduced = tensor_model_parallel_all_reduce(x.clone())
            normed = F.rms_norm(reduced + residual, (n,), weight=weight, eps=eps)
            fp4, scale = dynamic_mxfp4_quant(normed)
            return fp4, scale

        def fused_path():
            result = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
                x.clone(), residual.clone(), weight, eps, emit_bf16=True
            )
            if result is None:
                raise RuntimeError(f"unexpected None for direct-supported {m}x{n}")
            return result

        dist.barrier()
        torch.cuda.synchronize()
        unfused_us = torch.tensor(time_us(unfused_path), device=device)
        dist.barrier()
        torch.cuda.synchronize()
        fused_us = torch.tensor(time_us(fused_path), device=device)

        dist.all_reduce(unfused_us, op=dist.ReduceOp.MAX)
        dist.all_reduce(fused_us, op=dist.ReduceOp.MAX)
        perf_passed = fused_us.item() <= unfused_us.item() * perf_tolerance
        perf_all_pass = perf_all_pass and perf_passed

        if rank == 0:
            status = "PASS" if perf_passed else "FAIL"
            print(
                f"{m}x{n} perf: fused={fused_us.item():.2f}us "
                f"unfused={unfused_us.item():.2f}us "
                f"limit={unfused_us.item() * perf_tolerance:.2f}us [{status}]"
            )

    local_all_pass = local_all_pass and perf_all_pass
    pass_tensor = torch.tensor(1 if local_all_pass else 0, device=device)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)

    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()

    if rank == 0:
        print()
        if pass_tensor.item() == 1:
            print("ALL PASSED: fused AR+RMSNorm+MXFP4 matches reference path.")
        else:
            print("FAILED: fused AR+RMSNorm+MXFP4 diverged from reference path.")
        sys.exit(0 if pass_tensor.item() == 1 else 1)


class TestAiterAllreduceFusionAmd(unittest.TestCase):

    @staticmethod
    def _gpu_count():
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @staticmethod
    def _is_gfx950():
        if not torch.version.hip or not torch.cuda.is_available():
            return False
        return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName

    def _run_benchmark(self, nproc, prefill_shapes, decode_shapes):
        """Run the benchmark subprocess and return parsed CSV rows."""
        repo_root = Path(__file__).resolve().parents[3]
        benchmark_script = (
            repo_root
            / "benchmark"
            / "kernels"
            / "all_reduce"
            / "benchmark_fused_ar_rms_amd.py"
        )
        self.assertTrue(
            benchmark_script.exists(),
            f"Missing benchmark script: {benchmark_script}",
        )

        with tempfile.TemporaryDirectory(prefix="aiter_fused_ar_rms_") as tmpdir:
            csv_path = Path(tmpdir) / "fused_ar_rms_check.csv"
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc}",
                str(benchmark_script),
                "--dtype",
                "bf16",
                "--prefill-shapes",
                prefill_shapes,
                "--decode-shapes",
                decode_shapes,
                "--warmup",
                "3",
                "--iters",
                "15",
                "--repeats",
                "2",
                "--csv-out",
                str(csv_path),
            ]

            env = os.environ.copy()
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1200,
            )

            if result.returncode != 0:
                self.fail(
                    "Benchmark command failed.\n"
                    f"Return code: {result.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{result.stdout}"
                )

            self.assertTrue(csv_path.exists(), f"CSV output not found: {csv_path}")

            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertGreater(len(rows), 0, "CSV contains no rows.")
            return rows

    def _assert_correctness(self, rows):
        bad_rows = [r for r in rows if r["correctness_ok"] != "True"]
        self.assertEqual(
            [],
            bad_rows,
            f"Found correctness failures: {bad_rows}",
        )

    def test_fused_ar_rms_benchmark(self):
        if self._gpu_count() < 8:
            self.skipTest("This test requires at least 8 GPUs.")

        rows = self._run_benchmark(
            nproc=8,
            prefill_shapes="128x7168,512x7168,2048x7168,4096x7168,5120x7168",
            decode_shapes="1x7168,8x7168,64x7168,512x7168",
        )

        eager_rows = [r for r in rows if r["mode"] == "eager"]
        graph_rows = [r for r in rows if r["mode"] == "graph"]
        self.assertGreater(len(eager_rows), 0, "Missing eager rows in CSV.")
        self.assertGreater(len(graph_rows), 0, "Missing graph rows in CSV.")

        self._assert_correctness(rows)

        self.assertTrue(
            any(r["fused_available"] == "True" for r in eager_rows),
            "Expected at least one eager row with fused_available=True.",
        )
        self.assertTrue(
            any(r["fused_available"] == "True" for r in graph_rows),
            "Expected at least one graph row with fused_available=True.",
        )

        large_eager_rows = [
            r for r in eager_rows if int(r["bytes_per_rank"]) > 64 * 1024 * 1024
        ]
        self.assertTrue(
            any(r["fused_available"] == "False" for r in large_eager_rows),
            "Expected fused fallback for oversized eager shape(s) under default gate.",
        )

    def test_fused_ar_rms_multi_hidden_dim(self):
        """Correctness across hidden_dims from various models (TP=4)."""
        nproc = min(self._gpu_count(), 4)
        if nproc < 2:
            self.skipTest("This test requires at least 2 GPUs.")

        # hidden_dims: 2880 (GPT-OSS), 4096 (Qwen3.5), 5120, 6144 (Mixtral),
        # 7168 (DeepSeek), 8192 (Llama-70B)
        decode = ",".join(f"{m}x{n}" for n in HIDDEN_DIMS for m in [1, 4, 16])
        prefill = ",".join(f"128x{n}" for n in HIDDEN_DIMS)

        rows = self._run_benchmark(
            nproc=nproc,
            prefill_shapes=prefill,
            decode_shapes=decode,
        )

        self._assert_correctness(rows)

        fused_rows = [r for r in rows if r["fused_available"] == "True"]
        self.assertEqual(
            len(fused_rows),
            len(rows),
            f"Expected fused available for all shapes, but {len(rows) - len(fused_rows)} "
            f"rows were not fused: "
            f"{[r['shape'] for r in rows if r['fused_available'] != 'True']}",
        )

    def test_fused_ar_rms_residual_accuracy(self):
        """Residual accuracy within 1 bf16 ULP across 1-stage and 2-stage paths.

        Regression test for ROCm/aiter#2586.  The fused kernel must round the
        allreduce result to bf16 before adding residual (fix: 43b7379b8 in aiter).
        Tolerance is 1 bf16 ULP (atol=0.13) to accommodate prebuilt CI images
        that may predate the fix; multi-ULP divergence indicates a regression.
        Launches this file itself via torchrun with --residual-accuracy.
        """
        nproc = min(self._gpu_count(), 4)
        if nproc < 2:
            self.skipTest("This test requires at least 2 GPUs.")

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            __file__,
            "--residual-accuracy",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[3]),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            self.fail(
                "Residual accuracy check failed.\n"
                f"Return code: {result.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{result.stdout}"
            )

        self.assertIn(
            "ALL PASSED",
            result.stdout,
            f"Expected 'ALL PASSED' in output, got:\n{result.stdout}",
        )

    def test_fused_ar_rmsnorm_mxfp4_quant_accuracy(self):
        """SGLang API coverage for fused AR+RMSNorm+MXFP4 quantization."""
        nproc = min(self._gpu_count(), 4)
        if nproc < 2:
            self.skipTest("This test requires at least 2 GPUs.")
        if not self._is_gfx950():
            self.skipTest("The fused MXFP4 path is enabled only on gfx950.")

        try:
            import aiter  # noqa: F401
            from aiter.ops.triton.quant import dynamic_mxfp4_quant  # noqa: F401
            from aiter.utility.fp4_utils import mxfp4_to_f32  # noqa: F401
        except Exception as exc:
            self.skipTest(f"AITER MXFP4 helpers are unavailable: {exc}")

        for tp_size in (2, 4, 8):
            if self._gpu_count() < tp_size:
                continue
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={tp_size}",
                __file__,
                "--mxfp4-accuracy",
            ]

            env = os.environ.copy()
            env["SGLANG_USE_AITER_AR"] = "true"
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[3]),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                self.fail(
                    f"MXFP4 fused allreduce accuracy/perf check failed for TP={tp_size}.\n"
                    f"Return code: {result.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{result.stdout}"
                )

            self.assertIn(
                "ALL PASSED",
                result.stdout,
                f"Expected 'ALL PASSED' for TP={tp_size} in output, got:\n{result.stdout}",
            )


if __name__ == "__main__":
    if "--residual-accuracy" in sys.argv:
        _run_residual_accuracy_check()
    elif "--mxfp4-accuracy" in sys.argv:
        _run_mxfp4_accuracy_check()
    else:
        unittest.main()
