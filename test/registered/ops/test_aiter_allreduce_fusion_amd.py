import csv
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=240, suite="stage-c-test-large-8-gpu-amd")

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


class TestAiterAllreduceFusionAmd(unittest.TestCase):

    @staticmethod
    def _gpu_count():
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

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


def _fake_self(*, mlp_mode=ScatterMode.TP_ATTN_FULL, is_last_layer=False, tp_size=8):
    """Minimal stand-in for a LayerCommunicator with the fields the gate reads."""
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=mlp_mode),
        is_last_layer=is_last_layer,
        _context=types.SimpleNamespace(tp_size=tp_size),
    )


def _fake_forward_batch(batch_size=8):
    return types.SimpleNamespace(input_ids=types.SimpleNamespace(shape=(batch_size,)))


class TestAiterAllreduceFusionGate(CustomTestCase):
    """Pure-logic coverage of the aiter all-reduce + RMSNorm fusion gate.

    Covers ``LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer``,
    specifically the AMD/aiter branch guards that disable the fused path under
    DP attention or an expert-parallel A2A backend (e.g. mori). Without those
    guards the fused custom all-reduce is invoked during CUDA graph capture in
    those configs and crashes in ``custom_all_reduce.flush_graph_buffers``.

    The gate is pure decision logic, so the test stubs out the module-level
    dependencies and invokes the method on a minimal fake instance. No GPU or
    distributed initialization is required.
    """

    def _evaluate_gate(
        self,
        *,
        dp_attention,
        a2a_is_none,
        aiter_enabled=True,
        use_aiter=True,
        tp_world_size=8,
        mlp_mode=ScatterMode.TP_ATTN_FULL,
        is_last_layer=False,
        tp_size=8,
    ):
        """Run the gate with the aiter branch isolated (flashinfer forced off)."""
        server_args = types.SimpleNamespace(enable_aiter_allreduce_fusion=aiter_enabled)
        a2a_backend = types.SimpleNamespace(is_none=lambda: a2a_is_none)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(comm, "is_enable_moe_cp_allgather", lambda: False)
            )
            stack.enter_context(
                mock.patch.object(
                    comm,
                    "get_attn_tp_context",
                    lambda: types.SimpleNamespace(input_scattered=False),
                )
            )
            # Force the NVIDIA/flashinfer term off so the aiter branch decides.
            stack.enter_context(
                mock.patch.object(
                    comm, "apply_flashinfer_allreduce_fusion", lambda batch_size: False
                )
            )
            stack.enter_context(mock.patch.object(comm, "_use_aiter", use_aiter))
            stack.enter_context(
                mock.patch.object(
                    comm,
                    "get_parallel",
                    lambda: types.SimpleNamespace(tp_size=tp_world_size),
                )
            )
            stack.enter_context(
                mock.patch.object(comm, "get_server_args", lambda: server_args)
            )
            from sglang.srt.runtime_context import get_flags

            stack.enter_context(get_flags().dp.override(enabled=dp_attention))
            stack.enter_context(
                mock.patch.object(comm, "get_moe_a2a_backend", lambda: a2a_backend)
            )

            fake_self = _fake_self(
                mlp_mode=mlp_mode, is_last_layer=is_last_layer, tp_size=tp_size
            )
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                fake_self, _fake_forward_batch()
            )

    def test_dense_tp_fuses(self):
        # Baseline supported path: dense TP, no DP attention, no EP backend.
        self.assertTrue(self._evaluate_gate(dp_attention=False, a2a_is_none=True))

    def test_dp_attention_disables_fusion(self):
        # The fix: DP attention has no dense TP all-reduce to fuse.
        self.assertFalse(self._evaluate_gate(dp_attention=True, a2a_is_none=True))

    def test_ep_backend_disables_fusion(self):
        # The fix: with an EP A2A backend (e.g. mori) the reduction lives in
        # combine(), not a TP all-reduce.
        self.assertFalse(self._evaluate_gate(dp_attention=False, a2a_is_none=False))

    def test_dp_attention_and_ep_disables_fusion(self):
        # The crashing config from the TP8+EP8+mori repro.
        self.assertFalse(self._evaluate_gate(dp_attention=False, a2a_is_none=False))
        self.assertFalse(self._evaluate_gate(dp_attention=True, a2a_is_none=False))

    def test_flag_off_disables_fusion(self):
        # Sanity: the gate still respects the opt-in flag on the dense path.
        self.assertFalse(
            self._evaluate_gate(
                dp_attention=False, a2a_is_none=True, aiter_enabled=False
            )
        )

    def test_last_layer_disables_fusion(self):
        self.assertFalse(
            self._evaluate_gate(
                dp_attention=False, a2a_is_none=True, is_last_layer=True
            )
        )

    def test_tp1_disables_fusion(self):
        self.assertFalse(
            self._evaluate_gate(dp_attention=False, a2a_is_none=True, tp_size=1)
        )


if __name__ == "__main__":
    if "--residual-accuracy" in sys.argv:
        _run_residual_accuracy_check()
    else:
        unittest.main()
