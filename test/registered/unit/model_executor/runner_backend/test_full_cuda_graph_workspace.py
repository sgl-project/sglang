"""Real CUDA regression for cuBLAS workspace lifetime across graph captures.

One GPU per process, serial capture sessions. This does not qualify concurrent
multi-device capture or graphs created by unrelated graph backends.
"""

import contextlib
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphRegistry,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_backend(pool, stream, reuse_output=False):
    # Keep real CUDA capture/replay; omit model/distributed initialization.
    backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._pool = pool
    backend._capture_stream = stream
    backend._precarve = SimpleNamespace(
        measure=contextlib.nullcontext, mint=lambda: None
    )
    backend._reuse_output_buffer = reuse_output
    backend._output_buffer = None
    backend._memory_saver_adapter = None
    backend._cuda_graph_runner = SimpleNamespace(enable_profile_cuda_graph=False)
    backend._device_module = torch.cuda
    backend._tp_group = SimpleNamespace(barrier=lambda: None)
    return backend


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFullCudaGraphWorkspace(CustomTestCase):
    @mock.patch(
        "sglang.srt.model_executor.runner_utils.pool.graph_pool_borrow_enabled",
        return_value=False,
    )
    def test_breakable_then_full_capture_preserves_workspace(self, _borrow):
        if torch.version.cuda is None or not torch.cuda.is_bf16_supported():
            self.skipTest("NVIDIA CUDA with BF16 required")
        for dedup in (False, True):
            with self.subTest(dedup=dedup):
                self._run_mixed_case(dedup)

    def _run_mixed_case(self, dedup):
        torch.manual_seed(42)
        a = torch.randn(12, 6144, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(160, 6144, device="cuda", dtype=torch.bfloat16).t()
        stream = torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()
        registry = DedupedCudaGraphRegistry() if dedup else None
        captures = {}
        full = make_backend(pool, stream)
        self.addCleanup(full.cleanup)

        def cleanup_bcg():
            if registry is not None:
                registry.close()
            for bcg, _, _ in captures.values():
                if registry is None:
                    for segment in bcg._segments:
                        segment.reset()
                bcg._segments.clear()
                bcg._break_fns.clear()
            captures.clear()

        self.addCleanup(cleanup_bcg)

        @eager_on_graph
        def eager_gemm(inputs):
            return inputs @ b

        stream.wait_stream(torch.cuda.current_stream())
        capture_seconds = []
        with torch.cuda.stream(stream):
            for n in (12, 6):
                for _ in range(2):
                    warm = a[:n] @ b
                bcg = BreakableCUDAGraph(registry)
                # Register cleanup before capture so partial failures clean up.
                captures[n] = (bcg, None, None)
                started = time.perf_counter()
                with BreakableCUDAGraphCapture(bcg, pool=pool, stream=stream):
                    first = a[:n] @ b
                    bridge = eager_gemm(a[:n])
                    last = (a[:n] @ b) + bridge
                capture_seconds.append(time.perf_counter() - started)
                captures[n] = (bcg, first, last)
            if registry is not None:
                captured, executables = registry.stats()
                self.assertEqual(captured, 4)
                self.assertLess(
                    executables,
                    captured,
                    "No executable reuse: this run does not cover dedup updates",
                )
                registry.seal()
            for n in (12, 6):
                full.capture_one(ShapeKey(size=n), lambda n=n: a[:n] @ b)
            temporary = torch.cuda.CUDAGraph()
            with torch.cuda.graph(temporary, stream=stream):
                scratch = a + 1
            temporary.reset()
            del scratch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for n in (6, 12, 6, 12, 6):
            a.mul_(0.5)
            full_out = full.replay(ShapeKey(size=n), static_forward_batch=None)
            torch.cuda.synchronize()
            expected = a[:n] @ b
            torch.testing.assert_close(full_out, expected, atol=0, rtol=0)
            bcg, first, last = captures[n]
            bcg.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(first, expected, atol=0, rtol=0)
            torch.testing.assert_close(last, expected + expected, atol=0, rtol=0)
        print(
            {
                "dedup": dedup,
                "registry_stats": registry.stats() if registry is not None else None,
                "bcg_capture_seconds": capture_seconds,
                "note": "Capture timing only; not a matched overhead comparison",
            }
        )
        cleanup_bcg()
        full.cleanup()

    @mock.patch(
        "sglang.srt.model_executor.runner_utils.pool.graph_pool_borrow_enabled",
        return_value=False,
    )
    def test_workspace_survives_interleaved_captures_and_reset(self, _borrow):
        if torch.version.cuda is None or not torch.cuda.is_bf16_supported():
            self.skipTest("NVIDIA CUDA with BF16 required")
        for dual_stream in (False, True):
            with self.subTest(dual_stream=dual_stream):
                self._run_case(dual_stream)

    def _run_case(self, dual_stream):
        torch.manual_seed(42)
        options = dict(device="cuda", dtype=torch.bfloat16)
        wq = torch.randn(4096, 2048, **options) * 0.01
        wk = torch.randn(160, 6144, **options) * 0.01
        q = torch.randn(4096, 2048, **options) * 0.01
        x = torch.randn(4096, 6144, **options) * 0.01
        stream, alt = torch.cuda.Stream(), torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()
        decode = make_backend(pool, stream)
        prefill = make_backend(pool, stream, reuse_output=True)
        # unittest cleanups also run after an ordinary assertion failure.
        # cleanup() is idempotent, so normal-path cleanup below remains safe.
        self.addCleanup(decode.cleanup)
        self.addCleanup(prefill.cleanup)

        def forward(n):
            current = torch.cuda.current_stream()
            if dual_stream:
                alt.wait_stream(current)
                with torch.cuda.stream(alt):
                    qo = torch.mm(q[:n], wq.t()) + 1
                ko = torch.mm(x[:n], wk.t()) + 1
                current.wait_stream(alt)
            else:
                qo = torch.mm(q[:n], wq.t()) + 1
                ko = torch.mm(x[:n], wk.t()) + 1
            return qo, ko

        stream.wait_stream(torch.cuda.current_stream())
        # Warmup MUST use the capture stream: warming a different stream
        # fails to reproduce the cached eager workspace lifetime hazard.
        with torch.cuda.stream(stream):
            for n in (48, 42, 36, 30, 24, 18, 12, 6):
                decode.capture_one(ShapeKey(size=n), lambda n=n: forward(n))
            for n in (128, 64):
                prefill.capture_one(ShapeKey(size=n), lambda n=n: forward(n))
            forward(4096)  # Eager prefill between capture and replay.
            temporary = torch.cuda.CUDAGraph()
            with torch.cuda.graph(temporary, stream=stream):
                scratch = x + 1
            temporary.reset()
            del scratch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Update inputs to reject accidental comparisons against stale output.
        q.mul_(0.5)
        x.mul_(0.5)
        for backend, n in (
            (decode, 6),
            (prefill, 64),
            (decode, 48),
            (decode, 12),
            (prefill, 128),
            (decode, 36),
            (decode, 6),
            (decode, 24),
            (decode, 42),
            (decode, 18),
            (decode, 30),
            (decode, 6),
        ):
            actual = backend.replay(ShapeKey(size=n), static_forward_batch=None)
            torch.cuda.synchronize()
            expected = (torch.mm(q[:n], wq.t()) + 1, torch.mm(x[:n], wk.t()) + 1)
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, atol=0, rtol=0)
        torch.cuda.synchronize()
        prefill.cleanup()
        decode.cleanup()


if __name__ == "__main__":
    unittest.main()
