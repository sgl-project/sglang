"""Unit tests for the confidential-compute async D2H copy worker.

These validate that ``AsyncD2HCopyWorker`` performs the device->host readback
correctly off the calling thread, and that the CC detection env override works.

They need a GPU but NO model and NO actual confidential-compute hardware: the
worker logic is identical regardless of CC (CC only changes whether the
scheduler routes the readback through the worker). To exercise the worker on an
ordinary GPU, run inside the sglang container:

    python -m pytest test/registered/core/test_overlap_copy_worker.py -v
"""

import os
import threading
import unittest
from unittest import mock

import torch

from sglang.srt.managers.overlap_copy_worker import AsyncD2HCopyWorker
from sglang.srt.utils.common import is_confidential_compute
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "AsyncD2HCopyWorker requires CUDA")
class TestAsyncD2HCopyWorker(CustomTestCase):
    def setUp(self):
        self.copy_stream = torch.cuda.Stream()
        self.worker = AsyncD2HCopyWorker(torch.cuda, self.copy_stream)

    def tearDown(self):
        # Idempotent: a no-op if a test already shut the worker down.
        self.worker.shutdown()

    def _run_one(self, numel: int):
        src = torch.randn(numel, device="cuda")
        expected = src.detach().to("cpu")  # synchronous reference copy

        # Source is produced on the current (default) stream; record readiness
        # exactly as the scheduler does before handing work to the worker.
        src_ready = torch.cuda.Event()
        src_ready.record()

        out = {}

        def copy_fn():
            out["cpu"] = src.to("cpu", non_blocking=True)

        done = threading.Event()
        self.worker.submit(src_ready, copy_fn, done)

        self.assertTrue(done.wait(timeout=30), "worker did not signal completion")
        self.assertIn("cpu", out)
        torch.testing.assert_close(out["cpu"], expected)

    def test_single_readback_matches_synchronous(self):
        self._run_one(4)  # tiny, like next_token_ids at small batch
        self._run_one(151936)  # vocab-sized, like a logprob row

    def test_many_sequential_readbacks(self):
        # Mimics steady-state decode: one readback per "step".
        for _ in range(64):
            self._run_one(8)

    def test_shutdown_is_idempotent_and_joins(self):
        self.worker.shutdown()
        self.worker.shutdown()  # second call must be a safe no-op
        self.assertFalse(self.worker._thread.is_alive())

    def test_copy_fn_exception_still_sets_done(self):
        # A failing copy must not hang the scheduler: ``done`` is set in finally.
        src_ready = torch.cuda.Event()
        src_ready.record()
        done = threading.Event()

        def boom():
            raise RuntimeError("injected copy failure")

        self.worker.submit(src_ready, boom, done)
        self.assertTrue(
            done.wait(timeout=30), "done must be set even when the copy fails"
        )


class TestConfidentialComputeDetection(CustomTestCase):
    """CC detection env override. Does not require CUDA (override short-circuits)."""

    def test_env_override_true(self):
        is_confidential_compute.cache_clear()
        with mock.patch.dict(os.environ, {"SGLANG_CONFIDENTIAL_COMPUTE": "1"}):
            is_confidential_compute.cache_clear()
            self.assertTrue(is_confidential_compute())
        is_confidential_compute.cache_clear()

    def test_env_override_false(self):
        is_confidential_compute.cache_clear()
        with mock.patch.dict(os.environ, {"SGLANG_CONFIDENTIAL_COMPUTE": "0"}):
            is_confidential_compute.cache_clear()
            self.assertFalse(is_confidential_compute())
        is_confidential_compute.cache_clear()


if __name__ == "__main__":
    unittest.main()
