"""Unit tests for the async device->host copy worker (AsyncD2HCopyWorker).

These validate that ``AsyncD2HCopyWorker`` performs the device->host copy
correctly off the calling thread, and that the CC detection env override works.

They need a GPU but NO model and NO actual confidential-compute hardware: the
worker logic is identical regardless of CC (CC only changes whether the
scheduler routes the readback through the worker). To exercise the worker on an
ordinary GPU, run inside the sglang container:

    python -m pytest test/registered/unit/managers/test_async_d2h_copy_worker.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.async_d2h_copy_worker import AsyncD2HCopyWorker, HostCopyDone
from sglang.srt.utils.common import (
    async_d2h,
    is_confidential_compute,
    is_pin_memory_available,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "AsyncD2HCopyWorker requires CUDA")
class TestAsyncD2HCopyWorker(CustomTestCase):
    def setUp(self):
        self.worker = AsyncD2HCopyWorker(torch.cuda)

    def tearDown(self):
        # Idempotent: a no-op if a test already shut the worker down.
        self.worker.shutdown()

    def _run_one(self, numel: int):
        src = torch.randn(numel, device="cuda")
        expected = src.detach().to("cpu")  # synchronous reference copy

        out = {}

        def copy_fn():
            out["cpu"] = src.to("cpu", non_blocking=True)

        # submit() records readiness on the current stream (where src was
        # produced) before handing the copy to the worker.
        done = self.worker.submit(copy_fn)

        self.assertTrue(done._done.wait(timeout=30), "worker did not signal completion")
        done.synchronize()  # drop-in for copy_done; must not raise on success
        self.assertIn("cpu", out)
        torch.testing.assert_close(out["cpu"], expected)

    def test_single_readback_matches_synchronous(self):
        self._run_one(4)  # tiny, like next_token_ids at small batch
        self._run_one(151936)  # vocab-sized, like a logprob row

    def test_copy_waits_for_the_producing_stream(self):
        # Guards submit()'s src_ready.record(): with the fill still queued behind
        # slow work, a worker that skipped the event would read zeros.
        side_stream = torch.cuda.Stream()
        src = torch.zeros(4096, device="cuda")

        out = {}

        def copy_fn():
            out["cpu"] = src.to("cpu", non_blocking=True)

        with torch.cuda.stream(side_stream):
            torch.cuda._sleep(200_000_000)  # ~100ms at 2GHz; outlives the handoff
            src.fill_(7)
            done = self.worker.submit(copy_fn)

        self.assertTrue(done._done.wait(timeout=30), "worker did not signal completion")
        done.synchronize()
        torch.testing.assert_close(out["cpu"], torch.full((4096,), 7.0))

    def test_shutdown_is_idempotent_and_joins(self):
        self.worker.shutdown()
        self.worker.shutdown()  # second call must be a safe no-op
        self.assertFalse(self.worker._thread.is_alive())

    def test_copy_fn_exception_reports_error(self):
        # A failing copy must not hang the scheduler, but synchronize() must
        # re-raise so the consumer aborts instead of reading invalid CPU tensors.
        def boom():
            raise RuntimeError("injected copy failure")

        done = self.worker.submit(boom)
        self.assertTrue(
            done._done.wait(timeout=30), "handle must be signaled even on failure"
        )
        self.assertIsNotNone(done.error, "failed copy must record its error")
        with self.assertRaises(RuntimeError):
            done.synchronize()


class TestConfidentialComputeDetection(CustomTestCase):
    """CC detection env override. Does not require CUDA (override short-circuits)."""

    def test_env_override_true(self):
        is_confidential_compute.cache_clear()
        with envs.SGLANG_FORCE_CONFIDENTIAL_COMPUTE.override(True):
            is_confidential_compute.cache_clear()
            self.assertTrue(is_confidential_compute())
        is_confidential_compute.cache_clear()

    def test_env_override_false(self):
        is_confidential_compute.cache_clear()
        with envs.SGLANG_FORCE_CONFIDENTIAL_COMPUTE.override(False):
            is_confidential_compute.cache_clear()
            self.assertFalse(is_confidential_compute())
        is_confidential_compute.cache_clear()


SLEEP_CYCLES = 200_000_000  # ~100ms at 2GHz; outlives the handoff to the worker


def _late_source(value: int, numel: int = 4) -> torch.Tensor:
    """A device tensor that holds ``value`` only after a long kernel on this stream."""
    src = torch.zeros(numel, dtype=torch.int64, device="cuda")
    torch.cuda._sleep(SLEEP_CYCLES)
    return src.fill_(value)


@unittest.skipUnless(torch.cuda.is_available(), "pinned host memory requires CUDA")
class TestConfidentialComputeStaging(CustomTestCase):
    """Host staging is pageable under CC and pinned otherwise."""

    def setUp(self):
        is_confidential_compute.cache_clear()
        self.addCleanup(is_confidential_compute.cache_clear)

    def test_async_d2h_stages_pageable_under_cc(self):
        src = torch.arange(4, device="cuda")
        for cc in (False, True):
            with (
                self.subTest(cc=cc),
                envs.SGLANG_FORCE_CONFIDENTIAL_COMPUTE.override(cc),
            ):
                is_confidential_compute.cache_clear()
                self.assertEqual(is_pin_memory_available(src.device), not cc)
                host = async_d2h(src)
                torch.cuda.current_stream().synchronize()
                self.assertEqual(host.is_pinned(), not cc)
                self.assertEqual(host.tolist(), [0, 1, 2, 3])


@unittest.skipUnless(torch.cuda.is_available(), "the scheduler readback requires CUDA")
class TestSchedulerReadbackRouting(CustomTestCase):
    """Every readback path must wait for the forward: a copy issued too early reads
    the zeros the source held before it. Under CC only plain batches take the
    worker; hidden-state and DSpark batches stay inline, since they read CUDA
    graph buffers that the next replay rewrites."""

    def setUp(self):
        is_confidential_compute.cache_clear()
        self.addCleanup(is_confidential_compute.cache_clear)
        self.worker = AsyncD2HCopyWorker(torch.cuda)
        self.addCleanup(self.worker.shutdown)

    def _readback(self, cc: bool, hidden: bool, dspark: bool):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        forward_stream, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
        scheduler = SimpleNamespace(
            forward_stream=forward_stream,
            copy_stream=copy_stream,
            copy_stream_ctx=torch.cuda.stream(copy_stream),
            enable_async_d2h_copy=cc,
            async_d2h_worker=self.worker if cc else None,
        )
        # Distinct per case, so a reused host block cannot pass a stale value.
        value = 7 + cc + hidden + 2 * dspark
        with torch.cuda.stream(forward_stream):
            ids = _late_source(value, numel=1 << 24)  # 128 MiB: the copy takes a while
            logits = SimpleNamespace(
                hidden_states=_late_source(value) if hidden else None,
                sampling_mask_output=None,
                auxiliary_device_output=None,
            )
            result = GenerationBatchResult(
                logits_output=logits, next_token_ids=ids, copy_done=torch.cuda.Event()
            )
        algorithm = SpeculativeAlgorithm.DSPARK if dspark else SpeculativeAlgorithm.NONE
        batch = SimpleNamespace(
            return_logprob=False, return_hidden_states=hidden, spec_algorithm=algorithm
        )
        with envs.SGLANG_FORCE_CONFIDENTIAL_COMPUTE.override(cc):
            is_confidential_compute.cache_clear()
            Scheduler._copy_result_to_cpu(scheduler, result, batch)
            result.copy_done.synchronize()
        return result, logits.hidden_states, value

    def test_readback_lands_after_the_forward(self):
        cases = {
            "off-inline": (False, False, False),
            "cc-worker": (True, False, False),
            "cc-hidden-states-inline": (True, True, False),
            "cc-dspark-inline": (True, False, True),
        }
        for name, (cc, hidden, dspark) in cases.items():
            with self.subTest(name):
                result, hidden_states, value = self._readback(cc, hidden, dspark)
                offloaded = cc and not (hidden or dspark)
                self.assertEqual(isinstance(result.copy_done, HostCopyDone), offloaded)
                for host in (result.next_token_ids, hidden_states):
                    if host is None:
                        continue
                    self.assertEqual(
                        (host.device.type, host.is_pinned()), ("cpu", not cc)
                    )
                    self.assertTrue(torch.equal(host, torch.full_like(host, value)))


if __name__ == "__main__":
    unittest.main()
