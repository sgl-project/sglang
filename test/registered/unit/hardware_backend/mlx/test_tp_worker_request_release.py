"""Unit tests for MLX per-request state release in ``MlxTpModelWorker``.

Regression guard for the prefill-only request-state leak:
``MlxModelRunner.remove_request`` (the only entry point that frees a
request's per-layer ``ContiguousAttentionKVCache`` list back to the reuse
pool) used to be called exclusively from ``_cleanup_stale_rids``'s
decode branch. Workloads that never form a decode batch — prefill-only
traffic such as max_new_tokens=1 classification/scoring — therefore
leaked one full per-request cache list per unique request, unbounded
(~2 * num_layers * n_kv_heads * max_seq_len * head_dim * dtype bytes
each; ~0.44 GB/request measured for Qwen3-0.6B).

The fix marks a request at its KV release point
(``prepare_for_kv_cache_release``) and releases it on the next
scheduler-built forward of either mode. Release must NOT happen inside
``prepare_for_kv_cache_release`` itself: an overlap chained decode
launched before the request was known to be finished may still reference
its caches and token list, and finalizing it after an eager removal
would KeyError.

Tests mock the MLX runner and load no model. Apple-Silicon-only because
``tp_worker`` imports ``mlx.core`` at module load.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

# CPU marker is AST-parsed "this test exists"; actual CPU-side execution is
# gated by the @skipUnless guard below. MLX marker runs for real on the MLX
# lane's stage-a (model-free: mocks the runner, loads no model).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (tp_worker imports mlx.core at module load)"


class _FakeRunner:
    """Tracks live request state and remove_request calls."""

    def __init__(self, known_rids):
        self._known = set(known_rids)
        self.removed: list[str] = []
        self.aux_stored: list[str] = []
        self.synced: list[str] = []

    def has_request(self, rid):
        return rid in self._known

    def remove_request(self, rid):
        self.removed.append(rid)
        self._known.discard(rid)

    def store_auxiliary_state_for_request(self, rid):
        self.aux_stored.append(rid)

    def sync_and_release_request(self, rid):
        self.synced.append(rid)


def _make_worker(known_rids):
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

    worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
    worker._mlx_runner = _FakeRunner(known_rids)
    worker._mlx_active_rids = set(known_rids)
    worker._mlx_released_rids = set()
    return worker


def _finish(worker, rid):
    """Drive the scheduler's per-finished-request release hook."""
    req = SimpleNamespace(rid=rid, mamba_last_track_seqlen="stale")
    worker.prepare_for_kv_cache_release(req)
    return req


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestPrefillOnlyRequestRelease(unittest.TestCase):
    """Finished requests must be released without waiting for a decode batch."""

    def test_finished_request_released_on_next_extend_batch(self):
        # Request A finishes at prefill (max_new_tokens=1); the next
        # scheduler forward is another prefill (extend mode) for request B.
        # A's state must be released there — prefill-only traffic never
        # forms a decode batch.
        worker = _make_worker({"A"})
        _finish(worker, "A")
        self.assertEqual(worker._mlx_runner.removed, [])  # deferred, not eager

        worker._cleanup_stale_rids(ForwardMode.EXTEND, {"B"})

        self.assertEqual(worker._mlx_runner.removed, ["A"])
        self.assertNotIn("A", worker._mlx_active_rids)
        self.assertIn("B", worker._mlx_active_rids)
        self.assertEqual(worker._mlx_released_rids, set())

    def test_release_is_deferred_not_eager(self):
        # Removing inside prepare_for_kv_cache_release would break overlap:
        # a chained pending job launched before the finish was known may
        # still reference the request's caches and token list.
        worker = _make_worker({"A"})
        req = _finish(worker, "A")

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertTrue(worker._mlx_runner.has_request("A"))
        self.assertIn("A", worker._mlx_released_rids)
        # Existing contract of the hook is preserved.
        self.assertEqual(worker._mlx_runner.aux_stored, ["A"])
        self.assertEqual(worker._mlx_runner.synced, ["A"])
        self.assertIsNone(req.mamba_last_track_seqlen)

    def test_marked_rid_still_in_batch_is_not_released(self):
        # Safety: never remove state for a request that is part of the
        # forward being launched.
        worker = _make_worker({"A"})
        _finish(worker, "A")

        worker._cleanup_stale_rids(ForwardMode.EXTEND, {"A"})

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertIn("A", worker._mlx_released_rids)

    def test_unmarked_active_requests_survive_extend_batches(self):
        # A chunked-prefill continuation or still-decoding request (not
        # marked released) must not be dropped by an unrelated prefill.
        worker = _make_worker({"A"})

        worker._cleanup_stale_rids(ForwardMode.EXTEND, {"B"})

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertEqual(worker._mlx_active_rids, {"A", "B"})

    def test_decode_branch_still_drops_stale_and_purges_marks(self):
        # Existing decode-branch behavior is unchanged, and released marks
        # for rids dropped there don't linger.
        worker = _make_worker({"A", "B"})
        _finish(worker, "A")

        worker._cleanup_stale_rids(ForwardMode.DECODE, {"B"})

        self.assertEqual(worker._mlx_runner.removed, ["A"])
        self.assertEqual(worker._mlx_active_rids, {"B"})
        self.assertEqual(worker._mlx_released_rids, set())


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestPrepareModelWorkerKvRelease(unittest.TestCase):
    """The result processor's pre-release hook must reach the MLX worker.

    The leak's second half: prepare_for_kv_cache_release used to be invoked
    only from the decode result path, so requests finishing at prefill (the
    prefill-only workloads that leak) never reached the worker hook at all.
    process_batch_result_prefill now routes every finished request through
    _prepare_model_worker_kv_release before release_kv_cache.
    """

    def _processor(self, model_worker):
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        # Frozen dataclass; bypass __setattr__ for the partial test object.
        object.__setattr__(processor, "model_worker", model_worker)
        return processor

    def test_helper_invokes_worker_hook(self):
        worker = _make_worker({"A"})
        processor = self._processor(worker)
        req = SimpleNamespace(rid="A", mamba_last_track_seqlen=None)

        processor._prepare_model_worker_kv_release(req)

        self.assertIn("A", worker._mlx_released_rids)
        self.assertEqual(worker._mlx_runner.aux_stored, ["A"])

    def test_helper_is_noop_without_hook(self):
        processor = self._processor(SimpleNamespace())  # no hook attribute
        req = SimpleNamespace(rid="A")

        processor._prepare_model_worker_kv_release(req)  # must not raise


if __name__ == "__main__":
    unittest.main()
