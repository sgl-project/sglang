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
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

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
        self._invalidated = set()
        self.removed: list[str] = []
        self.aux_stored: list[str] = []
        self.synced: list[str] = []

    def has_request(self, rid):
        return rid in self._known and rid not in self._invalidated

    def invalidate_request(self, rid):
        if rid in self._known:
            self._invalidated.add(rid)

    def remove_request(self, rid):
        self.removed.append(rid)
        self._known.discard(rid)
        self._invalidated.discard(rid)

    def store_auxiliary_state_for_request(self, rid):
        self.aux_stored.append(rid)

    def sync_and_release_request(self, rid):
        self.synced.append(rid)


def _make_worker(known_rids):
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

    worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
    worker._mlx_runner = _FakeRunner(known_rids)
    worker._mlx_active_rids = set(known_rids)
    worker._mlx_released_rids = {}
    worker._mlx_retracted_rids = set()
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

        worker._cleanup_stale_rids(ForwardMode.EXTEND, [SimpleNamespace(rid="B")])

        self.assertEqual(worker._mlx_runner.removed, ["A"])
        self.assertNotIn("A", worker._mlx_active_rids)
        self.assertIn("B", worker._mlx_active_rids)
        self.assertEqual(worker._mlx_released_rids, {})

    def test_release_is_deferred_not_eager(self):
        # Removing inside prepare_for_kv_cache_release would break overlap:
        # a chained pending job launched before the finish was known may
        # still reference the request's caches and token list.
        worker = _make_worker({"A"})
        req = _finish(worker, "A")

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertTrue(worker._mlx_runner.has_request("A"))
        self.assertIs(worker._mlx_released_rids["A"], req)
        # Existing contract of the hook is preserved.
        self.assertEqual(worker._mlx_runner.aux_stored, ["A"])
        self.assertEqual(worker._mlx_runner.synced, ["A"])
        self.assertIsNone(req.mamba_last_track_seqlen)

    def test_marked_rid_still_in_batch_is_not_released(self):
        # Safety: never remove state for a request that is part of the
        # forward being launched.
        worker = _make_worker({"A"})
        req = _finish(worker, "A")

        worker._cleanup_stale_rids(ForwardMode.EXTEND, [req])

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertIn("A", worker._mlx_released_rids)

    def test_unmarked_active_requests_survive_extend_batches(self):
        # A chunked-prefill continuation or still-decoding request (not
        # marked released) must not be dropped by an unrelated prefill.
        worker = _make_worker({"A"})

        worker._cleanup_stale_rids(ForwardMode.EXTEND, [SimpleNamespace(rid="B")])

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertEqual(worker._mlx_active_rids, {"A", "B"})

    def test_decode_branch_still_drops_stale_and_purges_marks(self):
        # Existing decode-branch behavior is unchanged, and released marks
        # for rids dropped there don't linger.
        worker = _make_worker({"A", "B"})
        _finish(worker, "A")

        worker._cleanup_stale_rids(ForwardMode.DECODE, [SimpleNamespace(rid="B")])

        self.assertEqual(worker._mlx_runner.removed, ["A"])
        self.assertEqual(worker._mlx_active_rids, {"B"})
        self.assertEqual(worker._mlx_released_rids, {})

    def test_same_rid_resubmit_retires_old_incarnation_before_routing(self):
        worker = _make_worker({"A"})
        old_req = _finish(worker, "A")
        new_req = SimpleNamespace(
            rid="A", prefix_indices=SimpleNamespace(tolist=lambda: [7, 8])
        )

        worker._cleanup_stale_rids(ForwardMode.EXTEND, [new_req])

        self.assertEqual(worker._mlx_runner.removed, ["A"])
        self.assertIsNot(old_req, new_req)
        self.assertEqual(worker._mlx_released_rids, {})
        self.assertEqual(worker._route_extend_request("A", set()), "prefill")
        self.assertEqual(worker._gather_prefill_prefix_slots([new_req]), {7, 8})

    def test_deferred_retraction_hides_then_retires_same_req_incarnation(self):
        worker = _make_worker({"A"})
        req = SimpleNamespace(
            rid="A", prefix_indices=SimpleNamespace(tolist=lambda: [7, 8])
        )

        worker.defer_retraction(req)

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertFalse(worker._mlx_runner.has_request("A"))
        self.assertIn("A", worker._mlx_retracted_rids)

        # Unlike a normal-finish marker, a retracted request must be retired
        # even when the scheduler resumes the same Req object.
        worker._cleanup_stale_rids(ForwardMode.EXTEND, [req])

        self.assertEqual(worker._mlx_runner.removed, ["A"])
        self.assertEqual(worker._mlx_retracted_rids, set())
        self.assertEqual(worker._route_extend_request("A", set()), "prefill")
        self.assertEqual(worker._gather_prefill_prefix_slots([req]), {7, 8})


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


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestSchedulerMlxRetractionPaths(unittest.TestCase):
    """Every scheduler release/requeue path must retire MLX private state."""

    @staticmethod
    def _fresh_req(rid):
        return SimpleNamespace(
            rid=rid,
            req_pool_idx=1,
            prefix_indices=SimpleNamespace(tolist=lambda: [7, 8]),
            to_finish=None,
            finished=lambda: False,
            output_ids=[],
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(abort=lambda **_kwargs: None)
            ),
        )

    def test_chunked_abort_retires_state_before_kv_release_and_reuse(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import Scheduler

        worker = _make_worker({"A"})
        req = self._fresh_req("A")
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = worker
        scheduler._pending_chunked_abort_req = req
        scheduler.chunked_req = req
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = object()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=lambda *_args: None)
        )

        events = []
        original_remove = worker._mlx_runner.remove_request

        def remove(rid):
            events.append(f"remove:{rid}")
            original_remove(rid)

        worker._mlx_runner.remove_request = remove
        with (
            patch("sglang.srt.managers.scheduler.prepare_abort"),
            patch(
                "sglang.srt.managers.scheduler.release_kv_cache",
                side_effect=lambda *_args, **_kwargs: events.append("release-kv"),
            ),
        ):
            scheduler.process_pending_chunked_abort()

        self.assertEqual(events, ["remove:A", "release-kv"])
        self.assertFalse(worker._mlx_runner.has_request("A"))
        # The old state cannot participate in a later prefix read after its KV
        # slots are reused by another request.
        new_req = self._fresh_req("A")
        self.assertEqual(worker._route_extend_request("A", set()), "prefill")
        self.assertEqual(worker._gather_prefill_prefix_slots([new_req]), {7, 8})

    def test_pause_retract_quiesced_path_resumes_as_fresh_prefill(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import Scheduler

        worker = _make_worker({"A"})
        req = self._fresh_req("A")
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = worker
        scheduler.enable_overlap = False
        scheduler.last_batch = None
        scheduler.running_batch = SimpleNamespace(reqs=[req], batch_is_full=True)
        scheduler.chunked_req = None
        scheduler.cur_batch_for_debug = object()
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.server_args = SimpleNamespace()
        scheduler.req_to_token_pool = object()
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.tree_cache = object()
        scheduler.hisparse_coordinator = None
        scheduler.metrics_reporter = SimpleNamespace(
            last_gen_throughput=1.0,
            current_scheduler_metrics_enabled=False,
        )
        scheduler.kv_events_publisher = SimpleNamespace(publish_kv_events=lambda: None)
        resumed = []
        scheduler._add_request_to_queue = resumed.append

        def retract_after_mlx_is_quiesced(**kwargs):
            kwargs["prepare_for_retraction"](req)
            self.assertFalse(worker._mlx_runner.has_request("A"))

        with patch(
            "sglang.srt.managers.scheduler.retract_all",
            side_effect=retract_after_mlx_is_quiesced,
        ):
            scheduler.pause_generation(SimpleNamespace(mode="retract"))

        self.assertEqual(resumed, [req])
        self.assertEqual(worker._route_extend_request("A", set()), "prefill")
        self.assertEqual(worker._gather_prefill_prefix_slots(resumed), {7, 8})

    def test_pause_drain_finalizes_all_mlx_jobs_before_cache_reuse(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = SchedulerMlxOverlapMixin()
        pending_curr = object()
        pending_next = object()
        scheduler.result_queue = deque([pending_curr, pending_next])
        finalized = []
        scheduler._finalize_mlx_pending_job = finalized.append

        remaining = scheduler._drain_mlx_pending_jobs_for_pause(
            pending_curr, pending_next
        )

        self.assertEqual(finalized, [pending_curr, pending_next])
        self.assertEqual(remaining, (None, None))
        self.assertEqual(list(scheduler.result_queue), [])

    def test_priority_preempt_retires_state_at_release_then_resumes(self):
        from sglang.srt.managers.schedule_batch import release_req

        worker = _make_worker({"A"})
        req = self._fresh_req("A")
        req.reset_for_retract = lambda: None
        events = []
        original_remove = worker._mlx_runner.remove_request

        def remove(rid):
            events.append(f"remove:{rid}")
            original_remove(rid)

        worker._mlx_runner.remove_request = remove
        with (
            patch(
                "sglang.srt.managers.schedule_batch.release_kv_cache",
                side_effect=lambda *_args, **_kwargs: events.append("release-kv"),
            ),
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            release_req(
                req=req,
                remaing_req_count=0,
                server_args=SimpleNamespace(disaggregation_mode="null"),
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=object(),
                tree_cache=object(),
                hisparse_coordinator=None,
                prepare_for_retraction=worker.prepare_for_retraction,
            )

        self.assertEqual(events, ["remove:A", "release-kv"])
        self.assertEqual(worker._route_extend_request("A", set()), "prefill")
        self.assertEqual(worker._gather_prefill_prefix_slots([req]), {7, 8})

    def test_oom_release_invalidates_but_defers_physical_removal(self):
        from sglang.srt.managers.schedule_batch import release_req
        from sglang.srt.managers.scheduler import Scheduler

        worker = _make_worker({"A"})
        req = self._fresh_req("A")
        req.reset_for_retract = lambda: None
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = worker

        with (
            patch("sglang.srt.managers.schedule_batch.release_kv_cache"),
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            release_req(
                req=req,
                remaing_req_count=0,
                server_args=SimpleNamespace(disaggregation_mode="null"),
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=object(),
                tree_cache=object(),
                hisparse_coordinator=None,
                prepare_for_retraction=scheduler._defer_model_worker_retraction,
            )

        self.assertEqual(worker._mlx_runner.removed, [])
        self.assertFalse(worker._mlx_runner.has_request("A"))
        self.assertIn("A", worker._mlx_retracted_rids)

        # The next scheduler-built forward runs only after the chained pending
        # job is finalized, so cleanup can now return the cache to the pool.
        worker._cleanup_stale_rids(ForwardMode.EXTEND, [req])
        self.assertEqual(worker._mlx_runner.removed, ["A"])
        self.assertEqual(worker._route_extend_request("A", set()), "prefill")


if __name__ == "__main__":
    unittest.main()
