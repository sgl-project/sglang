"""Tests for deferred chunked-prefill aborts."""

import importlib.util
import platform
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=9, suite="base-a-test-cpu")
register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


class _FakeReq:
    """Minimal stand-in for Req: only the fields the abort paths touch."""

    def __init__(self, rid: str):
        self.rid = rid
        # Mirrors Req.kv; the abort paths read only these two predicates.
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.to_finish = None
        self._finished = False
        self.time_stats = SimpleNamespace(trace_ctx=Mock())

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.chunked_req = chunked_req
    sched._pending_chunked_abort_req = pending_req
    sched.waiting_queue = []
    sched.dllm_config = None
    sched.grammar_manager = Mock()
    sched.grammar_manager.grammar_queue = []
    sched.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
    sched.disaggregation_mode = None
    sched.enable_overlap = False
    sched.enable_hisparse = False
    sched.enable_hierarchical_cache = False
    sched.enable_hicache_storage = False
    sched.mm_receiver = None
    sched.tree_cache = Mock()
    sched.ps = SimpleNamespace(pp_size=1)
    sched.running_batch = SimpleNamespace(
        reqs=running_reqs, is_empty=lambda: not running_reqs
    )
    sched.last_batch = None
    return sched


class TestPendingChunkedAbortRace(CustomTestCase):
    def test_req_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        sched.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(sched._pending_chunked_abort_req)

    def test_finished_req_only_clears_marker(self):
        req = _FakeReq("done_rid")
        req._finished = True
        sched = _make_scheduler(req, chunked_req=None, running_reqs=[])

        sched.process_pending_chunked_abort()

        self.assertIsNone(req.to_finish)
        self.assertIsNone(sched._pending_chunked_abort_req)

    @unittest.skipUnless(
        _IS_APPLE_SILICON and _HAS_MLX, "requires Apple Silicon and mlx"
    )
    def test_last_chunked_abort_releases_mlx_state_at_non_overlap_idle(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        req = _FakeReq("last_chunk")
        scheduler = _make_scheduler(req, chunked_req=req, running_reqs=[])
        cache = [object()]
        runner = object.__new__(MlxModelRunner)
        runner.disable_radix_cache = True
        runner._cache_layout = SimpleNamespace(has_auxiliary_state=False)
        runner._cache_pool = []
        runner._req_caches = {req.rid: cache}
        runner._req_token_ids = {req.rid: [1]}
        runner._req_sampling = {req.rid: object()}
        runner._req_penalty_counts = {req.rid: mx.array([0, 1], dtype=mx.uint32)}
        runner._req_penalty_seed_ids = {req.rid: [1]}
        runner._req_pool_idx = {req.rid: 7}
        runner._req_synced_offset = {req.rid: 1}
        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = runner
        worker._mlx_active_rids = {req.rid}
        worker._mlx_active_reqs = {req.rid: (req, None, 0)}
        scheduler.tp_worker = worker
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.ingest_requests = Mock(side_effect=[None, RuntimeError("stop")])
        scheduler.on_idle = Mock()
        scheduler._release_aborted_request = Mock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )

        def abort_then_idle(*, running_batch, last_batch):
            scheduler.process_pending_chunked_abort()
            return SimpleNamespace(running_batch=running_batch, batch_to_run=None)

        scheduler.get_next_batch_to_run = Mock(side_effect=abort_then_idle)
        self.assertFalse(scheduler.is_fully_idle())
        self.assertTrue(runner.has_request(req.rid))
        self.assertEqual(worker._mlx_active_rids, {req.rid})

        with (
            patch(
                "sglang.srt.managers.scheduler.prepare_abort",
                side_effect=lambda aborted, _reason: setattr(
                    aborted, "_finished", True
                ),
            ),
            patch("sglang.srt.managers.scheduler.release_kv_cache"),
            patch(
                "sglang.srt.managers.scheduler._make_abort_req", return_value=object()
            ),
            patch("sglang.srt.managers.scheduler.use_mlx", return_value=True),
            self.assertRaisesRegex(RuntimeError, "stop"),
        ):
            Scheduler.event_loop_normal(scheduler)

        self.assertIsNone(scheduler.chunked_req)
        self.assertTrue(scheduler.is_fully_idle())
        self.assertEqual(worker._mlx_active_rids, set())
        self.assertEqual(worker._mlx_active_reqs, {})
        self.assertEqual(runner._req_caches, {})
        self.assertEqual(runner._req_token_ids, {})
        self.assertEqual(runner._req_sampling, {})
        self.assertEqual(runner._req_penalty_counts, {})
        self.assertEqual(runner._req_penalty_seed_ids, {})
        self.assertEqual(runner._req_pool_idx, {})
        self.assertEqual(runner._req_synced_offset, {})
        self.assertEqual(len(runner._cache_pool), 1)
        self.assertIs(runner._cache_pool[0], cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
