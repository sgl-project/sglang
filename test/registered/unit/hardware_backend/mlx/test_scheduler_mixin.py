"""Unit tests for the MLX overlap scheduler mixin (hardware_backend/mlx/scheduler_mixin.py).

Covers:
  - Every MLX launch advances forward_ct and stamps forward_iter/launch_ts.
  - The profiler predicate runs before the async forward is enqueued, matching
    Scheduler.run_batch() so step-bounded profiling stops on the right step.

Skips on non-Apple-Silicon platforms and when ``mlx`` is missing (importing
scheduler_mixin requires ``mlx.core``).
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires Apple Silicon and mlx"


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxLaunchBookkeeping(unittest.TestCase):
    """run_batch-style bookkeeping for the MLX overlap loop."""

    def _make_scheduler(self):
        scheduler = MagicMock()
        scheduler.forward_ct = 0
        result = MagicMock()
        result.next_token_ids = None
        scheduler.tp_worker.finalize_mlx_result.return_value = result
        return scheduler

    def test_prepare_launch_advances_forward_ct_and_runs_predicate(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()
        batch = MagicMock()

        SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, batch)

        self.assertEqual(scheduler.forward_ct, 1)
        self.assertEqual(batch.forward_iter, 1)
        self.assertIsInstance(batch.launch_ts, float)
        scheduler.profiler_manager._profile_batch_predicate.assert_called_once_with(
            batch
        )

    def test_forward_ct_advances_once_per_launch(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()

        for expected in (1, 2, 3):
            SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, MagicMock())
            self.assertEqual(scheduler.forward_ct, expected)

        self.assertEqual(
            scheduler.profiler_manager._profile_batch_predicate.call_count, 3
        )

    def test_finalize_does_not_double_count_launch(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = self._make_scheduler()
        pending = MagicMock()

        SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, pending.batch_copy)
        SchedulerMlxOverlapMixin._finalize_mlx_pending_job(scheduler, pending)

        self.assertEqual(scheduler.forward_ct, 1)
        self.assertEqual(pending.batch_copy.forward_iter, 1)
        scheduler.process_batch_result.assert_called_once()


class _StopLoop(Exception):
    """Sentinel to break out of the event loop's ``while True``."""


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxChainedDecodeAllocation(CustomTestCase):
    def setUp(self):
        from sglang.srt.runtime_context import restore_context, snapshot_context

        self.addCleanup(restore_context, snapshot_context())

    def _make_batch(self, *, capacity=16):
        from array import array
        from types import SimpleNamespace

        import torch

        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.mem_cache.radix_cache import RadixCache
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        pool = ReqToTokenPool(
            size=1, max_context_len=32, device="cpu", enable_memory_saver=False
        )
        allocator = TokenToKVPoolAllocator(
            size=capacity,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
        req = Req(
            rid="decode",
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=SamplingParams(max_new_tokens=8),
        )
        pool.alloc([req])
        req.output_ids.append(2)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 1
        pool.req_to_token.fill_(-1)
        pool.write((req.kv.req_pool_idx, slice(0, 1)), allocator.alloc(1))
        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            model_config=SimpleNamespace(is_encoder_decoder=False),
            device="cpu",
            spec_algorithm=SpeculativeAlgorithm.NONE,
            sampling_info=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
            req_pool_indices=torch.tensor([req.kv.req_pool_idx]),
            req_pool_indices_cpu=torch.tensor([req.kv.req_pool_idx]),
            seq_lens=torch.tensor([1]),
            seq_lens_cpu=torch.tensor([1]),
            orig_seq_lens=torch.tensor([1]),
            input_ids=torch.tensor([2]),
        )
        batch.prepare_for_decode()
        return batch

    def test_chained_steps_allocate_slots_and_advance_live_batch(self):
        """Chained forwards must leave reusable KV for every decoded position."""
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxLaunch

        batch = self._make_batch()
        scheduler = TestOverlapLoopStampsLaunchTs._make_scheduler(
            recv_side_effect=[[], [], [], _StopLoop()]
        )
        scheduler.get_next_batch_to_run.return_value.batch_to_run = batch
        launch = MlxLaunch(
            lazy_tokens=None, prefills=[], extends=[], decode=MagicMock(), mode="decode"
        )
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = launch
        scheduler.tp_worker.async_chained_decode_mlx.return_value = launch
        with self.assertRaises(_StopLoop):
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(scheduler.forward_ct, 3)
        self.assertEqual(batch.seq_lens.tolist(), [4])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [4])
        self.assertEqual(batch.orig_seq_lens.tolist(), [4])
        req = batch.reqs[0]
        self.assertEqual(req.kv.kv_committed_len, 4)
        self.assertEqual(req.kv.kv_allocated_len, 4)
        slots = batch.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :4]
        self.assertTrue((slots > 0).all())
        self.assertEqual(slots.unique().numel(), 4)
        self.assertEqual(batch.token_to_kv_pool_allocator.available_size(), 12)

    def test_full_pool_breaks_chain_before_allocating(self):
        """A full KV pool must return control to the scheduler's retraction path."""
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxLaunch

        batch = self._make_batch(capacity=2)
        scheduler = TestOverlapLoopStampsLaunchTs._make_scheduler(
            recv_side_effect=[[], [], _StopLoop()]
        )
        scheduler.get_next_batch_to_run.side_effect = [
            SimpleNamespace(batch_to_run=batch, running_batch=batch),
            SimpleNamespace(batch_to_run=None, running_batch=batch),
        ]
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = MlxLaunch(
            lazy_tokens=None, prefills=[], extends=[], decode=MagicMock(), mode="decode"
        )
        with self.assertRaises(_StopLoop):
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)
        self.assertEqual(scheduler.forward_ct, 1)
        self.assertEqual(len(scheduler.result_queue), 0)
        self.assertEqual(batch.seq_lens.tolist(), [2])
        self.assertEqual(batch.token_to_kv_pool_allocator.available_size(), 0)

    def test_retraction_discards_kv_before_request_row_reuse(self):
        """Retracted native KV must not be flushed into the next owner's slots."""
        self._check_released_kv_before_row_reuse(retract=True)

    def test_released_request_discards_kv_before_request_row_reuse(self):
        """Prefill finish and abort paths release rows without a worker hook."""
        self._check_released_kv_before_row_reuse(retract=False)

    def _check_released_kv_before_row_reuse(self, *, retract):
        from types import SimpleNamespace

        import mlx.core as mx
        import torch

        from sglang.srt.hardware_backend.mlx.kv_cache import (
            ContiguousAttentionKVCache,
            MlxAttentionKVPool,
            MlxModelCacheLayout,
        )
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
        from sglang.srt.mem_cache.common import release_kv_cache
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        batch = self._make_batch()
        batch.prepare_for_decode()
        req = batch.reqs[0]
        req.output_ids.append(3)
        row = batch.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        runner = object.__new__(MlxModelRunner)
        runner.disable_radix_cache = False
        runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
            [SimpleNamespace(self_attn=object())], ["self_attn"]
        )
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=16, num_layers=1, n_kv_heads=1, head_dim=1, dtype=mx.float32
        )
        runner._req_to_token_pool = batch.req_to_token_pool
        cache = ContiguousAttentionKVCache(
            n_kv_heads=1, head_dim=1, max_seq_len=32, dtype=mx.float32
        )
        for value in (11, 22, 33):
            token = mx.full((1, 1, 1, 1), value, dtype=mx.float32)
            cache.write_token(token, -token)
        runner._req_caches = {req.rid: [cache]}
        runner._req_pool_idx = {req.rid: req.kv.req_pool_idx}
        runner._req_synced_offset = {req.rid: 1}
        runner._req_token_ids = {req.rid: [1, 2, 3]}
        runner._req_sampling = {}
        runner._cache_pool = []
        worker = object.__new__(MlxTpModelWorker)
        worker._mlx_runner = runner
        worker._mlx_active_rids = {req.rid}
        worker._mlx_active_reqs = {req.rid: (req, req.retraction_count)}

        if retract:
            batch.release_req(0, 0, offload_kv=False)
        else:
            release_kv_cache(req, batch.tree_cache, is_insert=False)
        self.assertIsNone(req.kv.req_pool_idx)
        row[:3] = torch.tensor([4, 5, 6])
        token = mx.full((2, 1, 1), 99, dtype=mx.float32)
        runner._attention_kv_pool.set_kv(0, mx.array([5, 6]), token, -token)
        worker._cleanup_stale_rids(ForwardMode.EXTEND, {"new"})
        runner.flush_all_decode_kv()

        keys, values = runner._attention_kv_pool.get_kv(0, mx.array([5, 6]))
        self.assertEqual(keys.flatten().tolist(), [99, 99])
        self.assertEqual(values.flatten().tolist(), [-99, -99])
        self.assertFalse(runner.has_request(req.rid))
        self.assertNotIn(req.rid, worker._mlx_active_reqs)
        self.assertEqual(worker._route_extend_request(req.rid, set()), "prefill")


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestOverlapLoopStampsLaunchTs(unittest.TestCase):
    """Every batch the MLX overlap loop launches must carry ``launch_ts``.

    ``Scheduler.run_batch`` stamps ``batch.launch_ts`` on every forward, and
    ``process_batch_result`` -> ``_record_step_counters`` subtracts it
    unconditionally for prefill/decode batches.  The MLX overlap loop bypasses
    ``run_batch``, so if its launch paths skip the stamp, the first real
    request's result processing raises ``TypeError: float - NoneType`` and
    kills the scheduler (health-check requests are filtered from the counters,
    which keeps ``/health_generate`` green while every real request crashes).
    """

    @staticmethod
    def _make_scheduler(*, recv_side_effect):
        from collections import deque

        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        scheduler = MagicMock()
        scheduler.forward_ct = 0
        scheduler._prepare_mlx_launch.side_effect = lambda batch: (
            SchedulerMlxOverlapMixin._prepare_mlx_launch(scheduler, batch)
        )
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        scheduler.ingest_requests.side_effect = recv_side_effect
        result = MagicMock()
        result.next_token_ids = None
        scheduler.tp_worker.finalize_mlx_result.return_value = result
        return scheduler

    def test_fresh_launch_stamps_launch_ts_before_input_resolution(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxLaunch

        scheduler = self._make_scheduler(recv_side_effect=[[], _StopLoop()])

        batch = MagicMock()
        events = []
        scheduler.profiler_manager._profile_batch_predicate.side_effect = (
            lambda _batch: events.append("profile")
        )
        launch_ts_at_copy_time = []
        batch.copy.side_effect = lambda: (
            launch_ts_at_copy_time.append(batch.launch_ts),
            MagicMock(),
        )[1]
        plan = MagicMock()
        plan.batch_to_run = batch
        scheduler.get_next_batch_to_run.return_value = plan
        scheduler.tp_worker.async_forward_batch_generation_mlx.side_effect = (
            lambda _batch: (
                events.append("forward"),
                MlxLaunch(
                    lazy_tokens=None,
                    prefills=[],
                    extends=[],
                    decode=None,
                    mode="extend",
                ),
            )[1]
        )

        with (
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.time.monotonic",
                side_effect=lambda: (events.append("launch_ts"), 1.0)[1],
            ),
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.resolve_forward_inputs",
                side_effect=lambda *_args: events.append("resolve_inputs"),
            ),
            self.assertRaises(_StopLoop),
        ):
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(events, ["launch_ts", "profile", "resolve_inputs", "forward"])
        self.assertEqual(len(launch_ts_at_copy_time), 1)
        self.assertEqual(launch_ts_at_copy_time[0], 1.0)

    def test_chained_launch_restamps_launch_ts(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxLaunch

        # Iteration 1: fresh decode launch.  Iteration 2: chain a second
        # decode on top of it.  Iteration 3: stop.
        scheduler = self._make_scheduler(recv_side_effect=[[], [], _StopLoop()])

        events = []
        req = MagicMock()
        req.finished.return_value = False
        batch = MagicMock()
        batch.reqs = [req]
        fresh_copy = MagicMock()
        chained_copy = MagicMock()
        chained_copy.launch_ts = None
        snapshots = iter((fresh_copy, chained_copy))

        def copy_batch():
            snapshot = next(snapshots)
            snapshot.launch_ts = batch.launch_ts
            return snapshot

        batch.copy.side_effect = copy_batch
        plan = MagicMock()
        plan.batch_to_run = batch
        scheduler.get_next_batch_to_run.return_value = plan

        pending_decode = MagicMock()
        scheduler.tp_worker.async_forward_batch_generation_mlx.return_value = MlxLaunch(
            lazy_tokens=MagicMock(),
            prefills=[],
            extends=[],
            decode=pending_decode,
            mode="decode",
        )
        scheduler.tp_worker.async_chained_decode_mlx.side_effect = lambda _decode: (
            events.append("chained_forward"),
            MlxLaunch(
                lazy_tokens=MagicMock(),
                prefills=[],
                extends=[],
                decode=MagicMock(),
                mode="decode",
            ),
        )[1]

        launch_times = iter((1.0, 2.0))

        def record_launch_ts():
            launch_ts = next(launch_times)
            events.append(f"launch_ts:{launch_ts}")
            return launch_ts

        with (
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.time.monotonic",
                side_effect=record_launch_ts,
            ),
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.resolve_forward_inputs"
            ),
            self.assertRaises(_StopLoop),
        ):
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        scheduler.tp_worker.async_chained_decode_mlx.assert_called_once()
        self.assertLess(events.index("launch_ts:2.0"), events.index("chained_forward"))
        self.assertEqual(chained_copy.launch_ts, 2.0)
        self.assertEqual(batch.forward_iter, 2)
        self.assertEqual(batch.launch_ts, 2.0)
        self.assertEqual(fresh_copy.launch_ts, 1.0)


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestOverlapLoopGracefulExit(unittest.TestCase):
    """The MLX overlap loop must honor ``gracefully_exit`` like the standard loops.

    ``handle_shutdown`` (ShutdownReq) only sets ``scheduler.gracefully_exit``;
    actual teardown happens after the event loop returns —
    ``run_scheduler_process``'s ``finally`` calls ``release_host_resources()``
    only once the loop breaks.  ``event_loop_normal`` and ``event_loop_overlap``
    check the flag at the top of every iteration; a loop that never checks it
    spins forever, so the TokenizerManager's shutdown path times out after its
    15 s grace period and falls back to ``kill_process_tree`` — host resources
    never get their user-space release.
    """

    def _make_scheduler(self, *, recv_side_effect):
        from collections import deque

        scheduler = MagicMock()
        scheduler.forward_ct = 0
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        # Model handle_shutdown: processing a non-empty recv batch (the
        # ShutdownReq) flips the flag; the loop must notice at the top of the
        # next iteration instead of polling forever.
        incoming = iter(recv_side_effect)

        def ingest_requests():
            reqs = next(incoming)
            if isinstance(reqs, Exception):
                raise reqs
            if reqs:
                scheduler.gracefully_exit = True

        scheduler.ingest_requests.side_effect = ingest_requests
        plan = MagicMock()
        plan.batch_to_run = None
        scheduler.get_next_batch_to_run.return_value = plan
        return scheduler

    def test_loop_exits_after_shutdown_req(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        # Iteration 1: recv the ShutdownReq stand-in (flag flips inside
        # process_input_requests).  Iteration 2 must break before polling
        # again; the sentinel raising instead means the loop never exits.
        scheduler = self._make_scheduler(recv_side_effect=[[MagicMock()], _StopLoop()])

        with patch(
            "sglang.srt.hardware_backend.mlx.scheduler_mixin.mx.synchronize"
        ) as synchronize:
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(scheduler.ingest_requests.call_count, 1)
        synchronize.assert_called_once_with()

    def test_loop_exits_when_shutdown_arrives_while_paused(self):
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        # A paused engine still recvs and processes control requests — that is
        # how unpause (and shutdown) arrive — but `continue`s past the rest of
        # the body.  The flag check must sit above the paused-continue, like in
        # event_loop_normal/event_loop_overlap, or shutdown during a pause
        # spins forever.
        scheduler = self._make_scheduler(recv_side_effect=[[MagicMock()], _StopLoop()])
        scheduler._engine_paused = True

        with patch(
            "sglang.srt.hardware_backend.mlx.scheduler_mixin.mx.synchronize"
        ) as synchronize:
            SchedulerMlxOverlapMixin.event_loop_overlap_mlx(scheduler)

        self.assertEqual(scheduler.ingest_requests.call_count, 1)
        scheduler.get_next_batch_to_run.assert_not_called()
        synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
