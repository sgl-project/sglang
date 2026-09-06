"""Retracted requests must not survive into the next extend flush.

Regression guard for issue #33547: when the scheduler retracts a decoding
request R under memory pressure, it frees R's ``req_to_token`` row and KV
slots immediately (``release_kv_cache(..., is_insert=False)``) and requeues
R. The MLX runner's private decode cache for R stays in ``_req_caches``
until ``_cleanup_stale_rids`` disposes of it — but that cleanup only runs
on *decode* forwards. If the next forward is an *extend* (e.g. a new
prefill), ``flush_all_decode_kv`` iterates every rid in ``_req_caches``,
including the retracted R, and ``_sync_decode_kv_to_pool(R)`` reads
``req_to_token[R_row]`` — a row the scheduler may already have reused for
a new request P. The read yields P's slot ids and R's dirty decode KV is
scattered over P's pool slots, silently corrupting P's attention state.

(Naively moving ``_cleanup_stale_rids`` to the extend path is not a fix:
decode-mode rids legitimately absent from an extend batch would be
evicted too, killing live requests.)

The fix is a retract-time disposal hook: the scheduler notifies the model
worker right after ``retract_decode()``, and the MLX worker discards the
rid's runner state *without syncing* (the slots are freed; a sync would
be exactly the poisoning write above). These tests cover:

  * the end-to-end pollution scenario through the worker's extend path
    (red before the fix: R's decode KV lands in reused-row slots);
  * the hook's pure-discard semantics (no pool write, full state pop,
    idempotent);
  * requeue routing: a retracted request re-enters as a fresh prefill;
  * the scheduler-side wiring of the hook.

They drive a real ``MlxModelRunner`` state dict and a real small
``MlxAttentionKVPool`` with only the model-loading surface mocked, so no
model weights are needed. Apple-Silicon-only where ``mlx`` is imported.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (drives the real MLX runner state dicts)"


class TestSchedulerRetractWiring(CustomTestCase):
    """The scheduler must notify the worker at retract time (no mlx needed)."""

    def test_update_running_batch_notifies_worker_on_retract(self):
        from sglang.srt.managers.scheduler import Scheduler
        import inspect

        src = inspect.getsource(Scheduler.update_running_batch)
        self.assertIn(
            "prepare_for_retraction",
            src,
            "Scheduler.update_running_batch lost its retract-time disposal "
            "hook. Without it, backend-private per-request state (e.g. the "
            "MLX runner's decode caches) survives retraction and can poison "
            "a reused req_to_token row on the next extend flush "
            "(issue #33547).",
        )

    def test_mlx_worker_exposes_prepare_for_retraction(self):
        if not (_IS_APPLE_SILICON and _HAS_MLX):
            self.skipTest(_SKIP_REASON)
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        self.assertTrue(
            callable(getattr(MlxTpModelWorker, "prepare_for_retraction", None)),
            "MlxTpModelWorker must expose prepare_for_retraction for the "
            "scheduler's retract-time disposal hook.",
        )


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestRetractKvRowReuse(CustomTestCase):
    """The pollution scenario and the disposal-hook semantics."""

    # Marker values make pool writes attributable: R's dirty decode KV is
    # RETRACT_MARKER everywhere; valid pool data (P's radix prefix) is
    # OWNER_MARKER. Anything else stays zero.
    RETRACT_MARKER = 777.0
    OWNER_MARKER = 111.0

    @classmethod
    def setUpClass(cls):
        # The worker reads --mlx-enable-sampling off the device config bag,
        # which fails closed before a publish; the retract paths are
        # orthogonal to sampling.
        from sglang.srt.runtime_context import get_context

        cls._config = get_context().override_server_args(mlx_enable_sampling=False)
        cls._config.install()
        cls.addClassCleanup(cls._config.restore)

    # ---------- fixtures ----------

    @staticmethod
    def _make_runner():
        """Real MlxModelRunner state dicts + real pool; model surface mocked.

        Only the weight-loading / forward surface is faked (prefill_start,
        cache_state_arrays); every field the retract/flush paths touch
        (_req_caches, _req_pool_idx, _req_synced_offset, the pool, the
        req_to_token mirror) is the real implementation.
        """
        import mlx.core as mx
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
            MlxAttentionKVPool,
        )

        runner = object.__new__(MlxModelRunner)
        runner.disable_radix_cache = False
        runner._attention_kv_pool = MlxAttentionKVPool(
            pool_size=16, num_layers=1, n_kv_heads=2, head_dim=4, dtype=mx.float32
        )
        # Scheduler-owned req_to_token mirror: 4 rows x 64 positions.
        runner._req_to_token_pool = SimpleNamespace(
            req_to_token=torch.full((4, 64), -1, dtype=torch.int64)
        )
        runner._req_caches = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        runner._req_token_ids = {}
        runner._req_sampling = {}
        runner._cache_pool = []
        runner._cache_layout = SimpleNamespace(
            full_attention_layer_indices=[0],
            first_attention_layer_index=0,
            has_auxiliary_state=False,
        )

        def _fake_prefill_start(**kwargs):
            return SimpleNamespace(
                lazy_token=mx.array([0], dtype=mx.int32),
                cache=[TestRetractKvRowReuse._cache_layer(offset=0, value=0.0)],
                req_id=kwargs["req_id"],
                lazy_logprobs=None,
            )

        runner.prefill_start = _fake_prefill_start
        runner.cache_state_arrays = lambda caches: []
        return runner

    @staticmethod
    def _cache_layer(offset, value):
        """One attention cache adapter entry: keys/values (1, H, S, D), offset."""
        import mlx.core as mx

        return SimpleNamespace(
            keys=mx.full((1, 2, 16, 4), value, dtype=mx.float32),
            values=mx.full((1, 2, 16, 4), value, dtype=mx.float32),
            offset=offset,
        )

    def _seed_retracted_request(self, runner):
        """R: prefill 2 tokens (slots 1,2) + 3 unsynced decode steps.

        The scheduler wrote R's row 0 as [1,2,3,4,5]; R's synced prefix ends
        at offset 2, so cache positions [2:5] hold committed-but-unsynced
        decode KV (the RETRACT_MARKER payload).
        """
        runner._req_to_token_pool.req_to_token[0, 0:5] = torch.arange(
            1, 6, dtype=torch.int64
        )
        runner._req_caches["R"] = [self._cache_layer(offset=5, value=self.RETRACT_MARKER)]
        runner._req_pool_idx["R"] = 0
        runner._req_synced_offset["R"] = 2
        runner._req_token_ids["R"] = [1, 1, 2]
        runner._req_sampling["R"] = None

    def _reuse_row_for_new_prefill(self, runner):
        """P: radix-hits a 2-token prefix (slots 1,2) + 3 fresh tokens.

        P reuses row 0. The freed slots come back from the allocator's free
        list, so row 0 ends up [1,2,3,4,5] again — the collision that makes
        the stale read plausible instead of exotic. Slots 1,2 carry valid
        pool KV (P's borrowed prefix); slots 3,4,5 are P's fresh slots.
        """
        runner._req_to_token_pool.req_to_token[0, 0:5] = torch.arange(
            1, 6, dtype=torch.int64
        )
        import mlx.core as mx

        fill = mx.full((2, 4), self.OWNER_MARKER, dtype=mx.float32)
        runner._attention_kv_pool.set_kv(0, mx.array([1, 2], dtype=mx.int32), fill, fill)

    def _assert_pool_free_of_retract_kv(self, runner):
        """No buffer position may contain the retracted request's KV."""
        import mlx.core as mx

        pool = runner._attention_kv_pool
        mx.eval(*pool.all_buffers())
        for buf in pool.all_buffers():
            flat = buf.reshape(-1)
            self.assertNotIn(
                self.RETRACT_MARKER,
                flat.tolist(),
                "Retracted request's decode KV reached the shared pool: "
                "flush_all_decode_kv synced through a reused req_to_token "
                "row (issue #33547).",
            )

    @staticmethod
    def _worker(runner):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = runner
        worker._mlx_active_rids = set()
        worker._mlx_pool_initialized = True
        return worker

    @staticmethod
    def _extend_batch_with_prefill():
        """A minimal EXTEND batch whose single request routes to prefill."""

        class _Req:
            rid = "P"
            prefix_indices = torch.tensor([1, 2], dtype=torch.long)
            fill_ids = [9, 9, 7, 7, 7]

            def get_fill_ids(self):
                return self.fill_ids

            kv = SimpleNamespace(req_pool_idx=0)
            extend_range = None
            full_untruncated_fill_ids = fill_ids

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[_Req()],
            extend_lens=[3],
            decoding_reqs=None,
            sampling_info=None,
            return_logprob=False,
            input_ids=torch.arange(3, dtype=torch.long),
            out_cache_loc=torch.tensor([3, 4, 5], dtype=torch.long),
        )
        return batch

    # ---------- THE REGRESSION ----------

    def test_retracted_rid_does_not_flush_into_reused_row_on_extend(self):
        """End-to-end: retract -> row reused -> extend forward -> pool clean.

        Before the fix there is no retract-time disposal, so R survives in
        _req_caches past the row reuse, and the extend forward's
        flush_all_decode_kv syncs R through the reused row: R's decode KV
        (RETRACT_MARKER) is scattered over slots 3,4,5 — which now belong
        to the new prefill P.
        """
        runner = self._make_runner()
        worker = self._worker(runner)
        self._seed_retracted_request(runner)
        worker._mlx_active_rids = {"R"}

        # Scheduler retracts R (row + slots freed) and a new prefill reuses
        # the row before the next forward runs.
        self._reuse_row_for_new_prefill(runner)

        # The retract-time disposal hook (present only after the fix).
        hook = getattr(worker, "prepare_for_retraction", None)
        if callable(hook):
            hook(SimpleNamespace(rid="R"))

        # Next forward is an extend (P's prefill), the previously
        # unprotected mode for stale rids.
        worker.async_forward_batch_generation_mlx(self._extend_batch_with_prefill())

        self._assert_pool_free_of_retract_kv(runner)

    # ---------- hook semantics ----------

    def test_prepare_for_retraction_discards_state_without_sync(self):
        """The hook is a pure discard: full state pop, zero pool writes."""
        runner = self._make_runner()
        worker = self._worker(runner)
        self._seed_retracted_request(runner)
        worker._mlx_active_rids = {"R"}

        worker.prepare_for_retraction(SimpleNamespace(rid="R"))

        self.assertFalse(runner.has_request("R"))
        self.assertNotIn("R", runner._req_pool_idx)
        self.assertNotIn("R", runner._req_synced_offset)
        self.assertNotIn("R", runner._req_token_ids)
        self.assertNotIn("R", runner._req_sampling)
        # The cache list was recycled, not leaked.
        self.assertEqual(len(runner._cache_pool), 1)
        # Nothing was synced: the dirty decode tail must not reach the pool.
        self._assert_pool_free_of_retract_kv(runner)
        # Idempotent: retract bookkeeping may race with stale-rid cleanup.
        worker.prepare_for_retraction(SimpleNamespace(rid="R"))

    def test_discard_request_is_pure_discard(self):
        """Runner-level contract for the disposal primitive (no sync)."""
        runner = self._make_runner()
        self._seed_retracted_request(runner)

        runner.discard_request("R")

        self.assertFalse(runner.has_request("R"))
        self.assertNotIn("R", runner._req_pool_idx)
        self._assert_pool_free_of_retract_kv(runner)
        # Idempotent.
        runner.discard_request("R")

    def test_retracted_request_requeues_as_fresh_prefill(self):
        """After disposal, the requeued request must route to prefill.

        Without the hook the stale has_request() would route it as a
        continuation and extend the orphaned private cache against a
        freed (and possibly reused) req_to_token row.
        """
        worker = self._worker(self._make_runner())
        worker._mlx_runner._req_caches["R"] = [self._cache_layer(5, 0.0)]
        self.assertEqual(worker._route_extend_request("R", set()), "continuation")

        worker.prepare_for_retraction(SimpleNamespace(rid="R"))

        self.assertEqual(worker._route_extend_request("R", set()), "prefill")

    def test_prepare_for_retraction_drops_active_rid_membership(self):
        """Disposal also clears _mlx_active_rids so the later decode-mode
        stale-rid cleanup never re-processes the rid through the syncing
        remove_request path."""
        runner = self._make_runner()
        worker = self._worker(runner)
        self._seed_retracted_request(runner)
        worker._mlx_active_rids = {"R", "Q"}

        worker.prepare_for_retraction(SimpleNamespace(rid="R"))

        self.assertEqual(worker._mlx_active_rids, {"Q"})


if __name__ == "__main__":
    unittest.main()
