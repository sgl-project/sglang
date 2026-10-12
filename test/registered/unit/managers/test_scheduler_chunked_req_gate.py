"""Regression tests for scheduler prefill/decode transition gates."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_req(
    *,
    req_pool_idx: int,
    fill_ids: list,
    prefix_len: int,
    fill_len: int,
) -> Req:
    req = Req.__new__(Req)
    req.rid = "test-req"
    req.origin_input_ids = array("q", fill_ids)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill_ids)
    req.prefix_len = prefix_len
    req.extend_end = fill_len
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
    req.skip_radix_cache_insert = False
    req.finished_reason = None
    req.last_node = None
    req.lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.cache_salt = None
    req.kv.mamba_pool_idx = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_tree_cache() -> BasePrefixCache:
    # The gate under test lives in the scheduler; the cache is a stub.
    return MagicMock(spec=BasePrefixCache)


def _scheduler_for_get_next_batch(*, tree_cache, chunked_req) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.scheduler_stage_metrics = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    # Exercise the unconditional scheduler-loop HiCache event-drain point.
    s.enable_hierarchical_cache = True
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.last_batch = None
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.server_args = MagicMock(speculative_skip_dp_mlp_sync=True)
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.is_prefill_only = False
    s.running_batch.batch_is_full = False
    s.running_batch.reqs = []
    s.prefill_decode_interval = 0
    s._prefill_decode_interval_remaining = 0
    s.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=None, running_batch=s.running_batch)
    )
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.ngram_embedding_manager = MagicMock()
    s.ngram_embedding_manager.prepare_for_forward = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    tree_cache.check_hicache_events = MagicMock()
    s.tree_cache = tree_cache
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    s.result_queue = []
    return s


class TestStashGatePreservesPrefix(CustomTestCase):
    """The stash gate advances prefix_len iff `fill_len > prefix_len`, i.e. the
    chunk computed new KV; a parked chunk must be left untouched."""

    POOL_IDX = 4
    INITIAL_PREFIX_LEN = 8  # what was really cached last iter
    POST_RESET_FILL_LEN = 32  # length after init_next_round_input rebuilds

    def _build(self, *, fill_len: int):
        cache = _make_tree_cache()
        req = _make_req(
            req_pool_idx=self.POOL_IDX,
            fill_ids=list(range(self.POST_RESET_FILL_LEN)),
            prefix_len=self.INITIAL_PREFIX_LEN,
            fill_len=fill_len,
        )
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=req)
        return s, req

    def test_parked_chunked_req_keeps_its_prefix(self):
        # A parked chunk has fill_len == prefix_len: no new KV was computed,
        # so the gate must skip stash and leave the prefix intact.
        s, req = self._build(fill_len=self.INITIAL_PREFIX_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        self.assertEqual(req.prefix_len, self.INITIAL_PREFIX_LEN)
        s.tree_cache.checkpoint.assert_not_called()

    def test_scheduled_chunked_req_advances_prefix_via_real_stash(self):
        # Symmetric guard against over-gating: when fill_len has advanced past
        # the cached prefix, stash must run and advance prefix_len.
        s, req = self._build(fill_len=self.POST_RESET_FILL_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        self.assertEqual(req.prefix_len, self.POST_RESET_FILL_LEN)
        s.tree_cache.checkpoint.assert_called_once_with(
            req, up_to=self.POST_RESET_FILL_LEN
        )

    def test_no_chunked_req_never_mutates_state(self):
        # The outer `if chunked_req is not None` guard must hold on the retract
        # path that clears chunked_req.
        cache = _make_tree_cache()
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=None)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )
        self.assertIsNone(s.chunked_req)
        cache.checkpoint.assert_not_called()


class TestPendingPrefillAbortGate(CustomTestCase):
    @staticmethod
    def _make_last_batch():
        aborted_prefill = MagicMock(rid="aborted-prefill", to_finish=object())
        live_prefill = MagicMock(rid="live-prefill", to_finish=None)
        aborted_decode = MagicMock(rid="aborted-decode", to_finish=object())
        for req in (aborted_prefill, live_prefill, aborted_decode):
            req.finished.return_value = False

        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = True
        last_batch.reqs = [aborted_prefill, live_prefill, aborted_decode]
        last_batch.decoding_reqs = [aborted_decode]
        last_batch.chunked_req = None
        last_batch.is_prefill_only = False
        last_batch.batch_size.side_effect = lambda: len(last_batch.reqs)
        last_batch.is_empty.side_effect = lambda: not last_batch.reqs

        def filter_batch(*, chunked_req_to_exclude):
            excluded = set(chunked_req_to_exclude)
            last_batch.reqs = [req for req in last_batch.reqs if req not in excluded]

        last_batch.filter_batch.side_effect = filter_batch
        return last_batch, aborted_prefill, live_prefill, aborted_decode

    def _scheduled_reqs(self, *, has_pending_result):
        scheduler = _scheduler_for_get_next_batch(
            tree_cache=MagicMock(), chunked_req=None
        )
        scheduler.result_queue = [object()] if has_pending_result else []
        last_batch, aborted_prefill, live_prefill, aborted_decode = (
            self._make_last_batch()
        )
        scheduler.get_new_batch_prefill.side_effect = lambda running_batch: (
            NextBatchPlan(batch_to_run=None, running_batch=running_batch)
        )
        scheduled_reqs = []

        def update_running_batch(batch):
            scheduled_reqs.extend(batch.reqs)
            return batch

        scheduler.update_running_batch.side_effect = update_running_batch

        Scheduler.get_next_batch_to_run(
            scheduler,
            running_batch=scheduler.running_batch,
            last_batch=last_batch,
        )

        return scheduled_reqs, aborted_prefill, live_prefill, aborted_decode

    def test_abort_is_excluded_before_optimistic_decode(self):
        scheduled_reqs, aborted_prefill, live_prefill, aborted_decode = (
            self._scheduled_reqs(has_pending_result=True)
        )

        self.assertNotIn(aborted_prefill, scheduled_reqs)
        self.assertIn(live_prefill, scheduled_reqs)
        self.assertIn(aborted_decode, scheduled_reqs)

    def test_abort_is_not_excluded_after_prefill_result_was_processed(self):
        scheduled_reqs, aborted_prefill, live_prefill, aborted_decode = (
            self._scheduled_reqs(has_pending_result=False)
        )

        self.assertIn(aborted_prefill, scheduled_reqs)
        self.assertIn(live_prefill, scheduled_reqs)
        self.assertIn(aborted_decode, scheduled_reqs)


if __name__ == "__main__":
    unittest.main()
