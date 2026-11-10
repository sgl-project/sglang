"""Regression tests for scheduler prefill/decode transition gates."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import (
    NextBatchPlan,
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.scheduler_test_utils import empty_request_lists

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
    s.server_args = MagicMock()
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.batch_size.return_value = 0
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
    s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    s.running_reqs = KeyedReqList(ListKind.DECODING)
    s.invariant_checker = MagicMock()
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


class TestPendingPrefillAbortGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(model_path="dummy")
        override.install()
        self.addCleanup(override.restore)
        self.scheduler = Scheduler.__new__(Scheduler)
        empty_request_lists(self.scheduler)
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.scheduler.enable_continuous_input_polling = False
        self.scheduler.enable_skip_finishing_decode = True
        self.scheduler.mm_receiver = None
        self.reqs = [
            Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3]),
                sampling_params=SamplingParams(max_new_tokens=8),
            )
            for rid in ("aborted-prefill", "live-prefill", "aborted-decode")
        ]
        self.reqs[2].output_ids.append(7)
        for req in self.reqs:
            req.sampling_params.normalize(None)
        self.scheduler.running_reqs.join([self.reqs[2]])
        self.scheduler.running_batch = ScheduleBatch(reqs=[self.reqs[2]])
        self.batch = ScheduleBatch(
            reqs=self.reqs,
            forward_mode=ForwardMode.EXTEND,
            decoding_reqs=[self.reqs[2]],
        )
        self.batch.add_inflight_step()
        self.scheduler._join_prefill_launched(self.batch)
        self.scheduler.last_batch = self.batch

    def test_abort_discards_only_the_inflight_prefill_before_decode(self):
        """An abort between final-prefill launch and commit must not schedule
        another step; a decode row in the same batch keeps its partial output."""
        for message in (None, "Running request timeout"):
            with self.subTest(message=message):
                self.scheduler.abort_request(
                    AbortReq(rid="aborted", abort_message=message)
                )
                prefill, live, decode = self.reqs
                self.assertIsInstance(prefill.discard_output_reason, FINISH_ABORT)
                self.assertIs(prefill.discard_output_reason, prefill.to_finish)
                self.assertEqual(prefill.to_finish.message, message or "Aborted")
                self.assertEqual(
                    prefill.to_finish.status_code, 503 if message else None
                )
                self.assertIsNone(live.to_finish)
                self.assertIsNone(decode.discard_output_reason)
                self.assertIsInstance(decode.to_finish, FINISH_ABORT)
                self.assertEqual(list(decode.output_ids), [7])
                self.assertFalse(prefill.finished())
                self.assertEqual(
                    self.scheduler._reqs_finishing_inflight(self.batch), [prefill]
                )
                next_batch = self.scheduler.update_running_batch(
                    ScheduleBatch(reqs=[prefill])
                )
                self.assertTrue(next_batch.is_empty())
                self.assertEqual(prefill.decode_batch_idx, 0)

    def test_processed_prefill_and_pd_keep_existing_abort_semantics(self):
        """A committed prefill or a PD transfer must not acquire discard semantics."""
        for mode, inflight_steps in (
            (DisaggregationMode.NULL, 0),
            (DisaggregationMode.PREFILL, 1),
            (DisaggregationMode.DECODE, 1),
        ):
            with self.subTest(mode=mode):
                self.scheduler.disaggregation_mode = mode
                prefill = self.reqs[0]
                prefill.inflight_steps = inflight_steps
                self.scheduler.abort_request(AbortReq(rid=prefill.rid))
                self.assertIsInstance(prefill.to_finish, FINISH_ABORT)
                self.assertIsNone(prefill.discard_output_reason)


if __name__ == "__main__":
    unittest.main()
