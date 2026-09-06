"""Regression tests for the SWA chunked-req stash gate (#24252)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import complete_mm_embedding_validations
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _make_req(
    *,
    req_pool_idx: int,
    fill_ids: list,
    prefix_indices: torch.Tensor,
    extend_input_len: int,
    fill_len: int,
) -> Req:
    req = Req.__new__(Req)
    req.rid = "test-req"
    req.origin_input_ids = array("q", fill_ids)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill_ids)
    req.prefix_indices = prefix_indices
    req.extend_range = Range(fill_len - extend_input_len, fill_len)
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
    req.skip_radix_cache_insert = False
    req.mm_embedding_validation_count = 0
    req.last_node = None
    req.swa_uuid_for_lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.cache_salt = None
    req.kv.mamba_pool_idx = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_req_to_token_pool(num_slots: int, max_context: int) -> SimpleNamespace:
    # Slot s contains a recognizable fingerprint [s*1000, s*1000+1, ...]
    # so we can tell a corrupted prefix_indices from a healthy one by content.
    pool = SimpleNamespace()
    pool.req_to_token = (
        torch.arange(max_context, dtype=torch.int32).unsqueeze(0).repeat(num_slots, 1)
        + torch.arange(num_slots, dtype=torch.int32).unsqueeze(1) * 1000
    )
    return pool


def _make_chunk_cache(req_to_token_pool) -> ChunkCache:
    return ChunkCache(
        SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
    )


def _scheduler_for_get_next_batch(*, tree_cache, chunked_req) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.scheduler_stage_metrics = None
    s._abort_on_waiting_timeout = MagicMock()
    s._abort_on_running_timeout = MagicMock()
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
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
    s.tree_cache = tree_cache
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    return s


def _scheduler_for_raw_prefill(*, chunked_req, waiting_queue) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_hierarchical_cache = False
    s.enable_unified_cache_external_linker = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.chunked_req = chunked_req
    s.waiting_queue = waiting_queue
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = MagicMock(return_value=1)
    s.policy = MagicMock()
    s.dynamic_chunk_sizer = None
    s.chunked_prefill_size = 8
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(), prefill_aware_swa=False
        )
    )
    s.page_size = 1
    s.tree_cache = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock()
    s.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    s.max_prefill_tokens = 32
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 4
    s.max_running_requests = 4
    s.dllm_config = None
    s.enable_lora = False
    s.req_to_token_pool = SimpleNamespace(mamba_allocator=None)
    s.enable_hicache_storage = False
    s.enable_priority_scheduling = False
    s.model_config = MagicMock()
    s.enable_overlap = False
    s.spec_algorithm = MagicMock()
    s.load_inquirer = MagicMock()
    s.load_inquirer._get_num_pending_tokens.return_value = 0
    return s


class TestStashGatePreservesPrefixIndices(CustomTestCase):
    """Consumer side: real ChunkCache.cache_unfinished_req mutates
    req.prefix_indices iff stash actually runs, so prefix_indices content
    is the bug-detection signal. The stash gate is content-based:
    `fill_len > len(prefix_indices)` means there is freshly computed KV to
    cache; otherwise the chunk was parked and stashing must be skipped."""

    POOL_IDX = 4
    INITIAL_PREFIX_LEN = 8  # what was really cached last iter
    POST_RESET_FILL_LEN = 32  # length after init_next_round_input rebuilds
    NUM_SLOTS = 8
    MAX_CONTEXT = 64

    def _build(self, *, fill_len: int):
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        initial_prefix = pool.req_to_token[self.POOL_IDX, : self.INITIAL_PREFIX_LEN].to(
            dtype=torch.int64, copy=True
        )
        req = _make_req(
            req_pool_idx=self.POOL_IDX,
            fill_ids=list(range(self.POST_RESET_FILL_LEN)),
            prefix_indices=initial_prefix,
            extend_input_len=fill_len - self.INITIAL_PREFIX_LEN,
            fill_len=fill_len,
        )
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=req)
        return s, req, initial_prefix, pool

    def test_parked_chunked_req_keeps_real_prefix_indices(self):
        # A parked chunk has fill_len == len(prefix_indices): no new KV was
        # computed, so the gate must skip stash and leave prefix_indices intact.
        s, req, initial_prefix, _ = self._build(fill_len=self.INITIAL_PREFIX_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        self.assertEqual(req.prefix_indices.shape[0], self.INITIAL_PREFIX_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, initial_prefix))

    def test_scheduled_chunked_req_advances_prefix_indices_via_real_stash(self):
        # Symmetric guard against over-gating: when fill_len has advanced past
        # the cached prefix, stash must run and advance prefix_indices.
        s, req, _, pool = self._build(fill_len=self.POST_RESET_FILL_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        expected = pool.req_to_token[self.POOL_IDX, : self.POST_RESET_FILL_LEN].to(
            dtype=torch.int64
        )
        self.assertEqual(req.prefix_indices.shape[0], self.POST_RESET_FILL_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, expected))

    def test_no_chunked_req_never_mutates_state(self):
        # The outer `if chunked_req is not None` guard must hold on the retract
        # path that clears chunked_req.
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=None)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )
        self.assertIsNone(s.chunked_req)


class TestMMEmbeddingValidationGate(CustomTestCase):
    def _make_chunked_req(self):
        req = Mock()
        req.rid = "chunk-owner"
        req.mm_embedding_validation_count = 1
        req.inflight_middle_chunks = 0
        req.prefix_indices = torch.arange(8)
        req.extend_range = Range(0, 8)

        def advance():
            req.extend_range = Range(8, 16)

        req.init_next_round_input.side_effect = advance
        return req

    def test_pending_validation_parks_owner_before_init_and_admission(self):
        req = self._make_chunked_req()
        queued = [object(), object()]
        scheduler = _scheduler_for_raw_prefill(chunked_req=req, waiting_queue=queued)
        scheduler.enable_priority_preemption = True
        running_batch = SimpleNamespace(reqs=[], batch_is_full=True)
        original_prefix = req.prefix_indices
        original_range = req.extend_range

        with (
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch("sglang.srt.managers.scheduler.PrefillAdder") as prefill_adder,
        ):
            for _ in range(3):
                batch, returned_running = Scheduler._get_new_batch_prefill_raw(
                    scheduler, None, running_batch
                )
                self.assertIsNone(batch)
                self.assertIs(returned_running, running_batch)

        self.assertIs(scheduler.chunked_req, req)
        self.assertIs(req.prefix_indices, original_prefix)
        self.assertEqual(req.extend_range, original_range)
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertEqual(req.mm_embedding_validation_count, 1)
        self.assertEqual(scheduler.waiting_queue, queued)
        self.assertFalse(running_batch.batch_is_full)
        req.init_next_round_input.assert_not_called()
        scheduler.get_num_allocatable_reqs.assert_not_called()
        scheduler.policy.calc_priority.assert_not_called()
        prefill_adder.assert_not_called()

    def test_completed_validation_advances_next_chunk_once(self):
        req = self._make_chunked_req()
        scheduler = _scheduler_for_raw_prefill(chunked_req=req, waiting_queue=[])
        running_batch = SimpleNamespace(
            reqs=[], batch_is_full=False, is_empty=lambda: True, return_logprob=False
        )
        adder = MagicMock()
        adder.can_run_list = []
        adder.preempt_list = []
        adder.new_chunked_req = None

        def add_chunked_req(owner):
            adder.can_run_list.append(owner)
            return owner

        adder.add_chunked_req.side_effect = add_chunked_req
        new_batch = MagicMock(return_logprob=False, input_embeds=None)

        complete_mm_embedding_validations([req], torch.tensor([[1, 0, 0, 0]]))
        with (
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch("sglang.srt.managers.scheduler.PrefillAdder", return_value=adder),
            patch(
                "sglang.srt.managers.scheduler.get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=4),
            ),
            patch(
                "sglang.srt.managers.scheduler.ScheduleBatch.init_new",
                return_value=new_batch,
            ),
            patch("sglang.srt.managers.scheduler.PrefillStats.from_adder"),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            batch, _ = Scheduler._get_new_batch_prefill_raw(
                scheduler, None, running_batch
            )

            req.mm_embedding_validation_count = 1
            parked_batch, _ = Scheduler._get_new_batch_prefill_raw(
                scheduler, None, running_batch
            )

        self.assertIs(batch, new_batch)
        self.assertIsNone(parked_batch)
        self.assertIs(scheduler.chunked_req, req)
        self.assertEqual(req.extend_range, Range(8, 16))
        self.assertEqual(req.inflight_middle_chunks, 1)
        req.init_next_round_input.assert_called_once_with()
        adder.add_chunked_req.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
