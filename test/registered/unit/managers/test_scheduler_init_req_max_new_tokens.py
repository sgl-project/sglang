import logging
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSchedulerInitReqMaxNewTokens(CustomTestCase):
    """Property tests for Scheduler.init_req_max_new_tokens.

    Rules enforced when clipping a request's max_new_tokens:
      1. context: input_len + max_new_tokens < max_req_len
      2. admission budget (PrefillAdder):
         ceil_page(input_len) + max_new_tokens + page_size * shard_widening
         < max_total_num_tokens * shard_widening
      3. env limit: <= SGLANG_MAX_NEW_TOKENS_LIMIT when set and positive
      4. never above the requested value
      5. min_new_tokens <= max_new_tokens afterwards

    Each case asserts all rules hold and the result is tight: one more token
    would violate a rule or exceed the request. Request intake rejects raw
    inputs that can never pass PrefillAdder's total-token gate.
    """

    @classmethod
    def setUpClass(cls):
        # Silence the per-request capping warning; the sweep triggers it a lot.
        cls._scheduler_logger = logging.getLogger("sglang.srt.managers.scheduler")
        cls._old_level = cls._scheduler_logger.level
        cls._scheduler_logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        cls._scheduler_logger.setLevel(cls._old_level)

    def setUp(self):
        override = get_context().override_server_args(allow_auto_truncate=False)
        override.install()
        self.addCleanup(override.restore)
        # Tests use an unsharded topology unless they explicitly override it.
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def _new_scheduler(
        self,
        max_req_len: int = 128,
        max_total_num_tokens: int = 1024,
        page_size: int = 1,
        kv_shard_widening: int = 1,
    ) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = max_req_len
        scheduler.max_total_num_tokens = max_total_num_tokens
        scheduler.page_size = page_size
        scheduler.kv_shard_widening = kv_shard_widening
        scheduler.max_new_tokens_limit = envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()
        scheduler.sliding_window_size = None
        scheduler.chunked_prefill_size = None
        scheduler.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=max_total_num_tokens,
            dtype=torch.int64,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        scheduler.token_to_kv_pool_allocator.page_size = page_size
        return scheduler

    def _new_req(self, max_new_tokens, input_len: int = 8, min_new_tokens: int = 0):
        return SimpleNamespace(
            rid="test-req",
            origin_input_ids=[0] * input_len,
            sampling_params=SimpleNamespace(
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens
            ),
        )

    def _init_and_check(self, scheduler, req) -> int:
        """Run init_req_max_new_tokens, then assert all admission rules hold
        and the result is tight. Returns the resulting max_new_tokens."""
        requested = req.sampling_params.max_new_tokens
        scheduler.init_req_max_new_tokens(req)
        max_new_tokens = req.sampling_params.max_new_tokens

        input_len = len(req.origin_input_ids)
        page_size = scheduler.page_size
        shard_widening = scheduler.kv_shard_widening
        paged_input_len = -(-input_len // page_size) * page_size
        limit = scheduler.max_new_tokens_limit
        limit_active = limit is not None and limit > 0

        def satisfies_rules(candidate: int) -> bool:
            context_ok = input_len + candidate < scheduler.max_req_len
            budget_ok = (
                paged_input_len + candidate + page_size * shard_widening
                < scheduler.max_total_num_tokens * shard_widening
            )
            limit_ok = not limit_active or candidate <= limit
            requested_ok = requested is None or candidate <= requested
            return context_ok and budget_ok and limit_ok and requested_ok

        self.assertGreaterEqual(max_new_tokens, 0)
        if max_new_tokens > 0:
            self.assertTrue(satisfies_rules(max_new_tokens))
        self.assertFalse(satisfies_rules(max_new_tokens + 1))
        self.assertLessEqual(req.sampling_params.min_new_tokens, max_new_tokens)
        return max_new_tokens

    def test_limit_disabled_by_default(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(None):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64)
            self.assertEqual(self._init_and_check(scheduler, req), 64)

    def test_limit_clips_explicit_request(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64)
            self.assertEqual(self._init_and_check(scheduler, req), 16)

    def test_limit_applies_when_request_unset(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=None)
            self.assertEqual(self._init_and_check(scheduler, req), 16)

    def test_non_positive_limit_is_ignored(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(limit):
                    scheduler = self._new_scheduler()
                    req = self._new_req(max_new_tokens=64)
                    self.assertEqual(self._init_and_check(scheduler, req), 64)

    def test_context_rule_binds_tighter_than_limit(self):
        max_req_len, input_len = 32, 20
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler(max_req_len=max_req_len)
            req = self._new_req(max_new_tokens=64, input_len=input_len)
            self.assertEqual(
                self._init_and_check(scheduler, req), max_req_len - input_len - 1
            )

    def test_budget_rule_binds_tighter_than_limit(self):
        max_total_num_tokens, page_size, input_len = 24, 4, 8
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(32):
            scheduler = self._new_scheduler(
                max_total_num_tokens=max_total_num_tokens, page_size=page_size
            )
            req = self._new_req(max_new_tokens=64, input_len=input_len)
            paged_input_len = -(-input_len // page_size) * page_size
            self.assertEqual(
                self._init_and_check(scheduler, req),
                max_total_num_tokens - paged_input_len - page_size - 1,
            )

    def test_dcp_budget_is_already_in_logical_tokens(self):
        with get_parallel().override(attn_dcp_size=4):
            scheduler = self._new_scheduler(max_total_num_tokens=24, page_size=4)
            req = self._new_req(max_new_tokens=64, input_len=8)

            self.assertEqual(self._init_and_check(scheduler, req), 11)

    def test_sharded_budget_reserves_one_page_per_shard(self):
        max_total_num_tokens, page_size, input_len, shard_widening = 8, 4, 8, 3
        scheduler = self._new_scheduler(
            max_req_len=128,
            max_total_num_tokens=max_total_num_tokens,
            page_size=page_size,
            kv_shard_widening=shard_widening,
        )
        req = self._new_req(max_new_tokens=64, input_len=input_len)

        max_new_tokens = self._init_and_check(scheduler, req)

        widened_capacity = max_total_num_tokens * shard_widening
        per_req_overhead = page_size * shard_widening
        self.assertEqual(
            max_new_tokens,
            widened_capacity - input_len - per_req_overhead - 1,
        )
        self.assertLess(
            input_len + max_new_tokens + per_req_overhead,
            widened_capacity,
        )

    def test_max_req_input_len_matches_prefill_admission_boundary(self):
        # With a 190,784-token pool and 64-token pages, the old advertised
        # input limit was 190,778. Inputs in [190,720, 190,777] pass that
        # check but can never pass PrefillAdder's first total-token gate. The
        # threshold itself is rejected by validate_input_length, so 190,719 is
        # the largest admitted input.
        capacity, page_size = 190_784, 64
        threshold = Scheduler.get_max_admissible_input_len(
            capacity - 6, capacity, page_size
        )
        self.assertEqual(threshold, 190_720)

        scheduler = self._new_scheduler(
            max_req_len=capacity - 1,
            max_total_num_tokens=capacity,
            page_size=page_size,
        )
        req = self._new_req(max_new_tokens=1 << 20, input_len=threshold - 1)
        self.assertEqual(self._init_and_check(scheduler, req), 0)

        oversized_req = self._new_req(max_new_tokens=1 << 20, input_len=190_720)
        self.assertIsNotNone(
            validate_input_length(oversized_req, threshold, allow_auto_truncate=False)
        )

        # Auto truncation must produce an admissible input and must run before
        # init_req_max_new_tokens, so both use the same final prompt length.
        self.assertIsNone(
            validate_input_length(oversized_req, threshold, allow_auto_truncate=True)
        )
        self.assertEqual(len(oversized_req.origin_input_ids), threshold - 1)
        self.assertEqual(self._init_and_check(scheduler, oversized_req), 0)

    def test_max_req_input_len_does_not_double_count_dcp_capacity(self):
        with get_parallel().override(attn_dcp_size=4):
            self.assertEqual(
                Scheduler.get_max_admissible_input_len(
                    max_req_input_len=2_000,
                    max_total_num_tokens=512,
                    page_size=64,
                ),
                448,
            )

    def test_max_req_input_len_reserves_one_page_per_kv_shard(self):
        self.assertEqual(
            Scheduler.get_max_admissible_input_len(
                max_req_input_len=2_000,
                max_total_num_tokens=512,
                page_size=64,
                kv_shard_widening=2,
            ),
            896,
        )

    def test_auto_truncate_rejects_when_no_nonempty_prompt_can_fit(self):
        for threshold in (0, 1):
            with self.subTest(threshold=threshold):
                req = self._new_req(max_new_tokens=1, input_len=8)

                error_msg = validate_input_length(
                    req, threshold, allow_auto_truncate=True
                )

                self.assertIsNotNone(error_msg)
                self.assertIn("no room for a non-empty prompt", error_msg)
                self.assertEqual(len(req.origin_input_ids), 8)

    def test_auto_truncate_keeps_token_aligned_fields_consistent(self):
        req = self._new_req(max_new_tokens=1, input_len=10)
        req.origin_input_ids = list(range(10))
        req.origin_input_ids_unpadded = list(range(10))
        req.input_embeds = [[value] for value in range(10)]
        req.token_type_ids = [value % 2 for value in range(10)]
        req.multi_item_delimiter_indices = [0, 3, 7, 9]

        self.assertIsNone(validate_input_length(req, 8, allow_auto_truncate=True))

        self.assertEqual(req.origin_input_ids, list(range(7)))
        self.assertEqual(req.origin_input_ids_unpadded, list(range(7)))
        self.assertEqual(req.input_embeds, [[value] for value in range(7)])
        self.assertEqual(req.token_type_ids, [value % 2 for value in range(7)])
        self.assertEqual(req.multi_item_delimiter_indices, [0, 3])

    def test_min_new_tokens_clamped_to_limit(self):
        with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(16):
            scheduler = self._new_scheduler()
            req = self._new_req(max_new_tokens=64, min_new_tokens=32)
            self.assertEqual(self._init_and_check(scheduler, req), 16)
            self.assertEqual(req.sampling_params.min_new_tokens, 16)

    def test_admission_rules_sweep(self):
        for page_size in (1, 4, 16):
            for input_len in (1, 8, 100):
                for requested in (None, 0, 5, 64, 1 << 20):
                    for limit in (None, 0, 16, 1 << 20):
                        for max_req_len, max_total_num_tokens in (
                            (128, 1024),
                            (32, 24),
                            (128, 24),
                        ):
                            with self.subTest(
                                page_size=page_size,
                                input_len=input_len,
                                requested=requested,
                                limit=limit,
                                max_req_len=max_req_len,
                                max_total_num_tokens=max_total_num_tokens,
                            ):
                                with envs.SGLANG_MAX_NEW_TOKENS_LIMIT.override(limit):
                                    scheduler = self._new_scheduler(
                                        max_req_len=max_req_len,
                                        max_total_num_tokens=max_total_num_tokens,
                                        page_size=page_size,
                                    )
                                    req = self._new_req(
                                        max_new_tokens=requested, input_len=input_len
                                    )
                                    self._init_and_check(scheduler, req)

    def test_unified_budget_rounds_prompt_and_decode_together(self):
        bundle = init_unified_swa_pools(
            device="cpu",
            kv_cache_dtype=torch.float16,
            head_num=1,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=1,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=4,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            total_bytes=768,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=True,
        )
        scheduler = self._new_scheduler(page_size=4)
        scheduler.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        scheduler.sliding_window_size = 4
        scheduler.chunked_prefill_size = 4
        scheduler.max_new_tokens_limit = None
        for prompt_len in (4, 5, 6, 7):
            with self.subTest(prompt_len=prompt_len):
                req = self._new_req(max_new_tokens=1, input_len=prompt_len)
                scheduler.init_req_max_new_tokens(req)
                self.assertEqual(req.sampling_params.max_new_tokens, 1)

    def test_empty_running_batch_clears_stale_batch_is_full(self):
        scheduler = object.__new__(Scheduler)
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_priority_preemption = False
        scheduler.is_hybrid_swa = False
        scheduler.waiting_queue = []
        scheduler.chunked_req = None

        running_batch = SimpleNamespace(
            batch_is_full=True,
            is_empty=lambda: True,
        )
        batch_to_run, returned_batch = Scheduler._get_new_batch_prefill_raw(
            scheduler, None, running_batch
        )

        self.assertIsNone(batch_to_run)
        self.assertIs(returned_batch, running_batch)
        self.assertFalse(running_batch.batch_is_full)

    @patch.object(ScheduleBatch, "prepare_for_extend")
    def test_cancelled_near_capacity_request_does_not_block_followup(self, _prepare):
        """A rejected prefill must not strand later requests on an idle KV pool."""
        scheduler = self._new_scheduler(max_total_num_tokens=512, page_size=64)
        scheduler.tree_cache = RadixCache.create_simulated(page_size=64)
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_priority_preemption = False
        scheduler.enable_priority_scheduling = False
        scheduler.is_hybrid_swa = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_lmcache = False
        scheduler.enable_lora = False
        scheduler.chunked_req = None
        scheduler.min_free_slots_delayer = None
        scheduler.dynamic_chunk_sizer = None
        scheduler.is_mixed_chunk = False
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.max_prefill_tokens = 512
        scheduler.max_prefill_bs = 8
        scheduler.max_running_requests = 8
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.truncation_align_size = None
        scheduler.dllm_config = None
        scheduler.processed_tokens_counter = 0
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.req_to_token_pool = SimpleNamespace(
            available_size=lambda: 8, device="cpu"
        )
        scheduler.beam_coordinator = SimpleNamespace(pending_member_rows=lambda _: 0)
        scheduler.model_config = SimpleNamespace(vocab_size=2)
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(), prefill_aware_swa=False
            )
        )
        scheduler.load_inquirer = SimpleNamespace(_get_num_pending_tokens=lambda **_: 0)
        scheduler.policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=scheduler.tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        oversized = Req(
            "oversized", None, array("q", [1] * 448), SamplingParams(max_new_tokens=0)
        )
        scheduler.waiting_queue = [oversized]
        running_batch = ScheduleBatch(reqs=[], batch_is_full=False)

        with get_parallel().override(pp_max_micro_batch_size=8, tp_rank=0):
            batch, running_batch = scheduler._get_new_batch_prefill_raw(
                None, running_batch
            )
            self.assertIsNone(batch)
            self.assertTrue(running_batch.batch_is_full)

            oversized.set_finish_with_abort("Cancelled")
            scheduler.waiting_queue.remove(oversized)
            followup = Req(
                "followup", None, array("q", [1] * 8), SamplingParams(max_new_tokens=4)
            )
            scheduler.waiting_queue.append(followup)
            self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 512)

            # Only hardware tensor preparation is stubbed. Request matching,
            # PrefillAdder selection and waiting-queue updates are real.
            batch, running_batch = scheduler._get_new_batch_prefill_raw(
                None, running_batch
            )

        self.assertIsNotNone(batch)
        self.assertEqual(batch.reqs, [followup])
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertFalse(running_batch.batch_is_full)

    def test_oversized_embedding_request_is_finished_before_queueing(self):
        scheduler = object.__new__(Scheduler)
        scheduler.tokenizer = object()
        scheduler.max_req_input_len = 448
        scheduler.waiting_queue = []
        scheduler._add_request_to_queue = scheduler.waiting_queue.append

        recv_req = SimpleNamespace(
            rid="oversized-embedding",
            input_text=None,
            input_ids=array("q", [1] * 448),
            sampling_params=SamplingParams(max_new_tokens=0),
            positional_embed_overrides=None,
            token_type_ids=None,
            routed_dp_rank=None,
            priority=None,
            dimensions=None,
            lora_id=None,
            http_worker_ipc=None,
            time_stats=None,
            return_pooled_hidden_states=False,
            multi_item_delimiter_indices=None,
            token_indices_to_pool=None,
            mm_inputs=None,
        )

        with get_parallel().override(tp_rank=0):
            scheduler.handle_embedding_request(recv_req)

        self.assertEqual(len(scheduler.waiting_queue), 1)
        req = scheduler.waiting_queue[0]
        self.assertEqual(req.rid, "oversized-embedding")
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(req.to_finish.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Input length (448 tokens)", req.to_finish.message)
        self.assertEqual(len(req.origin_input_ids), 1)


if __name__ == "__main__":
    unittest.main()
