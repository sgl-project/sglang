import logging
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSchedulerInitReqMaxNewTokens(unittest.TestCase):
    """Property tests for Scheduler.init_req_max_new_tokens.

    Rules enforced when clipping a request's max_new_tokens:
      1. context: input_len + max_new_tokens < max_req_len
      2. admission budget (PrefillAdder):
         ceil_page(input_len) + max_new_tokens + page_size < max_total_num_tokens
      3. env limit: <= SGLANG_MAX_NEW_TOKENS_LIMIT when set and positive
      4. never above the requested value
      5. min_new_tokens <= max_new_tokens afterwards

    Each case asserts all rules hold and the result is tight: one more token
    would violate a rule or exceed the request. Over-long inputs degenerate to
    max_new_tokens = 0 and are rejected by later admission checks.
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
        # The scheduler scales the budget by the live DCP size
        # (`get_parallel().attn_dcp_size`), so the double states a topology
        # rather than publishing a config it does not otherwise need.
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def _new_scheduler(
        self,
        max_req_len: int = 128,
        max_total_num_tokens: int = 1024,
        page_size: int = 1,
    ) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = max_req_len
        scheduler.max_total_num_tokens = max_total_num_tokens
        scheduler.page_size = page_size
        scheduler.max_new_tokens_limit = envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()
        scheduler.token_to_kv_pool_allocator = None
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
        paged_input_len = -(-input_len // page_size) * page_size
        limit = scheduler.max_new_tokens_limit
        limit_active = limit is not None and limit > 0

        def satisfies_rules(candidate: int) -> bool:
            context_ok = input_len + candidate < scheduler.max_req_len
            budget_ok = (
                paged_input_len + candidate + page_size < scheduler.max_total_num_tokens
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
            head_dim=4,
            v_head_dim=4,
            swa_head_num=1,
            swa_head_dim=4,
            swa_v_head_dim=4,
            page_size=4,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            total_bytes=384,
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


if __name__ == "__main__":
    unittest.main()
