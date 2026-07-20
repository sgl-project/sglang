import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.swa_recompute import resolve_swa_recompute_prefix
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestPrefillAdder(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_token_allocator = self.create_token_allocator()

    def create_tree_cache(
        self,
        *,
        full_evictable_size: int = 0,
        swa_evictable_size: int = 0,
        evictable_size: int = 0,
    ) -> MagicMock:
        tree_cache = MagicMock()
        tree_cache.root_node = MagicMock(name="root_node")
        tree_cache.full_evictable_size.return_value = full_evictable_size
        tree_cache.swa_evictable_size.return_value = swa_evictable_size
        tree_cache.evictable_size.return_value = evictable_size
        tree_cache.disable = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.inc_lock_ref_excluding_components.side_effect = (
            lambda _node, skipped_components: IncLockRefResult(
                skipped_components=set(skipped_components)
            )
        )
        tree_cache.prefix_lock_exclusions.side_effect = lambda req: (
            {"swa"} if req.extra_compute_prefix_len > 0 else set()
        )

        def inc_request_lock_ref(req):
            result = tree_cache.inc_lock_ref_excluding_components(
                req.last_node, tree_cache.prefix_lock_exclusions(req)
            )
            req.swa_prefix_lock_released = bool(result.skipped_components)
            return result

        tree_cache.inc_request_lock_ref.side_effect = inc_request_lock_ref
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        tree_cache.is_tree_cache.return_value = False
        tree_cache.supports_mamba.return_value = False
        tree_cache.resolve_extra_compute_prefix.side_effect = (
            lambda req: resolve_swa_recompute_prefix(req, tree_cache)
        )
        return tree_cache

    def create_token_allocator(
        self,
        *,
        full_available_size: int = 0,
        swa_available_size: int = 0,
        available_size: int = 0,
    ) -> MagicMock:
        allocator = MagicMock()
        allocator.full_available_size.return_value = full_available_size
        allocator.swa_available_size.return_value = swa_available_size
        allocator.available_size.return_value = available_size
        return allocator

    def create_running_batch(self, reqs=None) -> MagicMock:
        batch = MagicMock()
        batch.reqs = list(reqs or [])
        batch.release_req.return_value = None
        batch.filter_batch.return_value = None
        return batch

    def create_server_args(
        self, *, schedule_low_priority_values_first: bool
    ) -> MagicMock:
        server_args = MagicMock()
        server_args.schedule_low_priority_values_first = (
            schedule_low_priority_values_first
        )
        return server_args

    def create_mock_req(self, rid, priority, max_new_tokens, output_len=0, wait_time=0):
        req = MagicMock(spec=Req)
        req.rid = str(rid)
        req.priority = priority
        req.prefix_indices = []
        req.full_untruncated_fill_ids = []
        req.output_ids = [0] * output_len
        req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
        req.time_stats = SimpleNamespace(wait_queue_entry_time=wait_time)
        req.retracted_stain = False
        req.extra_compute_prefix_len = 0
        req.finished.return_value = False
        req.needs_host_load_back.return_value = False
        return req

    def create_adder(self, running_batch, **kwargs):
        defaults = dict(
            page_size=1,
            tree_cache=self.mock_tree_cache,
            token_to_kv_pool_allocator=self.mock_token_allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=None,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        defaults.update(kwargs)
        return PrefillAdder(**defaults)

    def test_preempt_success_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)  # 50 + 75 + 100 - 50 = 175
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 125)  # 50 + 75 + 100 - 100 = 125
        running_batch.release_req.assert_called_once()

    def test_preempt_fail_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=2, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_fail_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=0, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=-1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_skip_already_preempted_request(self):
        params = [
            ("req_prio_0", 0, 50),
            ("req_prio_1", 1, 75),
            ("req_prio_2", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = 225
        self.mock_token_allocator.available_size.return_value = 225

        # New request preempts req_prio_0
        first_req = self.create_mock_req(
            "new_req_prio_1", priority=1, max_new_tokens=49
        )
        first_success = adder.preempt_to_schedule(first_req, mock_server_args)
        self.assertTrue(first_success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)
        running_batch.release_req.assert_called_once()

        # Second call needs more tokens than currently free, so it would need to
        # preempt req_prio_0 again if already-preempted requests were not filtered out.
        second_req = self.create_mock_req(
            "second_new_req_prio_1", priority=1, max_new_tokens=76
        )
        second_success = adder.preempt_to_schedule(second_req, mock_server_args)

        self.assertFalse(second_success)
        self.assertEqual(adder.rem_total_token_offset, 175)
        self.assertEqual(adder.preempt_list.count(running_reqs[0]), 1)
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_once(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=75)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 375
        )  # 50 + 75 + 100 + 125 + 125 - 100 = 375
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_twice(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=200)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertIn(running_reqs[3], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 250
        )  # 50 + 75 + 100 + 125 + 125 - 100 - 125 = 250
        self.assertEqual(running_batch.release_req.call_count, 2)

    def test_mixed_chunk_prefill_budgets(self):
        self.mock_token_allocator.available_size.return_value = 1000

        decode_reqs = [
            self.create_mock_req(f"decode_{i}", priority=0, max_new_tokens=50)
            for i in range(8)
        ]
        running_batch = self.create_running_batch(decode_reqs)

        adder = self.create_adder(
            running_batch,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(decode_reqs),
        )

        self.assertEqual(adder.rem_input_tokens, 192)  # 200 - 8
        self.assertEqual(adder.rem_chunk_tokens, 56)  # 64 - 8
        self.assertEqual(adder.rem_total_token_offset, 408)  # 8 + 8 * 50
        self.assertEqual(adder.cur_rem_token_offset, 8)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

        # Add a prefill that exactly consumes the chunk budget
        req1 = self.create_mock_req("req1", priority=0, max_new_tokens=64)
        req1.host_hit_length = 0
        req1.prefix_indices = []
        req1.full_untruncated_fill_ids = list(range(56))
        req1.last_node = MagicMock()
        req1.sampling_params.ignore_eos = False

        result1 = adder.add_one_req(
            req1, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(adder.rem_chunk_tokens, 0)  # 56 - 56
        self.assertEqual(adder.rem_input_tokens, 136)  # 192 - 56
        self.assertEqual(result1, AddReqResult.OTHER)

        # 3 decode requests finished
        remaining_decode_reqs = decode_reqs[3:]
        running_batch2 = self.create_running_batch(remaining_decode_reqs)

        adder2 = self.create_adder(
            running_batch2,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(remaining_decode_reqs),
        )

        self.assertEqual(adder2.rem_input_tokens, 195)  # 200 - 5
        self.assertEqual(adder2.rem_chunk_tokens, 59)  # 64 - 5
        self.assertEqual(adder2.rem_total_token_offset, 255)  # 5 + 5 * 50
        self.assertEqual(adder2.budget_state(), AddReqResult.CONTINUE)

        # Same prefill no longer exhausts the chunk budget
        req2 = self.create_mock_req("req2", priority=0, max_new_tokens=64)
        req2.host_hit_length = 0
        req2.prefix_indices = []
        req2.full_untruncated_fill_ids = list(range(56))
        req2.last_node = MagicMock()
        req2.sampling_params.ignore_eos = False

        result2 = adder2.add_one_req(
            req2, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 1)
        self.assertEqual(adder2.rem_chunk_tokens, 3)  # 59 - 56 = 3 remaining
        self.assertEqual(result2, AddReqResult.CONTINUE)

        # Fit last small prefill request
        req3 = self.create_mock_req("req3", priority=0, max_new_tokens=16)
        req3.host_hit_length = 0
        req3.prefix_indices = []
        req3.full_untruncated_fill_ids = list(range(3))
        req3.last_node = MagicMock()
        req3.sampling_params.ignore_eos = False

        result3 = adder2.add_one_req(
            req3, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 2)
        self.assertEqual(adder2.rem_chunk_tokens, 0)  # 3 - 3 = 0
        self.assertEqual(result3, AddReqResult.OTHER)

    def _build_hybrid_swa_chunked_req(
        self,
        *,
        page_size,
        rem_swa,
        rem_chunk=2048,
        extend_input_len=500,
        is_hybrid_swa=True,
        full_available=100_000,
    ):
        self.mock_token_allocator.swa_available_size.return_value = rem_swa
        self.mock_token_allocator.full_available_size.return_value = full_available
        self.mock_token_allocator.available_size.return_value = full_available
        self.mock_tree_cache.sliding_window_size = 128
        adder = self.create_adder(
            self.create_running_batch(),
            page_size=page_size,
            rem_chunk_tokens=rem_chunk,
        )
        adder.is_hybrid_swa = is_hybrid_swa

        req = self.create_mock_req("chunked", priority=0, max_new_tokens=128)
        req.prefix_indices = []
        req.full_untruncated_fill_ids = list(range(extend_input_len))
        # set_extend_range is the only writer of extend_range; the production
        # path reads req.extend_range.length right after calling it, so the mock
        # must actually set the attribute (a spec=Req mock has the method but
        # not the instance attribute).
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        return adder, req

    def test_add_chunked_req_hybrid_swa_reserves_page_for_alloc_extend(self):
        # alloc_extend needs extend_num_tokens + page_size per request. If the
        # scheduler hands out all of rem_swa_tokens, alloc_extend cannot get its
        # extra page and OOMs. With the fix, extend_input_len must cap at
        # rem_swa_tokens - page_size so the page is reserved.
        PAGE_SIZE = 64
        REM_SWA = 100
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=REM_SWA
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)  # truncated → chunked prefill continues
        req.set_extend_range.assert_called_once()
        start, end = req.set_extend_range.call_args.args
        new_len = end - start
        self.assertLessEqual(new_len + PAGE_SIZE, REM_SWA)
        self.assertEqual(new_len, REM_SWA - PAGE_SIZE)

    def test_add_chunked_req_hybrid_swa_defers_when_swa_below_page(self):
        # When rem_swa_tokens <= page_size there is no room to serve even the
        # reservation, so the chunked req must be deferred (returned unchanged)
        # instead of falling back to rem_chunk_tokens and bypassing SWA budget.
        PAGE_SIZE = 64
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=PAGE_SIZE
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)
        req.set_extend_range.assert_not_called()
        self.assertEqual(len(adder.can_run_list), 0)

    def test_swa_budget_for_req(self):
        cases = [
            # (extend, rem_chunk, window, page, recompute, expected, label)
            (64, None, 128, 16, 0, 128 + 16, "no_cap_floor_active"),
            (200, None, 256, 32, 0, 256 + 32, "no_cap_floor_active_other_dims"),
            (300, None, 128, 16, 0, 300 + 16, "no_cap_floor_inactive"),
            (200, 50, 64, 8, 0, 64 + 8, "cap_binds_then_floor"),
            (300, 500, 64, 64, 0, 300 + 64, "cap_does_not_bind"),
            (0, None, 128, 16, 0, 128 + 16, "extend_zero_floor_only"),
            (32, None, 128, 16, 96, 128 + 16, "recompute_inside_window_floor"),
            (64, None, 128, 16, 96, 64 + 96 + 16, "recompute_exceeds_floor"),
            (300, None, 128, 16, 64, 300 + 64 + 16, "recompute_extends_live_tail"),
        ]
        for extend, rem_chunk, window, page, recompute, expected, label in cases:
            with self.subTest(label=label):
                self.mock_tree_cache.sliding_window_size = window
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=page,
                    rem_chunk_tokens=rem_chunk,
                )
                self.assertEqual(
                    adder._swa_budget_for_req(
                        extend, extra_compute_prefix_len=recompute
                    ),
                    expected,
                )

    def _make_prefill_req(
        self,
        *,
        rid: str,
        prefix_len: int = 0,
        extend_input_len: int = 64,
        swa_recompute_len: int = 0,
    ):
        req = self.create_mock_req(rid, priority=0, max_new_tokens=16)
        req.prefix_indices = list(range(prefix_len))
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        req.full_untruncated_fill_ids = list(range(prefix_len + extend_input_len))
        req.extend_range = Range(prefix_len, prefix_len + extend_input_len)
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        req.last_node = MagicMock()
        req.sampling_params.ignore_eos = False
        req.extra_compute_prefix_len = swa_recompute_len
        return req

    def _make_tensor_prefill_req(
        self,
        *,
        rid: str,
        prefix_len: int,
        extend_input_len: int,
        host_hit_length: int = 0,
        swa_recompute_len: int = 0,
    ):
        req = self._make_prefill_req(
            rid=rid,
            prefix_len=0,
            extend_input_len=extend_input_len,
            swa_recompute_len=swa_recompute_len,
        )
        req.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
        req.host_hit_length = host_hit_length
        req.swa_host_hit_length = 0
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.num_matched_prefix_tokens = prefix_len + host_hit_length
        req.full_untruncated_fill_ids = list(range(prefix_len + extend_input_len))
        req.extend_range = Range(prefix_len, prefix_len + extend_input_len)
        req.cache_protected_len = prefix_len
        req.last_node = MagicMock(name=f"{rid}_last_node")
        req.last_host_node = MagicMock(name=f"{rid}_last_host_node")
        req.best_match_node = MagicMock(name=f"{rid}_best_match_node")

        req.needs_host_load_back.return_value = host_hit_length > 0
        return req

    def test_swa_recompute_falls_back_if_load_back_cannot_cover_window(self):
        self.mock_token_allocator.full_available_size.return_value = 100_000
        self.mock_token_allocator.swa_available_size.return_value = 100_000
        self.mock_token_allocator.available_size.return_value = 100_000
        self.mock_tree_cache.sliding_window_size = 128
        adder = self.create_adder(self.create_running_batch(), page_size=16)
        adder.is_hybrid_swa = True

        req = self._make_tensor_prefill_req(
            rid="loadback-fail",
            prefix_len=64,
            extend_input_len=256,
            host_hit_length=128,
            swa_recompute_len=128,
        )
        self.mock_tree_cache.init_load_back.return_value = (
            torch.empty((0,), dtype=torch.int64),
            req.last_node,
        )

        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(req.extra_compute_prefix_len, 0)
        self.assertEqual(len(req.prefix_indices), 0)
        self.assertEqual(req.extend_range, Range(0, len(req.full_untruncated_fill_ids)))
        self.assertEqual(req.last_node, self.mock_tree_cache.root_node)
        self.assertEqual(req.host_hit_length, 0)
        self.assertEqual(adder.can_run_list, [req])

    def test_swa_budget_excludes_full_host_hit_only_for_extra_compute(self):
        self.mock_token_allocator.full_available_size.return_value = 100_000
        self.mock_token_allocator.swa_available_size.return_value = 200
        self.mock_token_allocator.available_size.return_value = 100_000
        self.mock_tree_cache.sliding_window_size = 128
        host_hit = 896
        for extra_compute_len, expected in (
            (0, AddReqResult.NO_TOKEN),
            (96, AddReqResult.CONTINUE),
        ):
            with self.subTest(extra_compute_len=extra_compute_len):
                adder = self.create_adder(self.create_running_batch(), page_size=16)
                adder.is_hybrid_swa = True
                req = self._make_tensor_prefill_req(
                    rid=f"host-hit-budget-{extra_compute_len}",
                    prefix_len=512,
                    extend_input_len=host_hit + 64,
                    host_hit_length=host_hit,
                    swa_recompute_len=extra_compute_len,
                )
                self.mock_tree_cache.init_load_back.reset_mock()
                self.mock_tree_cache.init_load_back.return_value = (
                    torch.arange(host_hit, dtype=torch.int64),
                    req.best_match_node,
                )

                result = adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )

                self.assertEqual(result, expected)
                self.assertEqual(
                    self.mock_tree_cache.init_load_back.called,
                    extra_compute_len > 0,
                )

                if extra_compute_len > 0:
                    self.assertEqual(req.extra_compute_prefix_len, extra_compute_len)
                    self.assertEqual(
                        req.extend_range,
                        Range(512 + host_hit, 512 + host_hit + 64),
                    )
                    self.assertEqual(len(req.prefix_indices), 512 + host_hit)
                    self.assertEqual(adder.can_run_list, [req])

    def test_swa_recompute_mixed_batch_allowed(self):
        self.mock_token_allocator.full_available_size.return_value = 100_000
        self.mock_token_allocator.swa_available_size.return_value = 100_000
        self.mock_token_allocator.available_size.return_value = 100_000
        self.mock_tree_cache.sliding_window_size = 128
        adder = self.create_adder(self.create_running_batch(), page_size=16)
        adder.is_hybrid_swa = True

        normal_req = self._make_prefill_req(rid="normal")
        recompute_req = self._make_prefill_req(
            rid="recompute", prefix_len=256, swa_recompute_len=128
        )

        first = adder.add_one_req(
            normal_req, has_chunked_req=False, truncation_align_size=None
        )
        second = adder.add_one_req(
            recompute_req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(first, AddReqResult.CONTINUE)
        self.assertEqual(second, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [normal_req, recompute_req])

    def test_add_chunked_req_non_hybrid_no_swa_reservation(self):
        # Non-hybrid path: the SWA-pool reservation must NOT apply, otherwise
        # the fix would regress non-SWA models.
        PAGE_SIZE = 16
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE,
            rem_swa=10,
            rem_chunk=500,
            extend_input_len=200,
            is_hybrid_swa=False,
            full_available=300,
        )

        result = adder.add_chunked_req(req)
        self.assertIsNone(result)
        req.set_extend_range.assert_called_once_with(0, 200)
        self.assertIn(req, adder.can_run_list)


if __name__ == "__main__":
    unittest.main()
