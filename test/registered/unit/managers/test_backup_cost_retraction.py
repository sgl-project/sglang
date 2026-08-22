import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.retraction_policy import (
    BackupCostCandidate,
    build_backup_cost_retraction_order,
    compute_decode_shortfall,
    make_backup_cost_candidate,
    select_backup_cost_victims,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _candidate(index, cost, relief, priority=None):
    return BackupCostCandidate(
        index=index,
        backup_tokens=cost,
        estimated_relief=relief,
        priority=priority,
    )


def _req(input_len, output_len, *, allocated=None, committed=None, priority=None):
    seqlen = input_len + output_len
    return SimpleNamespace(
        rid=f"req-{input_len}-{output_len}-{priority}",
        origin_input_ids=[0] * input_len,
        output_ids=[0] * output_len,
        seqlen=seqlen,
        kv=SimpleNamespace(
            kv_allocated_len=seqlen - 1 if allocated is None else allocated
        ),
        kv_committed_len=seqlen - 1 if committed is None else committed,
        cache_protected_len=0,
        priority=priority,
        sampling_params=SimpleNamespace(max_new_tokens=128),
    )


def _args(policy="backup-cost", *, priority=False, low_first=False):
    return SimpleNamespace(
        retraction_policy=policy,
        enable_priority_scheduling=priority,
        schedule_low_priority_values_first=low_first,
    )


class TestBackupCostRetraction(CustomTestCase):
    def test_small_deficit_prefers_4k_over_32k(self):
        candidates = [_candidate(0, 4096, 4097), _candidate(1, 32768, 32769)]
        selected = select_backup_cost_victims(candidates, shortfall=1)
        self.assertEqual([candidate.index for candidate in selected], [0])

    def test_smallest_insufficient_uses_smallest_sufficient_candidate(self):
        candidates = [
            _candidate(0, 4, 4),
            _candidate(1, 8, 12),
            _candidate(2, 16, 20),
        ]
        selected = select_backup_cost_victims(candidates, shortfall=10)
        self.assertEqual([candidate.index for candidate in selected], [1])

    def test_multi_victim_fallback_is_deterministic(self):
        candidates = [
            _candidate(2, 4, 5),
            _candidate(0, 6, 6),
            _candidate(1, 8, 7),
        ]
        selected = select_backup_cost_victims(candidates, shortfall=11)
        self.assertEqual([candidate.index for candidate in selected], [2, 0])
        self.assertEqual(
            build_backup_cost_retraction_order(candidates, shortfall=11),
            [2, 0, 1],
        )

    def test_page_alignment_and_non_spec_next_decode_allocation(self):
        candidate = make_backup_cost_candidate(
            index=0,
            sequence_length=18,
            kv_allocated_len=17,
            next_decode_tokens=8,
            page_size=8,
        )
        self.assertEqual(candidate.backup_tokens, 24)
        self.assertEqual(candidate.estimated_relief, 25)

        batch = ScheduleBatch(
            reqs=[
                _req(16, 1, allocated=16, committed=16),
                _req(16, 2, allocated=17, committed=17),
            ],
            token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        self.assertEqual(batch.new_tokens_required_next_decode([0]), 8)
        self.assertEqual(batch.new_tokens_required_next_decode([1]), 0)

    def test_speculative_reserve_is_included_in_relief(self):
        req = _req(14, 1, allocated=16, committed=14)
        batch = ScheduleBatch(
            reqs=[req],
            token_to_kv_pool_allocator=SimpleNamespace(page_size=8),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
        )
        with patch(
            "sglang.srt.managers.schedule_batch.get_alloc_reserve_per_decode",
            return_value=5,
        ):
            next_tokens = batch.new_tokens_required_next_decode([0])
        self.assertEqual(next_tokens, 8)
        candidate = make_backup_cost_candidate(
            index=0,
            sequence_length=req.seqlen,
            kv_allocated_len=req.kv.kv_allocated_len,
            next_decode_tokens=next_tokens,
            page_size=8,
        )
        self.assertEqual(candidate.estimated_relief, 24)

    def test_tie_break_uses_stable_integer_index(self):
        candidates = [_candidate(3, 8, 12), _candidate(1, 8, 12)]
        selected = select_backup_cost_victims(candidates, shortfall=4)
        self.assertEqual([candidate.index for candidate in selected], [1])

    def test_priority_tier_is_not_crossed_for_lower_transfer_cost(self):
        candidates = [
            _candidate(0, 1, 100, priority=0),
            _candidate(1, 100, 100, priority=2),
            _candidate(2, 200, 100, priority=None),
        ]
        selected = select_backup_cost_victims(
            candidates,
            shortfall=1,
            respect_priority=True,
            schedule_low_priority_values_first=True,
        )
        self.assertEqual([candidate.index for candidate in selected], [2])

    def test_legacy_length_and_priority_orders_are_unchanged(self):
        reqs = [
            _req(8, 5, priority=2),
            _req(20, 1, priority=0),
            _req(8, 3, priority=None),
        ]
        self.assertEqual(
            ScheduleBatch._get_decode_retraction_order(reqs, _args("length")),
            [0, 2, 1],
        )
        self.assertEqual(
            ScheduleBatch._get_decode_retraction_order(
                reqs, _args("priority", priority=True, low_first=True)
            ),
            [1, 0, 2],
        )

    def test_zero_shortfall_keeps_test_retract_length_semantics(self):
        reqs = [_req(8, 5), _req(32, 1), _req(8, 3)]
        batch = ScheduleBatch(
            reqs=reqs,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        self.assertEqual(
            batch._get_decode_retraction_order_for_shortfall(
                _args("backup-cost"), shortfall=0
            ),
            [0, 2, 1],
        )

    def test_shortfall_and_post_release_checks_continue_on_estimation_error(self):
        self.assertEqual(compute_decode_shortfall(8, 3), 5)
        self.assertEqual(compute_decode_shortfall(3, 8), 0)

        reqs = [_req(8, 1), _req(16, 1), _req(32, 1)]
        batch = ScheduleBatch(
            reqs=reqs,
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=1, available_size=lambda: 0
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        batch.new_tokens_required_next_decode = MagicMock(return_value=1)
        batch._get_decode_retraction_order_for_shortfall = MagicMock(
            return_value=[2, 1, 0]
        )
        batch.check_decode_mem = MagicMock(side_effect=[False, True, True])
        batch.release_req = MagicMock(return_value=True)
        original_reqs = list(reqs)

        def filter_batch(*, keep_indices):
            batch.reqs = [original_reqs[i] for i in keep_indices]

        batch.filter_batch = MagicMock(side_effect=filter_batch)

        retracted, _, aborted = batch.retract_decode(_args("backup-cost"))

        self.assertEqual(retracted, original_reqs[:2])
        self.assertEqual(aborted, [])
        self.assertEqual(batch.release_req.call_count, 2)
        self.assertEqual(batch.check_decode_mem.call_count, 3)


if __name__ == "__main__":
    unittest.main()
