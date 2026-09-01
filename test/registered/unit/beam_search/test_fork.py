"""Unit tests for the fork primitives: prompt alias, share-on-fork reparent,
orphan reclaim, and group-owned member-row release."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.fork import (
    StagedOrphans,
    alias_members_prompt_kv,
    collect_orphan_slots,
    free_member_rows,
    neutral_member_sampling_params,
    remap_kv_mapping,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRemapKvMapping(CustomTestCase):
    def setUp(self):
        # 3 rows x 10 positions; every row maps to its own distinct slots.
        self.req_to_token = torch.arange(30, dtype=torch.int64).reshape(3, 10)
        self.rows = torch.tensor([0, 1, 2], dtype=torch.int64)

    def test_rows_adopt_parent_slots(self):
        # Survivors 0 and 1 both descend from row 2; row 2 from row 0.
        parent_idx = torch.tensor([2, 2, 0], dtype=torch.int64)
        before = self.req_to_token.clone()

        old_map, new_map = remap_kv_mapping(
            self.req_to_token, self.rows, parent_idx, prefix_len=4, seq_len=7
        )

        # Each row's window now names its parent's slots; nothing outside
        # [4, 7) moved, and no KV data was touched (mapping-only reparent).
        for j, p in enumerate(parent_idx.tolist()):
            self.assertTrue(torch.equal(self.req_to_token[j, 4:7], before[p, 4:7]))
            self.assertTrue(torch.equal(self.req_to_token[j, :4], before[j, :4]))
            self.assertTrue(torch.equal(self.req_to_token[j, 7:], before[j, 7:]))
        self.assertTrue(torch.equal(old_map, before[self.rows, 4:7]))
        self.assertTrue(torch.equal(new_map, before[parent_idx, 4:7]))

    def test_identity_parents_change_nothing(self):
        parent_idx = torch.arange(3, dtype=torch.int64)
        before = self.req_to_token.clone()
        remap_kv_mapping(
            self.req_to_token, self.rows, parent_idx, prefix_len=4, seq_len=7
        )
        self.assertTrue(torch.equal(self.req_to_token, before))


class TestCollectOrphanSlots(CustomTestCase):
    def test_returns_slots_nobody_inherits(self):
        req_to_token = torch.arange(30, dtype=torch.int64).reshape(3, 10)
        rows = torch.tensor([0, 1, 2], dtype=torch.int64)
        # Row 1 is nobody's parent, so its window dies.
        parent_idx = torch.tensor([0, 2, 2], dtype=torch.int64)
        before = req_to_token.clone()

        old_map, new_map = remap_kv_mapping(
            req_to_token, rows, parent_idx, prefix_len=4, seq_len=7
        )
        orphans = collect_orphan_slots(old_map, new_map)

        self.assertEqual(sorted(orphans.tolist()), sorted(before[1, 4:7].tolist()))

    def test_no_orphans_when_every_row_survives(self):
        req_to_token = torch.arange(30, dtype=torch.int64).reshape(3, 10)
        rows = torch.tensor([0, 1, 2], dtype=torch.int64)
        parent_idx = torch.tensor([2, 0, 1], dtype=torch.int64)  # a permutation

        old_map, new_map = remap_kv_mapping(
            req_to_token, rows, parent_idx, prefix_len=4, seq_len=7
        )

        self.assertEqual(collect_orphan_slots(old_map, new_map).numel(), 0)


class TestAliasMembersPromptKV(CustomTestCase):
    def test_alias_mapping(self):
        req_to_token = torch.arange(36, dtype=torch.int64).reshape(3, 12)
        leader_prompt = req_to_token[0, :5].clone()
        tails_before = req_to_token[1:, 5:].clone()

        alias_members_prompt_kv(
            req_to_token,
            dst_rows=torch.tensor([1, 2]),
            leader_row=0,
            prompt_len=5,
        )

        # Prompt indices aliased from the leader; the tails stay member-owned.
        self.assertTrue(torch.equal(req_to_token[1, :5], leader_prompt))
        self.assertTrue(torch.equal(req_to_token[2, :5], leader_prompt))
        self.assertTrue(torch.equal(req_to_token[1:, 5:], tails_before))


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.freed = []

    def free_rows(self, indices):
        self.freed.extend(indices)


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, slots):
        self.freed.extend(slots.tolist())


class TestFreeMemberRows(CustomTestCase):
    def _make_group(self, req_to_token, allocated_len):
        leader = SimpleNamespace(
            kv=ReqKvInfo(
                kv_allocated_len=allocated_len, kv_committed_len=allocated_len
            ),
        )
        return SimpleNamespace(
            leader=leader,
            prompt_len=5,
            member_rows=torch.tensor([1, 2], dtype=torch.int64),
            member_rows_cpu=torch.tensor([1, 2], dtype=torch.int64),
            all_rows=torch.tensor([0, 1, 2], dtype=torch.int64),
        )

    def test_frees_suffix_slots_and_rows(self):
        req_to_token = torch.arange(36, dtype=torch.int64).reshape(3, 12)
        pool = _FakeReqToTokenPool(req_to_token)
        allocator = _FakeAllocator()
        group = self._make_group(req_to_token, allocated_len=8)

        leader = group.leader
        free_member_rows(group, pool, allocator)

        # The group owns the whole decode region [5, 8) across all its rows
        # (leader included) and frees it once.
        expected = req_to_token[0:3, 5:8].flatten().tolist()
        self.assertEqual(sorted(allocator.freed), sorted(expected))
        # Leader rewound to the prompt: its own release must not free the
        # decode region a second time.
        self.assertEqual(leader.kv.kv_allocated_len, 5)
        self.assertEqual(leader.kv.kv_committed_len, 5)
        self.assertEqual(sorted(pool.freed), [1, 2])
        self.assertIsNone(group.member_rows)
        self.assertIsNone(group.member_rows_cpu)
        self.assertIsNone(group.all_rows)

        # Idempotent: a second free is a no-op.
        free_member_rows(group, pool, allocator)
        self.assertEqual(sorted(pool.freed), [1, 2])

    def test_empty_suffix_frees_rows_only(self):
        # Dead leader right after spawn: allocated == prompt, no KV to free.
        req_to_token = torch.arange(36, dtype=torch.int64).reshape(3, 12)
        pool = _FakeReqToTokenPool(req_to_token)
        allocator = _FakeAllocator()
        group = self._make_group(req_to_token, allocated_len=5)

        free_member_rows(group, pool, allocator)

        self.assertEqual(allocator.freed, [])
        self.assertEqual(sorted(pool.freed), [1, 2])


class TestRetireReclaimsStagedOrphans(CustomTestCase):
    """Regression: aborting a group must not leak the orphan slots staged by the
    launch half, which no surviving row names."""

    @staticmethod
    def _make_coordinator(allocator):
        from sglang.srt.beam_search.coordinator import BeamCoordinator

        return BeamCoordinator(
            model_config=None,
            spec_algorithm=None,
            dllm_enabled=False,
            max_req_len=0,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            tree_cache=None,
            future_map=None,
        )

    def test_retract_abort_does_not_leak_staged_orphans(self):
        req_to_token = torch.arange(36, dtype=torch.int64).reshape(3, 12)
        rows = torch.tensor([0, 1, 2], dtype=torch.int64)
        # Row 1 is nobody's parent, so its window [5, 8) -- slots 17, 18, 19 --
        # is orphaned by the remap the launch half already applied.
        old_map, new_map = remap_kv_mapping(
            req_to_token,
            rows,
            torch.tensor([0, 2, 2], dtype=torch.int64),
            prefix_len=5,
            seq_len=8,
        )
        orphans = sorted(collect_orphan_slots(old_map, new_map).tolist())
        self.assertEqual(orphans, [17, 18, 19])

        pool = _FakeReqToTokenPool(req_to_token)
        allocator = _FakeAllocator()
        group = SimpleNamespace(
            leader=SimpleNamespace(
                kv=ReqKvInfo(kv_allocated_len=8, kv_committed_len=8),
            ),
            prompt_len=5,
            member_rows=torch.tensor([1, 2], dtype=torch.int64),
            member_rows_cpu=torch.tensor([1, 2], dtype=torch.int64),
            all_rows=rows,
            pending_orphans=[StagedOrphans(7, old_map, new_map)],
            slots_freed=0,
            retired=False,
            _pending_steps=[],
        )

        # retract_decode's sequence: member rows released without the
        # coordinator, then the scheduler retires the group.
        free_member_rows(group, pool, allocator)
        by_rows = sorted(allocator.freed)
        self.assertEqual(by_rows, [5, 6, 7, 29, 30, 31])
        # The orphans are disjoint from what the rows still name, which is
        # exactly why free_member_rows alone leaks them.
        self.assertFalse(set(orphans) & set(by_rows))

        coordinator = self._make_coordinator(allocator)
        coordinator._num_live_groups = 1
        coordinator._retire_group(group)

        self.assertEqual(sorted(allocator.freed[len(by_rows) :]), orphans)
        self.assertEqual(group.slots_freed, len(orphans))
        self.assertEqual(group.pending_orphans, [])
        self.assertEqual(coordinator._num_live_groups, 0)

        # Retiring twice must not double-free or double-decrement.
        coordinator._retire_group(group)
        self.assertEqual(len(allocator.freed), len(by_rows) + len(orphans))
        self.assertEqual(coordinator._num_live_groups, 0)


class TestNeutralParams(CustomTestCase):
    def test_neutral_params(self):
        from sglang.srt.sampling.sampling_params import SamplingParams

        leader_params = SamplingParams(
            max_new_tokens=8,
            temperature=0.0,
            frequency_penalty=0.5,
            stop_token_ids={7},
        )
        params = neutral_member_sampling_params(leader_params)
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_p, 1.0)
        self.assertEqual(params.frequency_penalty, 0.0)
        self.assertTrue(params.ignore_eos)
        self.assertIsNone(params.stop_token_ids)
        self.assertGreater(params.max_new_tokens, 8)


if __name__ == "__main__":
    unittest.main()
