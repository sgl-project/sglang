"""Unit tests for the fork primitives: prompt alias, member-row free, reparent."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.fork import (
    alias_members_prompt_kv,
    free_member_rows,
    neutral_member_sampling_params,
    reparent_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestReparentKV(CustomTestCase):
    def setUp(self):
        # 4 rows x 12 positions; every row maps to its own distinct slots.
        self.req_to_token = torch.arange(48, dtype=torch.int64).reshape(4, 12)
        # Two fake per-layer buffers whose content encodes (buffer, slot).
        self.kv_buffers = [
            torch.arange(100, dtype=torch.float32) * 10,
            torch.arange(100, dtype=torch.float32) * 100,
        ]

    def test_copies_suffix_data_only(self):
        mapping_before = self.req_to_token.clone()
        originals = [buf.clone() for buf in self.kv_buffers]

        # Rows 1 and 2 both reparent onto row 0 over suffix [3, 8).
        reparent_kv(
            self.req_to_token,
            self.kv_buffers,
            dst_rows=torch.tensor([1, 2]),
            src_rows=torch.tensor([0, 0]),
            prefix_len=3,
            seq_len=8,
        )

        # Mapping is untouched: only buffer contents move.
        self.assertTrue(torch.equal(self.req_to_token, mapping_before))

        for buf, orig in zip(self.kv_buffers, originals):
            for dst_row in (1, 2):
                for pos in range(3, 8):
                    dst_slot = int(self.req_to_token[dst_row, pos])
                    src_slot = int(self.req_to_token[0, pos])
                    self.assertEqual(buf[dst_slot], orig[src_slot])
                # Prompt region and beyond-suffix region keep their own data.
                for pos in list(range(0, 3)) + list(range(8, 12)):
                    slot = int(self.req_to_token[dst_row, pos])
                    self.assertEqual(buf[slot], orig[slot])
            # Row 3 (not involved) is fully untouched.
            for pos in range(12):
                slot = int(self.req_to_token[3, pos])
                self.assertEqual(buf[slot], orig[slot])

    def test_empty_suffix_is_noop(self):
        originals = [buf.clone() for buf in self.kv_buffers]
        reparent_kv(
            self.req_to_token,
            self.kv_buffers,
            dst_rows=torch.tensor([1]),
            src_rows=torch.tensor([0]),
            prefix_len=5,
            seq_len=5,
        )
        for buf, orig in zip(self.kv_buffers, originals):
            self.assertTrue(torch.equal(buf, orig))


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

    def free_raw(self, indices):
        self.freed.extend(indices)


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, slots):
        self.freed.extend(slots.tolist())


class TestFreeMemberRows(CustomTestCase):
    def _make_group(self, req_to_token, allocated_len):
        leader = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=allocated_len))
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

        free_member_rows(group, pool, allocator)

        # Each member row owns its decode suffix [5, 8); the aliased prompt
        # is the leader's to free.
        expected = req_to_token[1:3, 5:8].flatten().tolist()
        self.assertEqual(sorted(allocator.freed), sorted(expected))
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
