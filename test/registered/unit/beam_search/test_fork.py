"""Unit tests for the fork primitive (S3): spawn, reparent, drift guard."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.fork import (
    FORK_STATE_PLAN,
    PENDING_MEMBER_FIELDS,
    collect_req_state_fields,
    init_member_kv_state,
    neutral_member_sampling_params,
    reparent_kv,
    spawn_member,
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


class TestInitMemberKvState(CustomTestCase):
    def test_alias_mapping_and_bookkeeping(self):
        req_to_token = torch.arange(24, dtype=torch.int64).reshape(2, 12)
        leader_prompt = req_to_token[0, :5].clone()
        member_tail_before = req_to_token[1, 5:].clone()
        member = SimpleNamespace()

        init_member_kv_state(
            member, req_to_token, leader_row=0, member_row=1, prompt_len=5
        )

        # Prompt indices aliased from the leader; the tail stays member-owned.
        self.assertTrue(torch.equal(req_to_token[1, :5], leader_prompt))
        self.assertTrue(torch.equal(req_to_token[1, 5:], member_tail_before))
        # Born-correct linear accounting; prompt is not the member's to free.
        self.assertEqual(member.req_pool_idx, 1)
        self.assertEqual(member.kv.kv_allocated_len, 5)
        self.assertEqual(member.kv_committed_len, 5)
        self.assertEqual(member.cache_protected_len, 5)
        self.assertTrue(member.skip_radix_cache_insert)


class TestForkStatePlan(CustomTestCase):
    def test_every_req_field_is_classified(self):
        fields = set(collect_req_state_fields())
        plan = set(FORK_STATE_PLAN)

        unclassified = fields - plan
        self.assertFalse(
            unclassified,
            f"New Req fields must be classified in FORK_STATE_PLAN: {unclassified}",
        )
        stale = plan - fields
        self.assertFalse(
            stale, f"FORK_STATE_PLAN entries no longer exist on Req: {stale}"
        )
        # Once these become real Req fields, move them into FORK_STATE_PLAN.
        already_real = set(PENDING_MEMBER_FIELDS) & fields
        self.assertFalse(
            already_real,
            f"Pending member fields now exist on Req; reclassify: {already_real}",
        )


class TestSpawnMember(CustomTestCase):
    def _make_leader(self):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        leader = Req(
            rid="leader",
            origin_input_text="",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(
                max_new_tokens=8,
                temperature=0.0,
                frequency_penalty=0.5,
                stop_token_ids={7},
            ),
        )
        leader.vocab_size = 1000
        return leader

    def test_neutral_params(self):
        leader = self._make_leader()
        params = neutral_member_sampling_params(leader.sampling_params)
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_p, 1.0)
        self.assertEqual(params.frequency_penalty, 0.0)
        self.assertTrue(params.ignore_eos)
        self.assertIsNone(params.stop_token_ids)
        self.assertGreater(params.max_new_tokens, 8)

    def test_spawn_member(self):
        leader = self._make_leader()
        member = spawn_member(leader, first_token=42, member_index=1)

        self.assertEqual(member.rid, "leader#beam1")
        self.assertIs(member.origin_input_ids, leader.origin_input_ids)
        self.assertEqual(list(member.output_ids), [42])
        self.assertEqual(member.vocab_size, 1000)
        self.assertEqual(member.sampling_params.temperature, 1.0)
        self.assertTrue(member.sampling_params.ignore_eos)
        self.assertFalse(member.stream)
        self.assertFalse(member.return_logprob)


if __name__ == "__main__":
    unittest.main()
