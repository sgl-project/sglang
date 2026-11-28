"""Unit tests for srt/mem_cache/in_batch_prefix_sharing.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.in_batch_prefix_sharing import (
    InBatchPrefixSharing,
    PrefixShare,
    borrowed_prefix_indices,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 4


def _waiting_req(tree, rid, token_ids, *, cache_salt=None):
    """A waiting request as the scheduler hands it to admission: matched
    against the tree."""
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", token_ids),
        sampling_params=SamplingParams(max_new_tokens=1),
        cache_salt=cache_salt,
    )
    req.init_next_round_input(tree)
    return req


def _admit(planner, req, *, chunk_end=None):
    """Run ``req`` through admission as PrefillAdder would: it takes its prompt,
    or up to ``chunk_end`` for a chunk, past its (possibly raised) prefix."""
    planner.propose(req)
    req.extend_end = len(req.origin_input_ids) if chunk_end is None else chunk_end
    planner.settle(req, admitted=True)
    return req


class TestPrefixSharePlanner(CustomTestCase):
    def setUp(self):
        self.tree = RadixCache.create_simulated(page_size=PAGE_SIZE)
        self.planner = InBatchPrefixSharing(
            tree_cache=self.tree, page_size=PAGE_SIZE
        ).new_planner()

    def test_followers_take_whole_pages_and_leave_the_last_token(self):
        # 11 tokens: the last one is recomputed for the logits, and the
        # partial page [8, 10) is not shared.
        prompt = list(range(100, 111))
        reqs = [
            _admit(self.planner, _waiting_req(self.tree, f"r{i}", prompt))
            for i in range(3)
        ]
        self.assertEqual([req.prefix_len for req in reqs], [0, 8, 8])
        self.assertEqual(
            self.planner.shares(reqs),
            (
                PrefixShare(follower=1, leader=0, length=8),
                PrefixShare(follower=2, leader=0, length=8),
            ),
        )

    def test_share_stops_at_the_pages_the_leader_chunk_computes(self):
        prompt = list(range(100, 117))
        leader = _admit(
            self.planner, _waiting_req(self.tree, "leader", prompt), chunk_end=9
        )
        follower = _admit(self.planner, _waiting_req(self.tree, "follower", prompt))
        self.assertEqual(
            self.planner.shares([leader, follower]),
            (PrefixShare(follower=1, leader=0, length=8),),
        )

    def test_rejected_request_keeps_its_own_prefix(self):
        """A request admission turns away stays queued with the prefix the tree
        gave it, and shares nothing."""
        prompt = list(range(100, 111))
        leader = _admit(self.planner, _waiting_req(self.tree, "leader", prompt))
        rejected = _waiting_req(self.tree, "rejected", prompt)
        self.planner.propose(rejected)
        self.planner.settle(rejected, admitted=False)
        self.assertEqual(rejected.prefix_len, 0)
        self.assertEqual(self.planner.shares([leader]), ())

    def test_different_cache_salts_do_not_share(self):
        prompt = list(range(100, 111))
        reqs = [
            _admit(
                self.planner,
                _waiting_req(self.tree, f"v{i}", prompt, cache_salt=f"policy-v{i}"),
            )
            for i in range(2)
        ]
        self.assertEqual(self.planner.shares(reqs), ())

    def test_follower_picks_the_leader_with_the_longest_common_prefix(self):
        # The second request shares only the system prompt with the first, and
        # leads the third past it.
        system = list(range(100, 104))
        short = system + list(range(200, 207))
        long = system + list(range(300, 312))
        reqs = [
            _admit(self.planner, _waiting_req(self.tree, rid, tokens))
            for rid, tokens in [("short", short), ("long", long), ("again", long)]
        ]
        self.assertEqual(
            self.planner.shares(reqs),
            (
                PrefixShare(follower=1, leader=0, length=4),
                PrefixShare(follower=2, leader=1, length=12),
            ),
        )

    def test_stale_leader_match_shares_nothing(self):
        """A leader whose span the tree got after its match (a preemption during
        admission releases KV into it) would, on publish, free its own slots for
        that span while its forward still writes them; its followers compute
        their prefix themselves instead."""
        prompt = list(range(100, 111))
        leader = _admit(self.planner, _waiting_req(self.tree, "leader", prompt))
        follower = _admit(self.planner, _waiting_req(self.tree, "follower", prompt))
        self.assertEqual(follower.prefix_len, 8)
        self.tree.insert(
            InsertParams(
                key=RadixKey(array("q", prompt[:PAGE_SIZE])),
                value=torch.arange(PAGE_SIZE, dtype=torch.int64),
            )
        )
        self.assertEqual(self.planner.shares([leader, follower]), ())
        self.assertEqual(follower.prefix_len, 0)


class TestBorrowedPrefixIndices(CustomTestCase):
    def test_follower_row_is_leader_prefix_then_leader_slots(self):
        # Row 0 is unrelated. Row 1 (a leader) has a 4-token cached prefix and
        # computes 7 tokens; row 2 takes [0, 8) of row 1 and computes 4; row 3
        # takes [0, 12) of row 2, whose own prefix is borrowed.
        empty = torch.empty(0, dtype=torch.int64)
        prefix_indices = [empty, torch.tensor([40, 41, 42, 43]), empty, empty]
        out_cache_loc = torch.arange(1000, 1000 + 3 + 7 + 4 + 1)
        result = borrowed_prefix_indices(
            prefix_indices=prefix_indices,
            out_cache_loc=out_cache_loc,
            prefix_lens=[0, 4, 8, 12],
            extend_lens=[3, 7, 4, 1],
            shares=(
                PrefixShare(follower=2, leader=1, length=8),
                PrefixShare(follower=3, leader=2, length=12),
            ),
        )
        self.assertEqual(result[2].tolist(), [40, 41, 42, 43, 1003, 1004, 1005, 1006])
        self.assertEqual(
            result[3].tolist(),
            [40, 41, 42, 43, 1003, 1004, 1005, 1006, 1010, 1011, 1012, 1013],
        )


if __name__ == "__main__":
    unittest.main()
