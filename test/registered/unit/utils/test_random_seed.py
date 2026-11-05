"""Each attention-DP group must sample from its own random stream.

``TpModelWorker`` seeds every rank from the one seed world rank 0 broadcasts.
Attention-DP groups step in lockstep, so with one shared seed the same prompt
served by different groups came back with the same "independent" sample. The
worker now seeds with ``derive_attn_dp_random_seed``; these pin its contract:
distinct across attention-DP groups, equal inside one group, the server's seed
untouched without attention DP.
"""

import unittest

from sglang.srt.runtime_context import derive_attention_ranks
from sglang.srt.utils.common import derive_attn_dp_random_seed
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SEED = 42


def _seed_at(tp_rank: int, *, attn_tp_size: int, attn_dp_size: int) -> int:
    """The seed a worker at ``tp_rank`` derives, with the rank layout the
    runtime context publishes (dp, cp, tp; tp changing fastest)."""
    _, attn_dp_rank = derive_attention_ranks(
        tp_rank=tp_rank, attn_tp_size=attn_tp_size, attn_cp_size=1
    )
    return derive_attn_dp_random_seed(
        SEED, attn_dp_rank=attn_dp_rank, attn_dp_size=attn_dp_size
    )


class TestDeriveAttnDpRandomSeed(CustomTestCase):
    def test_without_attention_dp_the_seed_is_the_servers(self):
        # Plain TP: every rank keeps the broadcast seed, as before.
        for tp_rank in range(8):
            self.assertEqual(_seed_at(tp_rank, attn_tp_size=8, attn_dp_size=1), SEED)

    def test_groups_draw_from_distinct_streams(self):
        # attn_dp=8, attn_tp=1: eight groups, eight different seeds.
        seeds = [_seed_at(r, attn_tp_size=1, attn_dp_size=8) for r in range(8)]
        self.assertEqual(len(set(seeds)), 8)

    def test_group_zero_keeps_the_user_visible_seed(self):
        # The offset is the group rank, so the seed a user set stays the base
        # and the derivation is reproducible across restarts.
        self.assertEqual(_seed_at(0, attn_tp_size=2, attn_dp_size=4), SEED)
        self.assertEqual(
            [_seed_at(r, attn_tp_size=1, attn_dp_size=4) for r in range(4)],
            [SEED + r for r in range(4)],
        )

    def test_ranks_of_one_group_share_a_seed(self):
        # tp=8 as attn_dp=4 x attn_tp=2: speculative decoding and TP sampling
        # need the two ranks of a group to draw the same numbers.
        by_group = {}
        for tp_rank in range(8):
            attn_tp_rank, attn_dp_rank = derive_attention_ranks(
                tp_rank=tp_rank, attn_tp_size=2, attn_cp_size=1
            )
            by_group.setdefault(attn_dp_rank, set()).add(
                _seed_at(tp_rank, attn_tp_size=2, attn_dp_size=4)
            )
        self.assertEqual(len(by_group), 4)
        for attn_dp_rank, seeds in by_group.items():
            self.assertEqual(len(seeds), 1, msg=f"group {attn_dp_rank}: {seeds}")
        self.assertEqual(len({s for seeds in by_group.values() for s in seeds}), 4)


if __name__ == "__main__":
    unittest.main()
