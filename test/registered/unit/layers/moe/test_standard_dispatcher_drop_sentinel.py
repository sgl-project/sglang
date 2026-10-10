"""``StandardDispatcher.dispatch`` must not let topk -1 alias a real expert.

``local_expert_mapping`` is a length-``num_experts`` table filled with -1,
then overwritten on this rank's slice with local ids. On the last EP rank
the last table entry is a real local expert, so ``mapping[topk_ids]`` turns
the -1 drop sentinel into that expert. Guard: clamp/restore keeps -1 as -1
while still translating valid global ids.
"""

from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_NUM_EXPERTS = 8
_NUM_LOCAL = 2
_LAST_RANK = 3


def _last_rank_mapping():
    mapping = torch.full((_NUM_EXPERTS,), -1, dtype=torch.int32)
    mapping[_LAST_RANK * _NUM_LOCAL : (_LAST_RANK + 1) * _NUM_LOCAL] = torch.arange(
        _NUM_LOCAL, dtype=torch.int32
    )
    return mapping


def _dispatcher(mapping: torch.Tensor) -> StandardDispatcher:
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 4
    dispatcher.skip_local_expert_mapping = False
    dispatcher.use_aiter_moe_runner = False
    dispatcher.local_expert_mapping = mapping
    dispatcher.expert_mask_gpu = None
    return dispatcher


class TestStandardDispatcherDropSentinel(CustomTestCase):
    def test_neg1_topk_ids_do_not_map_to_last_rank_local_expert(self):
        mapping = _last_rank_mapping()
        # The wrap that used to fire: mapping[-1] == last local expert (1).
        self.assertEqual(int(mapping[-1]), _NUM_LOCAL - 1)

        topk_ids = torch.tensor(
            [
                [7, -1],
                [6, 0],
                [-1, -1],
            ],
            dtype=torch.int32,
        )
        hidden = torch.zeros((topk_ids.shape[0], 4), dtype=torch.float32)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
            topk_ids=topk_ids,
            router_logits=None,
        )
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.standard.should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ):
            out = _dispatcher(mapping).dispatch(hidden, topk_output)

        got = out.topk_output.topk_ids
        # Valid last-rank experts 6/7 map to local 0/1; -1 stays -1;
        # expert 0 is not on this rank so it stays the mapping miss (-1).
        want = torch.tensor(
            [
                [1, -1],
                [0, -1],
                [-1, -1],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(got, want), f"got {got.tolist()}")
        self.assertTrue(torch.all(got[topk_ids < 0] == -1))


if __name__ == "__main__":
    import unittest

    unittest.main()
