"""Two CPU ranks exercise production CP gather, EP sum and CP take-back.

Run with torch.distributed.run --nproc-per-node=2. Only group construction is
adapted to Gloo; the row transport and expert-output reduction are production.
"""

from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.layers.layer_boundary.ops import moe_cp_gather, moe_cp_take_back
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.runtime_context import get_parallel


class Group:
    def __init__(self):
        self.rank_in_group = dist.get_rank()
        self.world_size = dist.get_world_size()

    def all_gather_into_tensor(self, output, value):
        dist.all_gather_into_tensor(output, value)

    def all_reduce(self, value):
        dist.all_reduce(value)
        return value


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    group = Group()
    # Exercise two zigzag token orders.
    for indices in (
        [torch.tensor([0, 1, 6, 7]), torch.tensor([2, 3, 4, 5])],
        [torch.tensor([0, 1, 5]), torch.tensor([2, 3, 4])],
    ):
        local = indices[rank].float().view(-1, 1)
        rows = [len(i) for i in indices]
        with get_parallel().override(
            moe_dp_group=group, moe_ep_group=group, moe_ep_size=2, moe_tp_size=1
        ):
            gathered = moe_cp_gather(local, rows, 2)
            # Expert 0 doubles even-token values; expert 1 triples odd values.
            # Whole experts live on separate EP ranks and see identical rows.
            routed = gathered.long().remainder(2) == rank
            partial = torch.where(routed, gathered * (rank + 2), 0)
            with patch.object(
                moe_utils, "post_experts_output_is_complete", return_value=False
            ):
                summed = moe_utils.sum_post_experts_output(partial)
            actual = moe_cp_take_back(summed, rows)
            expected = local * (local.long().remainder(2) + 2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rank == 0:
        print("PASS: production CP gather -> whole-expert EP sum -> CP take-back")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
