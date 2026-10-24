"""The DCP gather plan reorders every rank's owned rows into position order and remaps top-k slots."""

import unittest

import torch

from sglang.srt.layers.attention.dsa.dsa_dcp_gather import (
    build_dcp_gather_plan,
    remap_slots_to_gathered_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP = 4
PAGE = 64
ROWS = 2048


def _random_widened_table(rng, bs):
    """Each request takes whole widened pages (64 * DCP consecutive slots) at random page ids."""
    lens = torch.tensor([int(rng.integers(1, 600)) for _ in range(bs)])
    table = torch.full((bs, int(lens.max())), -1, dtype=torch.int32)
    free_pages = list(rng.permutation(ROWS * DCP // (PAGE * DCP)))
    for b in range(bs):
        pos = 0
        while pos < int(lens[b]):
            page = int(free_pages.pop())
            n = min(PAGE * DCP, int(lens[b]) - pos)
            table[b, pos : pos + n] = torch.arange(
                page * PAGE * DCP, page * PAGE * DCP + n
            )
            pos += n
    return table, lens


class TestDcpGatherPlan(unittest.TestCase):
    def test_gather_matches_owner_rule(self):
        rng = __import__("numpy").random.default_rng(0)
        rank_rows = [
            torch.randint(0, 255, (ROWS, 8), dtype=torch.uint8) for _ in range(DCP)
        ]
        for _ in range(20):
            table, lens = _random_widened_table(rng, bs=int(rng.integers(1, 5)))
            slot_maps = [
                torch.full((ROWS * DCP,), -1, dtype=torch.int32) for _ in range(DCP)
            ]
            plans = [
                build_dcp_gather_plan(
                    page_table=table,
                    seq_lens_cpu=lens,
                    dcp_size=DCP,
                    dcp_rank=r,
                    physical_page_size=PAGE,
                    slot_to_row=slot_maps[r],
                )
                for r in range(DCP)
            ]
            n_max = plans[0].send_rows.numel()
            recv = torch.cat(
                [rank_rows[r].index_select(0, plans[r].send_rows) for r in range(DCP)]
            )
            for r in range(DCP):
                plan = plans[r]
                out = torch.empty((plan.num_rows, 8), dtype=torch.uint8)
                torch.index_select(recv, 0, plan.recv_rows, out=out[: plan.num_tokens])
                t = 0
                for b in range(table.shape[0]):
                    for pos in range(int(lens[b])):
                        slot = int(table[b, pos])
                        self.assertTrue(
                            torch.equal(out[t], rank_rows[slot % DCP][slot // DCP])
                        )
                        self.assertEqual(int(plan.page_table[b, pos]), t)
                        self.assertEqual(int(plan.slot_to_row[slot]), t)
                        t += 1
                self.assertEqual(plan.num_rows % PAGE, 0)
                self.assertLess(int(plan.recv_rows.max()), DCP * n_max)

    def test_remap_keeps_padding(self):
        slot_to_row = torch.full((64,), -1, dtype=torch.int32)
        slot_to_row[torch.tensor([5, 9, 13])] = torch.tensor(
            [0, 1, 2], dtype=torch.int32
        )
        topk = torch.tensor([[13, -1, 5, 9, -1]], dtype=torch.int32)
        out = remap_slots_to_gathered_rows(topk, slot_to_row)
        self.assertEqual(out.tolist(), [[2, -1, 0, 1, -1]])
        self.assertEqual(out.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
