import unittest

import torch

from sglang.kernels.ops.attention.dsv4.index_page_slots import index_page_slots
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference(raw, starts, slots):
    columns = (raw + starts[:, None]).clamp_min(0)
    return torch.where(raw >= 0, slots[columns], -1).to(torch.int32)


class TestIndexPageSlots(CustomTestCase):
    def test_matches_reference(self):
        torch.manual_seed(7)
        for rows, width in ((1, 1), (3, 7), (100, 64), (16, 512), (5, 1000), (2, 2048)):
            lens = torch.randint(1, 300, (rows,), device="cuda")
            starts = (torch.cumsum(lens, 0) - lens).to(torch.int32)
            columns = int(lens.sum())
            slots = torch.randint(
                0, 1 << 20, (columns,), device="cuda", dtype=torch.int64
            )
            raw = torch.randint(0, 300, (rows, width), device="cuda", dtype=torch.int32)
            raw = torch.minimum(raw, (lens - 1)[:, None].to(torch.int32))
            raw[torch.rand(rows, width, device="cuda") < 0.3] = -1
            out = torch.full((rows + 2, width + 8), 7, device="cuda", dtype=torch.int32)
            index_page_slots(raw, starts, slots, out[:rows])
            torch.testing.assert_close(
                out[:rows, :width], _reference(raw, starts, slots)
            )
            self.assertTrue(bool((out[:rows, width:] == 7).all()))
            self.assertTrue(bool((out[rows:] == 7).all()))

    def test_empty(self):
        raw = torch.empty((0, 16), device="cuda", dtype=torch.int32)
        out = torch.empty((0, 16), device="cuda", dtype=torch.int32)
        index_page_slots(
            raw, raw.new_empty(0), raw.new_empty(0, dtype=torch.int64), out
        )


if __name__ == "__main__":
    unittest.main()
