import unittest

import torch

from sglang.kernels.ops.moe.eplb_map_record import (
    eplb_map_and_record,
    record_physical_counts,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestEplbMapRecord(CustomTestCase):
    def test_map_and_record_filters_padding(self):
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
        dispatch = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="cuda")
        load = torch.zeros(4, dtype=torch.int32, device="cuda")
        valid_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")
        output = eplb_map_and_record(topk_ids, dispatch, load, valid_tokens)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(output.cpu(), torch.tensor([[2, 0], [3, 1]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(load.cpu(), torch.tensor([1, 0, 1, 0], dtype=torch.int32))
        )

    def test_record_physical_counts_skips_invalid_ids(self):
        topk_ids = torch.tensor(
            [[-1, 0, 3, 4], [3, 9, 0, 3]], dtype=torch.int64, device="cuda"
        )
        load = torch.ones(4, dtype=torch.int32, device="cuda")
        record_physical_counts(topk_ids, load)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(load.cpu(), torch.tensor([3, 1, 1, 4], dtype=torch.int32))
        )


if __name__ == "__main__":
    unittest.main()
