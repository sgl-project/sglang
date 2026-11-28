import unittest

import torch

from sglang.srt.beam_search.joint_select import joint_select
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestBeamSelection(unittest.TestCase):
    def test_stop_compaction_matches_tensor_path(self):
        for width in (2, 7, 32):
            size = 2 * width
            # Distinct scores give the CPU and CUDA topk the same input order.
            scores = -torch.arange(width * size, dtype=torch.float32).reshape(
                width, size
            )
            cumulative = torch.zeros(width)
            ids = torch.arange(width * size).reshape_as(scores) % 5
            for stops in ([7], [0, 2], [0, 1, 2, 3]):
                for tokens in (ids, torch.full_like(ids, stops[0])):
                    args = (cumulative, scores, tokens, torch.tensor(stops))
                    expected = joint_select(*args, width)
                    actual = joint_select(*(arg.cuda() for arg in args), width)
                    for field in expected.__struct_fields__:
                        self.assertTrue(
                            torch.equal(
                                getattr(actual, field).cpu(), getattr(expected, field)
                            ),
                            field,
                        )


if __name__ == "__main__":
    unittest.main()
