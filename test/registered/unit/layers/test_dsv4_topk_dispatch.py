import unittest
from unittest.mock import Mock, patch

import torch

from sglang.jit_kernel.dsv4.topk import topk_transform_512_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4TopKDispatch(unittest.TestCase):
    def test_cluster_control_is_forwarded_to_jit_kernel(self):
        module = Mock()
        scores = torch.empty((1, 4), dtype=torch.float32)
        seq_lens = torch.tensor([4], dtype=torch.int32)
        page_table = torch.empty((1, 1), dtype=torch.int32)
        out = torch.empty((1, 1), dtype=torch.int32)
        metadata = torch.empty((2, 2), dtype=torch.int32)

        with patch(
            "sglang.jit_kernel.dsv4.topk._jit_topk_v2_module",
            return_value=module,
        ):
            topk_transform_512_v2(
                scores,
                seq_lens,
                page_table,
                out,
                256,
                metadata,
                enable_cluster=False,
            )

        module.topk_transform.assert_called_once_with(
            scores,
            seq_lens,
            page_table,
            out,
            256,
            metadata,
            None,
            False,
        )


if __name__ == "__main__":
    unittest.main()
