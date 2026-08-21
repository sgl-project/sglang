import unittest

import numpy as np
import torch

from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    gather_dcp_tokens_to_staging,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDCPStagingGather(unittest.TestCase):
    def test_gather_is_byte_preserving_and_layer_major(self):
        device = "cuda:0"
        layer0 = torch.arange(20 * 8, dtype=torch.uint8, device=device).reshape(20, 8)
        layer1 = (
            torch.arange(20 * 5, dtype=torch.uint8, device=device).reshape(20, 5) + 37
        )
        indices = np.array([3, 11, 19], dtype=np.int64)
        indices_cuda = torch.tensor(indices, dtype=torch.int64, device=device)
        staging = StagingBuffer(3 * (8 + 5), device, gpu_id=0)

        written = gather_dcp_tokens_to_staging(
            [layer0.data_ptr(), layer1.data_ptr()],
            indices,
            [8, 5],
            staging,
            gpu_id=0,
        )

        expected = torch.cat(
            [layer0[indices_cuda].reshape(-1), layer1[indices_cuda].reshape(-1)]
        )
        self.assertEqual(written, expected.numel())
        torch.testing.assert_close(staging.buffer[:written], expected)


if __name__ == "__main__":
    unittest.main()
