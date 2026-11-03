"""DSpark's accept softmax must not depend on the batch under deterministic inference."""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEVICE = "cuda"
# DeepSeek-V4's vocabulary. On B200, FlashInfer gives a row different bits at 37 rows and
# above than at fewer rows.
VOCAB = 129280
ROWS_PER_REQUEST = 6


class TestDSparkSoftmaxDeterministic(CustomTestCase):
    def test_request_gets_same_bits_in_any_batch(self):
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        logits = torch.randn(
            (8 * ROWS_PER_REQUEST, VOCAB), device=DEVICE, generator=generator
        )
        temperatures = torch.full((8,), 0.7, device=DEVICE)

        def softmax(batch_size):
            return SoftmaxTemp.execute(
                logits=logits[: batch_size * ROWS_PER_REQUEST],
                temperatures=temperatures[:batch_size],
                rows_per_request=ROWS_PER_REQUEST,
            )[:ROWS_PER_REQUEST]

        with get_context().override_server_args(enable_deterministic_inference=True):
            self.assertTrue(torch.equal(softmax(8), softmax(1)))


if __name__ == "__main__":
    unittest.main()
