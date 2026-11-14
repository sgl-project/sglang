"""DSpark's accept softmax must not depend on the batch under deterministic inference."""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# backend-specific: ROCm splits the vocabulary differently as the batch size changes.
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
# On B200, FlashInfer gives a row different bits at 37 rows and above at 130560
# (MiniCPM5-2B). torch.softmax gives some rows different bits at odd positions at 248094
# (MiniCPM-V-4.7-1B), so a request needs an odd row count to start at an odd position.
VOCABS = (130560, 248094)
BATCH_SIZE = 8
ROWS_PER_REQUEST = 7


class TestDSparkSoftmaxDeterministic(CustomTestCase):
    def test_request_gets_same_bits_in_any_batch(self):
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        temperatures = torch.full((BATCH_SIZE,), 0.7, device=DEVICE)

        def softmax(logits):
            return SoftmaxTemp.execute(
                logits=logits,
                temperatures=temperatures[: logits.shape[0] // ROWS_PER_REQUEST],
                rows_per_request=ROWS_PER_REQUEST,
            )

        for vocab in VOCABS:
            logits = torch.randn(
                (BATCH_SIZE * ROWS_PER_REQUEST, vocab),
                device=DEVICE,
                generator=generator,
            )
            with get_context().override_server_args(
                enable_deterministic_inference=True
            ):
                batch = softmax(logits).split(ROWS_PER_REQUEST)
                for slot, request in enumerate(logits.split(ROWS_PER_REQUEST)):
                    self.assertTrue(
                        torch.equal(batch[slot], softmax(request.clone())),
                        f"vocab {vocab}, slot {slot}",
                    )


if __name__ == "__main__":
    unittest.main()
