"""Pinned crop staging slots are reused across batches without corrupting bytes."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.models.unlimited_ocr_crop_transfer import UnlimitedOcrCropTransfer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class UnlimitedOcrCropTransferCudaTest(CustomTestCase):
    def test_repeated_pinned_reuse_preserves_bytes(self):
        # Prefetch off: every batch goes through the pinned staging slots.
        with envs.SGLANG_ENABLE_UOCR_CROP_PREFETCH.override(False):
            transfer = UnlimitedOcrCropTransfer(model_dtype=torch.bfloat16)
        self.addCleanup(transfer.close)
        sources = [
            [
                torch.full((256, 256), float(step * 7 + page) / 123)
                for page in range(1 + step % 7)
            ]
            for step in range(8)
        ]
        stream = torch.cuda.Stream(device=0)
        outputs = []
        with torch.cuda.stream(stream), torch.inference_mode():
            torch.cuda._sleep(30_000_000)
            for tensors in sources:
                outputs.append(
                    transfer.stack(
                        tensors=tensors,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda:0"),
                    )
                )
        stream.synchronize()
        for source, output in zip(sources, outputs):
            expected = torch.stack(source).bfloat16()
            self.assertTrue(
                torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
            )


if __name__ == "__main__":
    unittest.main()
