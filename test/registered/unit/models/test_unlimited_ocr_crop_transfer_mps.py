"""Crop transfer on MPS: no pinned buffers, which torch 2.13 returns as broken mps tensors."""

import time
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.models.unlimited_ocr_v2_support.crop_transfer import (
    UnlimitedOcrCropTransfer,
)
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=10, suite="stage-a-unit-test-mps")


@unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
class TestUocrCropTransferMps(CustomTestCase):
    def test_prefetch_stages_host_crops_beside_mps_work(self):
        # torch 2.13 hands out pinned "cpu" memory as an mps tensor, so a staging
        # thread copying into it encoded MPS work beside this thread and aborted.
        with envs.SGLANG_ENABLE_UOCR_CROP_PREFETCH.override(True):
            transfer = UnlimitedOcrCropTransfer(model_dtype=torch.bfloat16)
        self.addCleanup(transfer.close)
        x = torch.randn(1024, 1024, device="mps")
        deadline = time.monotonic() + 3
        rounds = 0
        while time.monotonic() < deadline:
            crops = [torch.full((3, 64, 64), float(rounds * 7 + i)) for i in range(6)]
            transfer.prefetch_sources(crops, torch.bfloat16)
            for _ in range(4):
                (x @ x).sum().item()
            staged = [transfer.entries[(id(c), torch.bfloat16)] for c in crops]
            self.assertTrue(all(e.future.result().is_cpu for e in staged))
            out = transfer.stack(crops, torch.bfloat16, torch.device("mps"))
            expected = torch.stack(crops).to(torch.bfloat16)
            self.assertTrue(torch.equal(out.cpu(), expected))
            rounds += 1
        self.assertGreater(rounds, 0)

    def test_stack_without_prefetch_returns_usable_mps_tensor(self):
        with envs.SGLANG_ENABLE_UOCR_CROP_PREFETCH.override(False):
            transfer = UnlimitedOcrCropTransfer(model_dtype=torch.bfloat16)
        self.addCleanup(transfer.close)
        crops = [torch.randn(1, 6, 3, 64, 64) for _ in range(2)]
        out = transfer.stack(crops, torch.bfloat16, torch.device("mps"))
        self.assertTrue(out.is_mps)
        expected = torch.stack(crops).to(torch.bfloat16)
        self.assertTrue(
            torch.equal(torch.cat([out[0], out[1]]).cpu(), expected.flatten(0, 1))
        )


if __name__ == "__main__":
    unittest.main()
