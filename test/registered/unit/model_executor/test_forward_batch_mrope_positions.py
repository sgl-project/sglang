"""M-RoPE decode positions: the delta upload does not synchronize the host."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import is_pin_memory_available
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestMropeDecodePositions(CustomTestCase):
    @patch("sglang.srt.model_executor.forward_batch_info.get_exec")
    def test_delta_upload_does_not_synchronize(self, get_exec):
        if not is_pin_memory_available("cuda"):
            self.skipTest("no pinned host memory; the upload is pageable by design")
        # A detector blind to pageable H2D copies would pass with or without the fix.
        torch.cuda.set_sync_debug_mode("error")
        try:
            torch.tensor([1], device="cuda")
        except RuntimeError:
            pass
        else:
            self.skipTest("sync debug mode does not flag a pageable H2D copy here")
        finally:
            torch.cuda.set_sync_debug_mode("default")
        get_exec.return_value.deterministic.rl_on_policy_target = None
        # Image requests with negative, zero and positive deltas, and a text-only one.
        mm_inputs = [
            None
            if d is None
            else SimpleNamespace(
                mrope_position_delta=torch.tensor([[d]]), mrope_positions=None
            )
            for d in (-5, None, 0, 7)
        ]
        seq_lens = torch.tensor([100, 107, 114, 121])
        batch = SimpleNamespace(seq_lens=seq_lens.cuda(), seq_lens_cpu=seq_lens)
        decode = partial(
            ForwardBatch._compute_mrope_positions_decode,
            batch,
            SimpleNamespace(device="cuda"),
            SimpleNamespace(multimodal_inputs=mm_inputs),
        )

        decode()  # warm-up: a first-time pinned block can synchronize on its own
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            decode()
        finally:
            torch.cuda.set_sync_debug_mode("default")

        expected = torch.tensor([[94, 106, 113, 127]] * 3)
        torch.testing.assert_close(
            batch.mrope_positions.cpu(), expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
