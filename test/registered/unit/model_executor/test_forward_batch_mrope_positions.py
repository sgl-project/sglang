"""M-RoPE decode positions in ForwardBatch: values and the delta upload."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # noqa: E402
from sglang.srt.utils.common import is_pin_memory_available  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

# A mixed batch: image requests with negative, zero and positive deltas, and a
# text-only request (None).
SEQ_LENS = [100, 107, 114, 121]
DELTAS = [-5, None, 0, 7]
EXPECTED = torch.tensor([[94, 106, 113, 127]] * 3, dtype=torch.int64)


def _decode(seq_lens, seq_lens_cpu, device):
    # Both copies of seq_lens are built by the caller, so that no copy that would
    # wait for queued GPU work runs inside the call under test.
    forward_batch = SimpleNamespace(seq_lens=seq_lens, seq_lens_cpu=seq_lens_cpu)
    mm_inputs = [
        None
        if d is None
        else SimpleNamespace(
            mrope_position_delta=torch.tensor([[d]]), mrope_positions=None
        )
        for d in DELTAS
    ]
    execution = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None))
    with patch(
        "sglang.srt.model_executor.forward_batch_info.get_exec",
        return_value=execution,
    ):
        ForwardBatch._compute_mrope_positions_decode(
            forward_batch,
            SimpleNamespace(device=device),
            SimpleNamespace(multimodal_inputs=mm_inputs),
        )
    return forward_batch.mrope_positions


class TestMropeDecodePositions(CustomTestCase):
    def test_mixed_batch_positions(self):
        seq_lens = torch.tensor(SEQ_LENS)
        actual = _decode(seq_lens, seq_lens, "cpu")
        torch.testing.assert_close(actual, EXPECTED, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestMropeDeltaUploadDoesNotWait(CustomTestCase):
    def test_upload_does_not_wait_for_queued_work(self):
        device = torch.device("cuda")
        if not is_pin_memory_available(device):
            self.skipTest("no pinned host memory; the upload is pageable by design")
        seq_lens_cpu = torch.tensor(SEQ_LENS)
        seq_lens = seq_lens_cpu.to(device)
        # First-time pinned-host and device allocations stay outside the timed call.
        _decode(seq_lens, seq_lens_cpu, device)
        torch.cuda.synchronize()

        # Queue a long kernel; a host that waits for the stream finds it finished.
        torch.cuda._sleep(1_000_000_000)
        queued = torch.cuda.Event()
        queued.record()
        actual = _decode(seq_lens, seq_lens_cpu, device)
        pending = not queued.query()
        torch.cuda.synchronize()

        self.assertTrue(pending, "M-RoPE delta upload waited for queued GPU work")
        torch.testing.assert_close(actual.cpu(), EXPECTED, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
