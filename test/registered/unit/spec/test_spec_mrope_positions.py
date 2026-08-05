"""compute_spec_mrope_positions: expanded-view fast path == the repeat form.

Text-only spec batches used to pay zeros + add + repeat kernels per
init_new (~3x per decode step) although every mrope delta is zero. The fast
path publishes a stride-0 expanded view of the positions instead. Pins:
(1) numeric equality with the old ``(pos + delta).repeat(3, 1)`` form for
text-only and mixed batches, (2) the no-copy property (the view aliases the
positions storage), (3) a consumer-style ``copy_`` into a contiguous static
buffer, which is how every CUDA-graph runner ingests it.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"


def _call(bs, draft, mm_inputs, positions):
    fb = ForwardBatch.__new__(ForwardBatch)
    fb.seq_lens = torch.zeros(bs, dtype=torch.int64, device=DEVICE)
    fb.spec_info = SimpleNamespace(positions=positions)
    batch = SimpleNamespace(multimodal_inputs=mm_inputs, spec_info=fb.spec_info)
    runner = SimpleNamespace(device=DEVICE)
    ForwardBatch.compute_spec_mrope_positions(fb, runner, batch)
    return fb.mrope_positions


def _reference(bs, positions, deltas):
    pos = positions.view(bs, -1)
    return (pos + deltas).flatten().unsqueeze(0).repeat(3, 1)


class TestSpecMropePositions(CustomTestCase):
    def test_text_only_matches_and_aliases(self):
        bs, draft = 3, 4
        positions = torch.arange(bs * draft, device=DEVICE, dtype=torch.int64)
        got = _call(bs, draft, [None] * bs, positions)
        ref = _reference(
            bs, positions, torch.zeros((bs, 1), dtype=torch.int64, device=DEVICE)
        )
        self.assertTrue(torch.equal(got, ref))
        self.assertEqual(got.shape, (3, bs * draft))
        self.assertEqual(got.data_ptr(), positions.data_ptr())

    def test_mixed_batch_matches(self):
        bs, draft = 2, 4
        positions = torch.arange(bs * draft, device=DEVICE, dtype=torch.int64)
        delta = torch.tensor([[7]], dtype=torch.int64)
        mm = [None, SimpleNamespace(mrope_position_delta=delta)]
        got = _call(bs, draft, mm, positions)
        deltas = torch.tensor([[0], [7]], dtype=torch.int64, device=DEVICE)
        ref = _reference(bs, positions, deltas)
        self.assertTrue(torch.equal(got, ref))

    def test_graph_buffer_copy_ingests_view(self):
        bs, draft = 2, 4
        positions = torch.arange(bs * draft, device=DEVICE, dtype=torch.int64)
        got = _call(bs, draft, [None] * bs, positions)
        buf = torch.zeros((3, 32), dtype=torch.int64, device=DEVICE)
        buf[:, : bs * draft].copy_(got)
        self.assertTrue(torch.equal(buf[:, : bs * draft], got))
        positions.add_(100)
        self.assertFalse(torch.equal(buf[:, : bs * draft], got))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
