"""Text-only spec mrope fast path == the delta+repeat slow path.

Text-only deltas are all zero, so the mrope rows equal the flattened
positions; the fast path returns a stride-0 ``expand(3, -1)`` view instead of
zeros + add + repeat. Pins value equality against the original formula and
that a graph-style static-buffer ``copy_`` accepts the expanded view.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"


def _run(positions, bs):
    self_ns = SimpleNamespace(
        seq_lens=torch.zeros(bs, device=DEVICE),
        mrope_positions=None,
    )
    batch_ns = SimpleNamespace(
        multimodal_inputs=[None] * bs,
        spec_info=SimpleNamespace(positions=positions),
    )
    runner_ns = SimpleNamespace(device=DEVICE)
    ForwardBatch.compute_spec_mrope_positions(self_ns, runner_ns, batch_ns)
    return self_ns.mrope_positions


class TestSpecMropeFastPath(CustomTestCase):
    def test_matches_delta_repeat_formula(self):
        for bs, width in ((1, 4), (3, 4), (5, 1)):
            positions = torch.randint(
                0, 10_000, (bs * width,), device=DEVICE, dtype=torch.int64
            )
            got = _run(positions, bs)
            ref = positions.view(bs, -1).flatten().unsqueeze(0).repeat(3, 1)
            self.assertEqual(got.shape, ref.shape)
            self.assertTrue(torch.equal(got.contiguous(), ref))

    def test_static_buffer_copy_accepts_expanded_view(self):
        positions = torch.arange(8, device=DEVICE, dtype=torch.int64)
        got = _run(positions, 2)
        buf = torch.zeros(3, 16, device=DEVICE, dtype=torch.int64)
        buf[:, :8].copy_(got)
        self.assertTrue(torch.equal(buf[:, :8], positions.unsqueeze(0).repeat(3, 1)))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
