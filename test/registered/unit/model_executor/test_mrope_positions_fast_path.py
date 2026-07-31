"""Text-only mrope fast path == the per-row CPU list path.

The fast path replaces the per-row ``torch.tensor`` build + pageable H2D copy
with ``positions.unsqueeze(0).expand(3, -1)`` on device. Its correctness rests
on the derived identity that text-only mrope rows equal the regular positions
(extend: ``arange(prefix, prefix + extend)``; decode: ``seq_len - 1``), so this
pins the fast output against the original slow path for both modes.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"


def _run(mode, positions, seq_lens_cpu, extend_lens, prefix_lens, mm_inputs):
    self_ns = SimpleNamespace(
        forward_mode=mode,
        positions=positions,
        seq_lens_cpu=seq_lens_cpu,
        mrope_positions=None,
    )
    batch_ns = SimpleNamespace(
        multimodal_inputs=mm_inputs,
        extend_lens=extend_lens,
        prefix_lens=prefix_lens,
    )
    runner_ns = SimpleNamespace(device=DEVICE)
    with get_context().override_server_args():
        ForwardBatch._compute_mrope_positions(self_ns, runner_ns, batch_ns)
    return self_ns.mrope_positions


class TestMropePositionsFastPath(CustomTestCase):
    def test_extend_matches_slow_path(self):
        prefix_lens = [0, 100, 3]
        extend_lens = [5, 2048, 1]
        seq_lens_cpu = torch.tensor(
            [p + e for p, e in zip(prefix_lens, extend_lens)], dtype=torch.int64
        )
        positions = torch.cat(
            [
                torch.arange(p, p + e, device=DEVICE)
                for p, e in zip(prefix_lens, extend_lens)
            ]
        )
        mm = [None] * len(prefix_lens)
        fast = _run(
            ForwardMode.EXTEND, positions, seq_lens_cpu, extend_lens, prefix_lens, mm
        )
        slow = _run(
            ForwardMode.EXTEND, None, seq_lens_cpu, extend_lens, prefix_lens, mm
        )
        self.assertTrue(fast.is_contiguous())
        self.assertEqual(fast.dtype, torch.int64)
        self.assertTrue(torch.equal(fast.cpu(), slow.cpu()))

    def test_decode_matches_slow_path(self):
        seq_lens_cpu = torch.tensor([1, 17, 4096], dtype=torch.int64)
        positions = (seq_lens_cpu - 1).to(DEVICE)
        mm = [None] * 3
        fast = _run(ForwardMode.DECODE, positions, seq_lens_cpu, None, None, mm)
        slow = _run(ForwardMode.DECODE, None, seq_lens_cpu, None, None, mm)
        self.assertTrue(torch.equal(fast.cpu(), slow.cpu()))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
