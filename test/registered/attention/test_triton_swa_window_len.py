"""Unit test: the Triton backend's sliding-window DECODE buffer must be
two-side inclusive.

``update_sliding_window_buffer`` is passed ``sliding_window_size =
config.sliding_window - 1`` (a radius). The canonical window for a query at
position ``p`` is ``[p - radius, p]``, i.e. ``radius + 1`` tokens -- this is what
FlashInfer (``min(seq_lens, sliding_window_size + 1)``) and FlashAttention
(two-side-inclusive ``(sliding_window_size, 0)``) produce. The Triton decode
kernel consumes the gathered window verbatim with no per-query masking, so the
buffer must already hold the full window.

Regression: the helper capped the decode window at ``radius`` instead of
``radius + 1``, silently dropping the oldest in-window key on every decode step
past the window size (e.g. ``seq_len=5000, radius=4095`` kept keys ``[905, 4999]``
and lost key ``904``). The extend / target-verify paths
(``include_current_token=False``) re-mask per query and must keep the radius-only
gather, so this test also pins that path unchanged.
"""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.triton_backend as tb
from sglang.srt.layers.attention.triton_backend import update_sliding_window_buffer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class _CPUGatherKernel:
    """CPU stand-in for the ``create_flashinfer_kv_indices_triton`` launch.

    The GPU gather kernel is not launchable on a CPU-only runner; this double
    performs the identical index copy so the helper's pure-torch window-length /
    start-index math (the code under test) runs unchanged.
    """

    def __getitem__(self, grid):
        def launch(
            req_to_token,
            req_pool_indices,
            kv_lens,
            kv_indptr,
            kv_start,
            kv_indices,
            stride,
        ):
            for b in range(req_pool_indices.shape[0]):
                s = int(kv_start[b])
                n = int(kv_lens[b])
                dst = int(kv_indptr[b])
                req = int(req_pool_indices[b])
                kv_indices[dst : dst + n] = req_to_token[req, s : s + n]

        return launch


def _run(seq_len, radius, include_current_token):
    bs = 1
    max_ctx = seq_len + 8
    # Identity req_to_token so a gathered kv loc equals its token position.
    req_to_token = torch.arange(bs * max_ctx, dtype=torch.int32).reshape(bs, max_ctx)
    req_pool_indices = torch.zeros(bs, dtype=torch.int64)
    seq_lens = torch.tensor([seq_len], dtype=torch.int64)
    window_kv_indptr = torch.zeros(bs + 1, dtype=torch.int64)
    with patch.object(tb, "create_flashinfer_kv_indices_triton", _CPUGatherKernel()):
        _, indices, lens, start = update_sliding_window_buffer(
            window_kv_indptr,
            req_to_token,
            radius,
            seq_lens,
            req_pool_indices,
            bs,
            device=torch.device("cpu"),
            include_current_token=include_current_token,
        )
    return int(lens[0]), int(start[0]), indices.tolist()


class TestTritonSwaWindowLen(CustomTestCase):
    RADIUS = 4095  # config.sliding_window = 4096

    def test_decode_window_is_two_side_inclusive(self):
        # Matches FlashInfer's min(seq_lens, sliding_window_size + 1).
        for seq_len in (4096, 4097, 5000):
            length, start, indices = _run(
                seq_len, self.RADIUS, include_current_token=True
            )
            self.assertEqual(length, min(seq_len, self.RADIUS + 1))
            self.assertEqual(start, seq_len - length)
            oldest = seq_len - 1 - self.RADIUS
            self.assertIn(oldest, indices)  # the key the old code dropped
            self.assertEqual(indices[-1], seq_len - 1)  # current token retained

    def test_decode_window_no_clip_below_window(self):
        # No change when the sequence is at/under the window size.
        for seq_len in (1, 100, 4095):
            length, start, _ = _run(seq_len, self.RADIUS, include_current_token=True)
            self.assertEqual(length, seq_len)
            self.assertEqual(start, 0)

    def test_extend_path_keeps_radius_only(self):
        # include_current_token=False (extend / target-verify) must NOT widen:
        # those kernels re-mask per query, so the radius-only gather is correct.
        length, start, _ = _run(5000, self.RADIUS, include_current_token=False)
        self.assertEqual(length, self.RADIUS)
        self.assertEqual(start, 5000 - self.RADIUS)


if __name__ == "__main__":
    unittest.main()
