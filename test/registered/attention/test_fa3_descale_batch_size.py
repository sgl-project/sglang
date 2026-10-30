"""
Regression tests for the FA k/v_descale batch dimension.

FA validates ``k_descale.shape == (batch_size, num_heads_k)`` where ``batch_size``
is the one implied by the ``cu_seqlens_q`` handed to that same call, i.e.
``cu_seqlens_q.numel() - 1``. Sizing the descale from the logical
``forward_batch.batch_size`` instead is wrong whenever the two differ, and the
kernel then rejects the call with

    RuntimeError: k_descale must have shape (batch_size, num_heads_k)

taking the whole batch (and the scheduler process) down with it.

The two are not the same tensor-shape source: some paths build ``cu_seqlens_q``
from a per-request segment list (``pad(cumsum(extend_seq_lens))``) while
``batch_size`` counts batch rows, and some paths grow ``batch_size`` after the
attention metadata was already built.
"""

import unittest

import torch

from sglang.srt.layers.attention.flashattention_backend import fa_descale_batch_size
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-large-amd")


def _cu_seqlens_q_from_segments(segments, device="cpu"):
    """Build cu_seqlens_q the way the varlen extend paths do."""
    lens = torch.tensor(segments, dtype=torch.int32, device=device)
    return torch.nn.functional.pad(torch.cumsum(lens, dim=0, dtype=torch.int32), (1, 0))


class TestFaDescaleBatchSize(unittest.TestCase):
    def test_batch_size_comes_from_cu_seqlens_not_forward_batch(self):
        """cu_seqlens_q wins whenever it disagrees with forward_batch.batch_size."""
        # (segments, forward_batch.batch_size) pairs where the two disagree.
        # The batch_size values are larger here, which is the direction produced
        # when the batch is padded after the metadata was built.
        for segments, logical_batch_size in [
            ([4], 2),
            ([4, 4], 3),
            ([4, 4, 4], 4),
            ([1, 3, 2], 5),
        ]:
            with self.subTest(segments=segments, batch_size=logical_batch_size):
                cu_seqlens_q = _cu_seqlens_q_from_segments(segments)
                self.assertNotEqual(cu_seqlens_q.numel() - 1, logical_batch_size)
                self.assertEqual(
                    fa_descale_batch_size(cu_seqlens_q, logical_batch_size),
                    len(segments),
                )

    def test_descale_first_dim_matches_cu_seqlens_q(self):
        """The expanded descale is what FA will accept for that cu_seqlens_q."""
        num_kv_heads = 8
        # A scalar scale, which is what RadixAttention holds when the KV cache
        # quantization method created it.
        k_scale = torch.tensor(0.5, dtype=torch.float32)
        v_scale = torch.tensor(0.25, dtype=torch.float32)

        segments = [4, 4, 4]
        cu_seqlens_q = _cu_seqlens_q_from_segments(segments)
        logical_batch_size = 5  # padded, disagrees with cu_seqlens_q on purpose

        descale_shape = (
            fa_descale_batch_size(cu_seqlens_q, logical_batch_size),
            num_kv_heads,
        )
        k_descale = k_scale.expand(descale_shape)
        v_descale = v_scale.expand(descale_shape)

        expected_batch = cu_seqlens_q.numel() - 1
        self.assertEqual(k_descale.shape, (expected_batch, num_kv_heads))
        self.assertEqual(v_descale.shape, (expected_batch, num_kv_heads))
        self.assertNotEqual(k_descale.shape[0], logical_batch_size)

    def test_falls_back_when_cu_seqlens_q_is_absent(self):
        """Non-varlen paths keep the logical batch size."""
        self.assertEqual(fa_descale_batch_size(None, 7), 7)

    def test_agrees_when_segments_match_batch_rows(self):
        """The uniform case is unchanged: one segment per batch row."""
        for batch_size in (1, 2, 8, 40):
            with self.subTest(batch_size=batch_size):
                cu_seqlens_q = _cu_seqlens_q_from_segments([1] * batch_size)
                self.assertEqual(
                    fa_descale_batch_size(cu_seqlens_q, batch_size), batch_size
                )


if __name__ == "__main__":
    unittest.main()
