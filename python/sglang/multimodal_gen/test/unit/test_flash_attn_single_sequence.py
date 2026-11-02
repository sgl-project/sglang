import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import flash_attn
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
)

HEADS, HEAD_DIM = 4, 8


class TestFlashAttnSingleSequence(unittest.TestCase):
    def _run(self, bounds, *, padding=True, max_seqlen=None):
        total = bounds[-1]
        q, k, v = (torch.randn(total, HEADS, HEAD_DIM) for _ in range(3))
        impl = SimpleNamespace(
            _request_skip_softmax_threshold=lambda: (False, None),
            softmax_scale=HEAD_DIM**-0.5,
            causal=False,
            packed_trailing_padding=padding,
        )
        calls = []

        def fake(q, k, v, *, cu_seqlens_q, out=None, **kwargs):
            calls.append((tuple(q.shape), cu_seqlens_q, kwargs))
            if out is None:
                return torch.ones_like(q)
            out.fill_(1)

        with (
            patch.object(flash_attn, "fa_ver", 3),
            patch.object(flash_attn, "flash_attn_varlen_func", fake),
        ):
            output = FlashAttentionImpl.forward_varlen(
                impl,
                q,
                k,
                v,
                cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
                max_seqlen=max_seqlen or bounds[1],
                cu_seqlens_host=bounds,
            )
        return output, calls

    def test_one_sequence_runs_the_batch_kernel(self):
        output, [(shape, cu, kwargs)] = self._run((0, 12))
        self.assertEqual(shape, (1, 12, HEADS, HEAD_DIM))
        self.assertIsNone(cu)
        self.assertEqual(kwargs["max_seqlen_q"], 12)
        self.assertTrue((output == 1).all())

    def test_trailing_padding_runs_live_rows_and_zeroes_the_rest(self):
        output, [(shape, cu, kwargs)] = self._run((0, 10, 16))
        self.assertEqual(shape, (1, 10, HEADS, HEAD_DIM))
        self.assertIsNone(cu)
        self.assertTrue((output[:10] == 1).all())
        self.assertTrue((output[10:] == 0).all())

    def test_several_sequences_keep_the_varlen_kernel(self):
        _, [(shape, cu, _)] = self._run((0, 6, 12), padding=False, max_seqlen=6)
        self.assertEqual(shape, (12, HEADS, HEAD_DIM))
        self.assertEqual(cu.tolist(), [0, 6, 12])

    def test_padding_is_only_dropped_where_the_layout_declares_it(self):
        _, [(shape, cu, _)] = self._run((0, 10, 16), padding=False)
        self.assertEqual(shape, (16, HEADS, HEAD_DIM))
        self.assertEqual(cu.tolist(), [0, 10, 16])


if __name__ == "__main__":
    unittest.main()
