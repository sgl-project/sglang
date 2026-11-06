# SPDX-License-Identifier: Apache-2.0
"""Layout, values and the length gate of the SDPA operand helper.

``scaled_dot_product_attention`` takes ``[B, H, S, D]``. The heads-first view of
a ``[B, S, H, D]`` activation is strided, and AOTriton's flash path only applies
to dense operands, so on HIP the backend materializes the view - but only for
sequences long enough that the kernel gain outweighs the copy.
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import sdpa as sdpa_backend
from sglang.multimodal_gen.runtime.platforms import current_platform

_SHAPE = (2, 5, 3, 8)  # [B, S, H, D]
_LONG_SEQ = sdpa_backend._DENSE_SDPA_MIN_SEQ
_LONG_SHAPE = (1, _LONG_SEQ, 2, 8)


def _activations(shape=_SHAPE, dtype=torch.bfloat16):
    torch.manual_seed(0)
    return tuple(torch.randn(shape, dtype=dtype) for _ in range(3))


class TestSDPAOperandLayout(unittest.TestCase):
    def test_layout_is_heads_first(self):
        query, key, value = _activations()
        out = sdpa_backend.sdpa_operands(query, key, value)
        for src, got in zip((query, key, value), out):
            self.assertEqual(got.shape, (2, 3, 5, 8))
            self.assertTrue(torch.equal(got, src.transpose(1, 2)))

    def test_short_sequences_keep_the_view(self):
        query, key, value = _activations()
        got = sdpa_backend.sdpa_operands(query, key, value)
        # Below the threshold no memory is spent, whatever the platform.
        for out, src in zip(got, (query, key, value)):
            self.assertEqual(
                out.untyped_storage().data_ptr(), src.untyped_storage().data_ptr()
            )

    def test_long_sequences_are_dense_on_hip(self):
        query, key, value = _activations(_LONG_SHAPE)
        got = sdpa_backend.sdpa_operands(query, key, value)
        materialized = sdpa_backend._DENSE_SDPA_OPERANDS
        for out, src in zip(got, (query, key, value)):
            self.assertEqual(out.is_contiguous(), materialized)
            # The copy is layout-only: values stay bit-identical either way.
            self.assertTrue(torch.equal(out, src.transpose(1, 2)))

    def test_guard_matches_the_platform(self):
        self.assertEqual(sdpa_backend._DENSE_SDPA_OPERANDS, current_platform.is_hip())


if __name__ == "__main__":
    unittest.main()
