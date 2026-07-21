"""Unit test for the multimodal embedding hidden-size guard.

Bug: _adjust_embedding_length reconciles only the row count of a multimodal
embedding, not its hidden dimension. A user-supplied precomputed_embedding with
a wrong hidden size then reached masked_scatter_ and either crashed the
scheduler (src smaller than the masked positions) or was silently misaligned
row-major (src larger), producing wrong embeddings with no error.
"""

import unittest

import torch

from sglang.srt.managers.mm_utils import _check_mm_embedding_hidden_size
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMMEmbeddingHiddenCheck(CustomTestCase):
    def test_smaller_hidden_raises(self):
        # src smaller -> masked_scatter_ would raise a cryptic "number of
        # elements of source < number of ones in mask".
        with self.assertRaises(ValueError):
            _check_mm_embedding_hidden_size(torch.zeros(5, 4), torch.zeros(2, 3))

    def test_larger_hidden_raises(self):
        # src larger -> masked_scatter_ would silently misalign (no error).
        with self.assertRaises(ValueError):
            _check_mm_embedding_hidden_size(torch.zeros(5, 4), torch.zeros(2, 6))

    def test_matching_hidden_passes(self):
        _check_mm_embedding_hidden_size(torch.zeros(5, 4), torch.zeros(2, 4))


if __name__ == "__main__":
    unittest.main()
