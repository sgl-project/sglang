"""Multi-CTA verify tree sampling: the shapes it takes from the stock tree sampler (CPU)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative import multi_cta_verify_sampling as mcta
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _probs(bs=1, draft=4, vocab=248320, dtype=torch.float32, is_cuda=True):
    return SimpleNamespace(
        is_cuda=is_cuda,
        dtype=dtype,
        shape=(bs, draft, vocab),
        dim=lambda: 3,
        device=SimpleNamespace(index=0),
    )


class TestApplies(CustomTestCase):
    def _applies(self, probs, arch_checked=True, capturing=False):
        with (
            mock.patch.object(mcta, "_arch_checked", return_value=arch_checked),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capturing
            ),
        ):
            return mcta.applies(probs)

    def test_only_the_shapes_that_gain(self):
        # up to the scratch's 1024 rows of fp32 rows of 128000+ entries on CUDA
        self.assertTrue(self._applies(_probs()))
        self.assertTrue(self._applies(_probs(bs=256, vocab=128000)))
        for rejected in (
            _probs(bs=0),
            _probs(bs=257),
            _probs(vocab=127999),
            _probs(dtype=torch.bfloat16),
            _probs(is_cuda=False),
        ):
            self.assertFalse(self._applies(rejected), rejected)

    def test_vocabs_past_the_chunk_scratch_take_the_stock_kernel(self):
        # chunks of 1024 * gcd(4, vocab) entries, at most 512
        self.assertTrue(self._applies(_probs(vocab=1 << 21)))
        for vocab in (600001, (1 << 21) + 2, (1 << 21) + 4):
            self.assertGreater(mcta._num_chunks(vocab), mcta._MAX_CHUNKS)
            self.assertFalse(self._applies(_probs(vocab=vocab)), vocab)

    def test_only_on_the_checked_architecture(self):
        self.assertFalse(self._applies(_probs(), arch_checked=False))

    def test_never_first_built_inside_a_capture(self):
        with mock.patch.dict(mcta._scratch, clear=True):
            self.assertFalse(self._applies(_probs(), capturing=True))
            mcta._scratch[0] = {}
            self.assertTrue(self._applies(_probs(), capturing=True))

    def test_scratch_bounds_raise(self):
        probs = torch.zeros(1025, 1, 8)
        with self.assertRaisesRegex(ValueError, "past the scratch"):
            mcta.tree_speculative_sampling_target_only(*([None] * 9), probs, probs)


if __name__ == "__main__":
    unittest.main()
