"""An unquantized embedding or lm_head refuses a float8 checkpoint weight: the copy
would keep the raw float8 codes as values and the weight's scale has nowhere to go."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
    VocabParallelEmbedding,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB, HIDDEN = 128, 64


class TestUnquantizedEmbeddingFloat8Weight(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology("test", tp_size=1, pp_size=1))

    def _layers(self):
        return (
            (
                "model.embed_tokens",
                VocabParallelEmbedding(
                    VOCAB,
                    HIDDEN,
                    params_dtype=torch.bfloat16,
                    prefix="model.embed_tokens",
                ),
            ),
            (
                "lm_head",
                ParallelLMHead(
                    VOCAB, HIDDEN, params_dtype=torch.bfloat16, prefix="lm_head"
                ),
            ),
        )

    def test_float8_weight_is_refused(self):
        codes = (torch.randn(VOCAB, HIDDEN) * 100).to(torch.float8_e4m3fn)
        for name, layer in self._layers():
            with self.subTest(layer=name):
                with self.assertRaisesRegex(ValueError, name):
                    layer.weight.weight_loader(layer.weight, codes)

    def test_unquantized_weight_still_loads(self):
        rows = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
        for name, layer in self._layers():
            with self.subTest(layer=name):
                layer.weight.weight_loader(layer.weight, rows)
                torch.testing.assert_close(layer.weight.data[:VOCAB], rows)


if __name__ == "__main__":
    unittest.main()
