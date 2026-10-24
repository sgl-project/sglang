# SPDX-License-Identifier: Apache-2.0
"""Regression test: KitchenW4A4Config must route a quantized embedding table
to ComfyInt8EmbeddingMethod instead of dropping it.

``get_quant_method`` used to return ``None`` for any non-``LinearBase``
layer, silently leaving a quantized embedding table unquantized (and
missing its packed weight loader), unlike the sibling W4A8/NVFP4 configs.
"""

import unittest
from unittest import mock

from sglang.multimodal_gen.runtime.layers.quantization.comfy_int8 import (
    ComfyInt8EmbeddingMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a4_config import (
    KitchenW4A4Config,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)


class TestKitchenW4A4EmbeddingRouting(unittest.TestCase):
    def test_quantized_embedding_marker_routes_to_comfy_int8(self):
        config = KitchenW4A4Config(
            layer_markers={
                "embed_tokens": {
                    "format": "int8_tensorwise",
                    "_is_rowwise": True,
                }
            }
        )
        embedding = mock.Mock(spec=VocabParallelEmbedding)

        method = config.get_quant_method(embedding, "embed_tokens")

        self.assertIsInstance(method, ComfyInt8EmbeddingMethod)
        self.assertTrue(config.quantizes_embedding("embed_tokens"))

    def test_unmarked_embedding_is_left_alone(self):
        config = KitchenW4A4Config(layer_markers={})
        embedding = mock.Mock(spec=VocabParallelEmbedding)

        method = config.get_quant_method(embedding, "embed_tokens")

        self.assertIsNone(method)


if __name__ == "__main__":
    unittest.main()
