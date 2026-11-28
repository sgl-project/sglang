"""Unit tests for compressed-tensors quantized-embedding handling — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.test.test_utils import CustomTestCase

# What llm-compressor writes for a 4-bit group-quantized embedding table, as
# found in checkpoints that quantize `Embedding` alongside `Linear`.
_INT4_GROUP_WEIGHTS = {
    "num_bits": 4,
    "type": "int",
    "strategy": "group",
    "group_size": 64,
    "symmetric": True,
    "dynamic": False,
}


def _config(targets, ignore=()):
    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": list(targets),
                    "weights": _INT4_GROUP_WEIGHTS,
                    "input_activations": None,
                }
            },
        }
    )


class _Embedding(VocabParallelEmbedding):
    """isinstance-compatible stand-in.

    The real ``__init__`` builds vocab shards from the live TP topology, which a
    CPU unit test has no reason to stand up; only the class identity matters for
    the dispatch under test.
    """

    def __init__(self):
        torch.nn.Module.__init__(self)


class _Head(ParallelLMHead):
    def __init__(self):
        torch.nn.Module.__init__(self)


_GET_LM_HEAD_SCHEME = (
    "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors."
    "CompressedTensorsConfig.get_lm_head_scheme"
)


class TestQuantizedEmbeddingIsRefused(CustomTestCase):
    """No scheme here implements the embedding lookup, so a checkpoint that
    quantizes the embedding table must fail loudly rather than fall back to
    UnquantizedEmbeddingMethod and serve a table nothing can fill."""

    def test_quantized_embedding_raises(self):
        config = _config(["Linear", "Embedding"])
        with self.assertRaises(NotImplementedError) as ctx:
            config.get_quant_method(_Embedding(), prefix="model.embed_tokens")
        message = str(ctx.exception)
        self.assertIn("model.embed_tokens", message)
        # The message has to carry the scheme, or the user cannot tell which
        # part of their recipe to change.
        self.assertIn("num_bits=4", message)
        self.assertIn("group_size=64", message)

    def test_embedding_targeted_by_name_raises(self):
        # A recipe may name the module instead of its type.
        config = _config(["Linear", "re:.*embed_tokens"])
        with self.assertRaises(NotImplementedError):
            config.get_quant_method(_Embedding(), prefix="model.embed_tokens")


class TestUnquantizedEmbeddingStillWorks(CustomTestCase):
    """The ordinary case must keep working: recipes target `Linear` and leave
    the embedding table alone, and find_matched_target raises for a module no
    target names — so the branch has to swallow that rather than propagate it."""

    def test_linear_only_config_returns_none(self):
        config = _config(["Linear"])
        self.assertIsNone(
            config.get_quant_method(_Embedding(), prefix="model.embed_tokens")
        )

    def test_ignored_embedding_returns_none(self):
        config = _config(["Linear", "Embedding"], ignore=["model.embed_tokens"])
        self.assertIsNone(
            config.get_quant_method(_Embedding(), prefix="model.embed_tokens")
        )

    def test_no_layer_name_returns_none(self):
        config = _config(["Linear", "Embedding"])
        self.assertIsNone(config.get_quant_method(_Embedding(), prefix=None))


class TestLmHeadIsUnaffected(CustomTestCase):
    """ParallelLMHead subclasses VocabParallelEmbedding, so it must be answered
    by the head path and never reach the embedding branch."""

    def test_head_uses_lm_head_scheme(self):
        config = _config(["Linear", "Embedding"])
        with patch(_GET_LM_HEAD_SCHEME, return_value=None) as mock_resolve:
            self.assertIsNone(config.get_quant_method(_Head(), prefix="lm_head"))
        mock_resolve.assert_called_once()


class TestGetEmbeddingWeightQuant(CustomTestCase):
    def test_returns_the_declared_args(self):
        config = _config(["Linear", "Embedding"])
        weight_quant = config.get_embedding_weight_quant(
            _Embedding(), layer_name="model.embed_tokens"
        )
        self.assertIsNotNone(weight_quant)
        self.assertEqual(weight_quant.num_bits, 4)
        self.assertEqual(weight_quant.group_size, 64)

    def test_returns_none_when_untargeted(self):
        config = _config(["Linear"])
        self.assertIsNone(
            config.get_embedding_weight_quant(
                _Embedding(), layer_name="model.embed_tokens"
            )
        )


if __name__ == "__main__":
    unittest.main()
