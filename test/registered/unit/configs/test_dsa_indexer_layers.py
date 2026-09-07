import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import dsa_layer_skips_topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _glm52_config(**overrides):
    """GLM-5.2's own indexer settings, from the released checkpoint's config."""
    defaults = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        num_hidden_layers=78,
        index_head_dim=128,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _indexer_layers(config, num_layers):
    return [i for i in range(num_layers) if not dsa_layer_skips_topk(config, i)]


class TestDsaIndexerLayers(CustomTestCase):
    """Which layers own an Indexer decides how much index-K cache to allocate.

    A layer that skips top-k reuses the previous layer's indices and is built
    with ``self.indexer = None`` (models/deepseek_v2.py), so it never writes
    index-K. Both the CUDA ``IndexKeyCache`` and ``NPUMLATokenToKVPool`` size
    their buffers from this predicate, and on a 1M-token GLM-5.2 pool each
    wrongly-included layer costs ~0.24 GiB per rank.
    """

    def test_glm52_indexes_21_of_78_layers(self):
        config = _glm52_config()
        layers = _indexer_layers(config, 78)

        # freq 4 with offset 3 gives max(layer_id - 2, 0) % 4 == 0.
        self.assertEqual(layers, [0, 1, 2] + list(range(6, 78, 4)))
        self.assertEqual(len(layers), 21)

    def test_the_predicate_matches_the_checkpoints_indexer_types_array(self):
        """The config ships an independent description of the same fact; if the
        two ever disagree, the pool mask is the one that will be wrong."""
        config = _glm52_config()
        indexes = set(_indexer_layers(config, 78))
        indexer_types = ["full" if i in indexes else "shared" for i in range(78)]

        self.assertEqual(indexer_types[:7], ["full"] * 3 + ["shared"] * 3 + ["full"])
        self.assertEqual(indexer_types.count("full"), 21)
        self.assertEqual(indexer_types[-3:], ["shared"] * 3)

    def test_every_layer_indexes_when_the_frequency_is_one(self):
        # The default shape, and the one that must keep allocating densely.
        config = _glm52_config(index_topk_freq=1, index_skip_topk_offset=None)
        self.assertEqual(len(_indexer_layers(config, 78)), 78)

    def test_an_explicit_pattern_overrides_the_frequency(self):
        pattern = ["F", "S", "S", "F"]
        config = _glm52_config(index_topk_pattern=pattern)
        # "S" marks a skip; layers past the pattern's end never skip.
        self.assertEqual(_indexer_layers(config, 4), [0, 3])


if __name__ == "__main__":
    unittest.main()
