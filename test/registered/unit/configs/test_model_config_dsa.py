import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_index_topk_pattern,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_dsa_config(**overrides):
    defaults = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        index_topk_freq=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestDsaIndexTopkPattern(CustomTestCase):
    def test_explicit_index_topk_pattern_takes_precedence(self):
        config = _make_dsa_config(
            index_topk_pattern="FSF", indexer_types=["S", "S", "S"]
        )

        self.assertEqual(get_dsa_index_topk_pattern(config), "FSF")
        self.assertFalse(dsa_layer_skips_topk(config, 0))
        self.assertTrue(dsa_layer_skips_topk(config, 1))
        self.assertFalse(dsa_layer_skips_topk(config, 2))

    def test_derives_index_topk_pattern_from_fs_indexer_types(self):
        config = _make_dsa_config(index_topk_pattern=None, indexer_types=["F", "S"])

        self.assertEqual(get_dsa_index_topk_pattern(config), "FS")
        self.assertFalse(dsa_layer_skips_topk(config, 0))
        self.assertTrue(dsa_layer_skips_topk(config, 1))

    def test_derives_index_topk_pattern_from_word_indexer_types(self):
        config = _make_dsa_config(
            index_topk_pattern=None, indexer_types=["full", "shared", "skip_topk"]
        )

        self.assertEqual(get_dsa_index_topk_pattern(config), "FSS")
        self.assertFalse(dsa_layer_skips_topk(config, 0))
        self.assertTrue(dsa_layer_skips_topk(config, 1))
        self.assertTrue(dsa_layer_skips_topk(config, 2))

    def test_falls_back_to_frequency_when_indexer_types_are_unknown(self):
        config = _make_dsa_config(
            index_topk_freq=2,
            index_skip_topk_offset=1,
            index_topk_pattern=None,
            indexer_types=["full", "unknown"],
        )

        self.assertIsNone(get_dsa_index_topk_pattern(config))
        self.assertFalse(dsa_layer_skips_topk(config, 0))
        self.assertTrue(dsa_layer_skips_topk(config, 1))


if __name__ == "__main__":
    unittest.main()
