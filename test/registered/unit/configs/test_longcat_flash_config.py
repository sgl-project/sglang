"""Unit tests for ``sglang.srt.configs.longcat_flash.LongcatFlashConfig``."""

import unittest

from sglang.srt.configs.longcat_flash import LongcatFlashConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLongcatFlashConfig(CustomTestCase):
    def test_ngram_embedding_accepts_released_oe_aliases(self):
        cfg = LongcatFlashConfig(
            vocab_size=1000,
            oe_vocab_size_ratio=0.25,
            oe_neighbor_num=3,
            oe_split_num=4,
        )

        self.assertTrue(cfg.use_ngram_embedding)
        self.assertEqual(cfg.ngram_embedding_m, 250)
        self.assertEqual(cfg.ngram_embedding_n, 3)
        self.assertEqual(cfg.ngram_embedding_k, 4)
        self.assertEqual(cfg.ngram_vocab_size_ratio, 0.25)
        self.assertEqual(cfg.emb_neighbor_num, 3)
        self.assertEqual(cfg.emb_split_num, 4)

    def test_canonical_ngram_fields_take_priority_over_aliases(self):
        cfg = LongcatFlashConfig(
            vocab_size=1000,
            ngram_vocab_size_ratio=0.2,
            emb_neighbor_num=2,
            emb_split_num=5,
            oe_vocab_size_ratio=0.25,
            oe_neighbor_num=3,
            oe_split_num=4,
        )

        self.assertEqual(cfg.ngram_embedding_m, 200)
        self.assertEqual(cfg.ngram_embedding_n, 2)
        self.assertEqual(cfg.ngram_embedding_k, 5)
        self.assertEqual(cfg.ngram_vocab_size_ratio, 0.2)
        self.assertEqual(cfg.emb_neighbor_num, 2)
        self.assertEqual(cfg.emb_split_num, 5)


if __name__ == "__main__":
    unittest.main()
