import unittest
from unittest.mock import patch

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_config import validate_ngram_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNgramConfig(unittest.TestCase):
    valid = dict(
        min_bfs_breadth=1,
        max_bfs_breadth=10,
        max_trie_depth=18,
        capacity=1000,
        draft_token_num=12,
        external_sam_budget=0,
        external_corpus_max_tokens=1000,
    )

    def test_valid_boundaries(self):
        validate_ngram_config(**self.valid)
        validate_ngram_config(**dict(self.valid, capacity=19, external_sam_budget=11))
        validate_ngram_config(**dict(self.valid, draft_token_num=1))

    def test_invalid_values_fail_before_ffi(self):
        invalid = [
            dict(min_bfs_breadth=0),
            dict(min_bfs_breadth=-1),
            dict(min_bfs_breadth=11),
            dict(max_bfs_breadth=2**31),
            dict(max_trie_depth=1),
            dict(max_trie_depth=2**31),
            dict(capacity=18),
            dict(capacity=-1),
            dict(draft_token_num=0),
            dict(external_sam_budget=-1),
            dict(external_sam_budget=12),
            dict(external_corpus_max_tokens=0),
            dict(external_corpus_max_tokens=-1),
        ]
        with patch(
            "sglang.srt.speculative.cpp_ngram.ngram_corpus.get_ngram_corpus_cls"
        ) as ffi:
            for values in invalid:
                with self.subTest(values=values), self.assertRaises(ValueError):
                    NgramCorpus(**dict(self.valid, **values))
            ffi.assert_not_called()


if __name__ == "__main__":
    unittest.main()
