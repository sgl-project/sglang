"""CPU unit tests for the experimental --sparse-attention-config parsing.

These do not require a GPU: they only exercise config parsing and algorithm
construction (no representation pools are allocated).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.factory import (
    _create_sparse_algorithm,
    parse_sparse_attention_config,
)


def _server_args(cfg=None, backend="fa3", page_size=16):
    return SimpleNamespace(
        sparse_attention_config=cfg,
        attention_backend=backend,
        page_size=page_size,
    )


class TestSparseAttentionConfig(unittest.TestCase):
    def test_parse_defaults(self):
        config = parse_sparse_attention_config(_server_args(None))
        self.assertEqual(config.algorithm, "quest")
        self.assertEqual(config.backend, "fa3")
        self.assertEqual(config.page_size, 16)
        # min_sparse_prompt_len must be an int (None would crash _compute_sparse_mask)
        self.assertEqual(config.min_sparse_prompt_len, 0)
        self.assertIsInstance(config.min_sparse_prompt_len, int)

    def test_flat_keys_go_to_extra_config(self):
        config = parse_sparse_attention_config(
            _server_args(
                '{"algorithm":"quest","sparsity_ratio":0.5,'
                '"num_recent_pages":4,"min_sparse_prompt_len":256}'
            )
        )
        self.assertEqual(config.min_sparse_prompt_len, 256)
        self.assertEqual(config.sparse_extra_config["sparsity_ratio"], 0.5)
        self.assertEqual(config.sparse_extra_config["num_recent_pages"], 4)

    def test_nested_algorithm_config(self):
        config = parse_sparse_attention_config(
            _server_args(
                '{"algorithm":"quest","algorithm_config":{"sparsity_ratio":0.3}}'
            )
        )
        self.assertEqual(config.sparse_extra_config["sparsity_ratio"], 0.3)

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            parse_sparse_attention_config(_server_args("{not json"))

    def test_factory_builds_quest(self):
        config = parse_sparse_attention_config(_server_args('{"algorithm":"quest"}'))
        algo = _create_sparse_algorithm(config, torch.device("cpu"))
        self.assertIsInstance(algo, QuestAlgorithm)

    def test_unknown_algorithm_rejected_at_parse(self):
        with self.assertRaises(ValueError):
            parse_sparse_attention_config(
                _server_args('{"algorithm":"does_not_exist"}')
            )

    def test_unsupported_backend_rejected(self):
        with self.assertRaises(ValueError):
            parse_sparse_attention_config(
                _server_args('{"algorithm":"quest","backend":"triton"}')
            )

    def test_bad_sparsity_ratio_rejected(self):
        for bad in ("1.0", "1.5", "-0.1"):
            with self.assertRaises(ValueError):
                parse_sparse_attention_config(
                    _server_args('{"algorithm":"quest","sparsity_ratio":%s}' % bad)
                )

    def test_negative_min_sparse_prompt_len_rejected(self):
        with self.assertRaises(ValueError):
            parse_sparse_attention_config(
                _server_args('{"algorithm":"quest","min_sparse_prompt_len":-1}')
            )


if __name__ == "__main__":
    unittest.main()
