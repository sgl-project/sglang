"""Unit tests for utils.py — CPU-only tests"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import sys
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.utils import (
    convert_to_bigram_key,
    maybe_init_custom_mem_pool,
)
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# convert_to_bigram_key
# ---------------------------------------------------------------------------


class TestConvertToBigramKey(CustomTestCase):
    def test_empty_list(self):
        self.assertEqual(convert_to_bigram_key([]), [])

    def test_single_token(self):
        self.assertEqual(convert_to_bigram_key([42]), [])

    def test_two_tokens(self):
        self.assertEqual(convert_to_bigram_key([1, 2]), [(1, 2)])

    def test_multiple_tokens(self):
        self.assertEqual(
            convert_to_bigram_key([1, 2, 3, 4]),
            [(1, 2), (2, 3), (3, 4)],
        )

    def test_output_length(self):
        tokens = list(range(10))
        self.assertEqual(len(convert_to_bigram_key(tokens)), 9)

    def test_consecutive_pairs_overlap(self):
        out = convert_to_bigram_key([10, 20, 30, 40])
        for i in range(len(out) - 1):
            self.assertEqual(out[i][1], out[i + 1][0])

    def test_duplicate_tokens(self):
        self.assertEqual(convert_to_bigram_key([5, 5, 5]), [(5, 5), (5, 5)])

    def test_already_tuples_passthrough(self):
        tokens = [(1, 2), (2, 3)]
        self.assertIs(convert_to_bigram_key(tokens), tokens)

    def test_large_token_ids(self):
        out = convert_to_bigram_key([999999, 0, 123456])
        self.assertEqual(out, [(999999, 0), (0, 123456)])


# ---------------------------------------------------------------------------
# maybe_init_custom_mem_pool
# ---------------------------------------------------------------------------


class TestMaybeInitCustomMemPool(CustomTestCase):
    def test_disabled_when_env_not_set(self):
        with patch(
            "sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL"
        ) as m:
            m.get.return_value = None
            enabled, pool, pool_type = maybe_init_custom_mem_pool("cpu")
        self.assertFalse(enabled)
        self.assertIsNone(pool)
        self.assertIsNone(pool_type)

    def test_returns_three_tuple(self):
        with patch(
            "sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL"
        ) as m:
            m.get.return_value = None
            result = maybe_init_custom_mem_pool("cpu")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_enabled_when_env_set(self):
        mock_mooncake = MagicMock()
        mock_mooncake.init_mooncake_custom_mem_pool.return_value = (
            True,
            object(),
            "mooncake",
        )
        with patch(
            "sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL"
        ) as m, patch.dict(
            sys.modules,
            {"sglang.srt.disaggregation.mooncake.utils": mock_mooncake},
        ):
            m.get.return_value = "/some/path"
            from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

            enabled, _, _ = maybe_init_custom_mem_pool("cuda")
        self.assertTrue(enabled)


if __name__ == "__main__":
    unittest.main()
