"""Pin the Marlin M-block sizes Muse Spark's NVFP4 experts pick per batch size."""

import unittest

from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import _marlin_block_size_m
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")

# Muse Spark routes each token to 8 of 256 experts; EP8 leaves 32 on each rank.
TOPK = 8
NUM_LOCAL_EXPERTS = 32


def _stock_block_size_m(num_tokens):
    """The uncapped rule: the first candidate the mean routes per expert underfill."""
    for block_size_m in (8, 16, 32, 48, 64):
        if num_tokens * TOPK / NUM_LOCAL_EXPERTS / block_size_m < 0.9:
            return block_size_m
    return 64


class TestMuseSparkMarlinBlockSize(CustomTestCase):
    def block_size_m(self, num_tokens):
        return _marlin_block_size_m(
            num_tokens=num_tokens, topk=TOPK, num_local_experts=NUM_LOCAL_EXPERTS
        )

    def test_pinned_sizes(self):
        expected = {8: 8, 32: 16, 128: 16, 256: 16, 753: 64, 16384: 64}
        for num_tokens, block_size_m in expected.items():
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(self.block_size_m(num_tokens), block_size_m)
        # The batches the cap changes.
        self.assertEqual(_stock_block_size_m(128), 48)
        self.assertEqual(_stock_block_size_m(256), 64)

    def test_small_batches_cap_the_stock_rule_at_16_rows(self):
        # Up to 64 routes per local expert: 256 tokens here.
        for num_tokens in range(257):
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(
                    self.block_size_m(num_tokens),
                    min(_stock_block_size_m(num_tokens), 16),
                )

    def test_larger_batches_keep_the_stock_rule(self):
        for num_tokens in range(257, 4097):
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(
                    self.block_size_m(num_tokens), _stock_block_size_m(num_tokens)
                )


if __name__ == "__main__":
    unittest.main()
