import unittest

from sglang.srt.models.deepseek_v4 import _tbo_collective_sizes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4TboCollectiveSizes(unittest.TestCase):
    def test_attention_tp_shards_preserve_dp_layout(self):
        dp_sizes, tp_sizes = _tbo_collective_sizes(
            [8, 8, 8, 8, 12, 12, 12, 12], attn_tp_size=4
        )

        self.assertEqual(dp_sizes, [8, 12])
        self.assertEqual(tp_sizes, [2, 2, 2, 2, 3, 3, 3, 3])
        self.assertEqual(sum(dp_sizes), sum(tp_sizes))

    def test_rejects_inconsistent_attention_tp_replicas(self):
        with self.assertRaisesRegex(ValueError, "differ within attention TP"):
            _tbo_collective_sizes([8, 8, 4, 8], attn_tp_size=4)

    def test_rejects_unshardable_token_count(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            _tbo_collective_sizes([6, 6, 6, 6], attn_tp_size=4)


if __name__ == "__main__":
    unittest.main()
