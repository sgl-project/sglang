import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestKimiK3Platform(unittest.TestCase):
    def test_non_cuda_skips_fused_front_weight_merge(self):
        moe = SimpleNamespace(use_latent_moe=True)

        with patch(
            "sglang.srt.models.kimi_k3.get_platform",
            return_value=SimpleNamespace(is_cuda=False),
        ):
            self.assertIsNone(KimiK3MoE._merge_front_weights(moe))


if __name__ == "__main__":
    unittest.main()
