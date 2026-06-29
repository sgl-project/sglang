import math
import unittest

from sglang.srt.configs.model_config import compute_mla_mscale_scaling
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMlaMscaleScaling(CustomTestCase):
    def test_respects_disabled_yarn_scaling(self):
        base_scaling = 1 / math.sqrt(128)
        rope_scaling = {
            "rope_type": "deepseek_yarn",
            "factor": 128,
            "mscale_all_dim": 1,
            "apply_yarn_scaling": False,
        }

        self.assertEqual(
            compute_mla_mscale_scaling(rope_scaling, base_scaling), base_scaling
        )

    def test_applies_yarn_scaling_by_default(self):
        base_scaling = 1 / math.sqrt(128)
        rope_scaling = {
            "rope_type": "deepseek_yarn",
            "factor": 128,
            "mscale_all_dim": 1,
        }

        self.assertGreater(
            compute_mla_mscale_scaling(rope_scaling, base_scaling), base_scaling
        )

    def test_respects_disabled_native_apply_scale(self):
        base_scaling = 1 / math.sqrt(128)
        rope_scaling = {
            "rope_type": "mistral",
            "factor": 128,
            "mscale_all_dim": 1,
            "apply_scale": False,
        }

        self.assertEqual(
            compute_mla_mscale_scaling(rope_scaling, base_scaling), base_scaling
        )


if __name__ == "__main__":
    unittest.main()
