import unittest
from types import SimpleNamespace

from sglang.srt.models.laguna import LagunaForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLagunaForCausalLM(CustomTestCase):
    def test_attention_sliding_window_is_exclusive(self):
        model = SimpleNamespace(config=SimpleNamespace(sliding_window=512))

        self.assertEqual(
            LagunaForCausalLM.get_attention_sliding_window_size(model), 511
        )


if __name__ == "__main__":
    unittest.main()
