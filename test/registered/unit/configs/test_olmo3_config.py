import unittest

from sglang.srt.configs.olmo3 import Olmo3Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOlmo3Config(CustomTestCase):
    def test_default_architectures(self):
        cfg = Olmo3Config()
        self.assertEqual(cfg.architectures, ["Olmo2ForCausalLM"])

    def test_none_architectures_defaults_to_olmo2(self):
        cfg = Olmo3Config(architectures=None)
        self.assertEqual(cfg.architectures, ["Olmo2ForCausalLM"])

    def test_caller_architectures_list_not_mutated_and_order_preserved(self):
        caller_archs = ["Olmo3ForCausalLM", "CustomArch"]
        cfg = Olmo3Config(architectures=caller_archs)
        self.assertEqual(caller_archs, ["Olmo3ForCausalLM", "CustomArch"])
        self.assertEqual(cfg.architectures, ["Olmo2ForCausalLM", "CustomArch"])

    def test_tuple_architectures_supported(self):
        cfg = Olmo3Config(architectures=("Olmo3ForCausalLM",))
        self.assertEqual(cfg.architectures, ["Olmo2ForCausalLM"])


if __name__ == "__main__":
    unittest.main()
