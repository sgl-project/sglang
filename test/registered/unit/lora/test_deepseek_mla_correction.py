import unittest
from types import SimpleNamespace

from sglang.srt.lora.deepseek_mla_correction import is_kv_b_lora_active
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestDeepseekMLACorrection(unittest.TestCase):
    def test_kv_b_lora_probe(self):
        self.assertFalse(is_kv_b_lora_active(SimpleNamespace()))
        self.assertFalse(
            is_kv_b_lora_active(SimpleNamespace(kv_b_proj=SimpleNamespace()))
        )
        self.assertTrue(
            is_kv_b_lora_active(
                SimpleNamespace(kv_b_proj=SimpleNamespace(set_lora=True))
            )
        )


if __name__ == "__main__":
    unittest.main()
