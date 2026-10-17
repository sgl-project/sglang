"""CPU unit tests for PEFT rsLoRA scaling in LoRAAdapter."""

import math
import unittest
from types import SimpleNamespace

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _adapter(config: LoRAConfig) -> LoRAAdapter:
    return LoRAAdapter(
        uid="test",
        config=config,
        base_hf_config=SimpleNamespace(num_hidden_layers=1),
        load_config=None,
        lora_backend=None,
    )


class TestRsLoRAScaling(CustomTestCase):
    def test_standard_lora_scaling(self):
        config = LoRAConfig.from_dict(
            {
                "peft_type": "LORA",
                "r": 64,
                "lora_alpha": 128,
                "target_modules": ["q_proj"],
                "use_rslora": False,
            }
        )
        self.assertFalse(config.use_rslora)
        self.assertEqual(_adapter(config).scaling, 128 / 64)

    def test_rslora_scaling_matches_peft(self):
        # PEFT: scaling = lora_alpha / sqrt(r) when use_rslora=True
        config = LoRAConfig.from_dict(
            {
                "peft_type": "LORA",
                "r": 64,
                "lora_alpha": 128,
                "target_modules": ["q_proj"],
                "use_rslora": True,
            }
        )
        self.assertTrue(config.use_rslora)
        self.assertEqual(_adapter(config).scaling, 128 / math.sqrt(64))

    def test_use_rslora_defaults_false(self):
        config = LoRAConfig.from_dict(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj"],
            }
        )
        self.assertFalse(config.use_rslora)
        self.assertEqual(_adapter(config).scaling, 32 / 16)


if __name__ == "__main__":
    unittest.main()
