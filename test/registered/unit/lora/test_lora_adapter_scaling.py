"""LoRA adapter scaling must match PEFT for standard and rsLoRA adapters."""

import unittest
from types import SimpleNamespace

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_adapter(**peft_fields) -> LoRAAdapter:
    config = LoRAConfig(
        config_dict={
            "peft_type": "LORA",
            "target_modules": ["q_proj"],
            "r": 64,
            "lora_alpha": 128,
            **peft_fields,
        }
    )
    return LoRAAdapter(
        uid="adapter",
        config=config,
        base_hf_config=SimpleNamespace(num_hidden_layers=1),
        load_config=None,
        lora_backend=None,
    )


class TestLoRAAdapterScaling(CustomTestCase):
    def test_rslora_adapter_scales_by_square_root_of_rank(self):
        """An rsLoRA adapter (use_rslora=True) got alpha / r instead of
        alpha / sqrt(r), so its update was 8 times too weak at r=64."""
        self.assertEqual(_make_adapter(use_rslora=True).scaling, 16.0)

    def test_standard_adapter_scales_by_rank(self):
        """An adapter without use_rslora keeps the alpha / r scale."""
        self.assertEqual(_make_adapter().scaling, 2.0)


if __name__ == "__main__":
    unittest.main()
