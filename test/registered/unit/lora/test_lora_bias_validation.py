"""Reject trained bias adapters before changing LoRA loading state."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestLoRABiasValidation(CustomTestCase):
    @staticmethod
    def config(bias):
        config = {"target_modules": ["q_proj", "v_proj"], "r": 2, "lora_alpha": 2}
        if bias is not None:
            config["bias"] = bias
        return config

    @staticmethod
    def manager():
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_hf_config = SimpleNamespace(vocab_size=16)
        manager.enable_dp_attention = False
        manager.init_lora_adapters()
        manager.load_lora_weights = Mock()
        manager.load_lora_weights_from_tensors = Mock()
        return manager

    def test_supported_bias_configurations(self):
        for bias in (None, "none"):
            with self.subTest(bias=bias):
                config = LoRAConfig.from_dict(self.config(bias))
                self.assertEqual(config.bias, "none")
                self.manager().validate_new_adapter(config, LoRARef(pinned=False))

    def test_loading_rejects_bias_before_mutating_state(self):
        for bias in ("lora_only", "all"):
            for route in ("file", "tensor"):
                with (
                    self.subTest(bias=bias, route=route),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    config = self.config(bias)
                    (Path(directory) / "adapter_config.json").write_text(
                        json.dumps(config)
                    )
                    ref = LoRARef(lora_name="biased", lora_path=directory, pinned=False)
                    manager = self.manager()
                    result = (
                        manager._load_lora_adapter(ref)
                        if route == "file"
                        else manager.load_lora_adapter_from_tensors(ref, {}, config)
                    )
                    self.assertFalse(result.success)
                    self.assertIn("biased", result.error_message)
                    self.assertIn(f"bias={bias!r}", result.error_message)
                    self.assertIn("trained with bias='none'", result.error_message)
                    self.assertEqual(manager.configs, {})
                    self.assertEqual(manager.loras, {})
                    self.assertEqual(manager.lora_refs, {})
                    self.assertEqual(manager.num_pinned_loras, 0)
                    manager.load_lora_weights.assert_not_called()
                    manager.load_lora_weights_from_tensors.assert_not_called()

    def test_initial_adapter_rejects_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "adapter_config.json").write_text(
                json.dumps(self.config("all"))
            )
            ref = LoRARef(lora_name="initial", lora_path=directory, pinned=False)
            manager = self.manager()
            with self.assertRaisesRegex(RuntimeError, "bias='all'"):
                manager.init_lora_adapters([ref])
            self.assertEqual(manager.configs, {})
            self.assertEqual(manager.lora_refs, {})
            manager.load_lora_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
