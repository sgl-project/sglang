"""A PEFT adapter whose rank_pattern or alpha_pattern changes a module's rank or
alpha must fail to load, on every load route, without leaving loading state behind."""

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


class TestLoRAPatternValidation(CustomTestCase):
    @staticmethod
    def config(**patterns):
        return {
            "target_modules": ["q_proj", "v_proj", "o_proj"],
            "r": 2,
            "lora_alpha": 4,
            **patterns,
        }

    @staticmethod
    def manager():
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_hf_config = SimpleNamespace(vocab_size=16)
        manager.attn_dp_enabled = False
        manager.init_lora_adapters()
        manager.load_lora_weights = Mock()
        manager.load_lora_weights_from_tensors = Mock()
        return manager

    def test_uniform_adapters_are_accepted(self):
        cases = {
            "absent": {},
            "empty": {"rank_pattern": {}, "alpha_pattern": {}},
            "null": {"rank_pattern": None, "alpha_pattern": None},
            # An entry that repeats the adapter-wide value changes nothing.
            "redundant": {
                "rank_pattern": {"o_proj": 2},
                "alpha_pattern": {"o_proj": 4},
            },
        }
        for name, patterns in cases.items():
            with self.subTest(name):
                config = LoRAConfig.from_dict(self.config(**patterns))
                self.assertEqual(config.rank_overrides, {})
                self.assertEqual(config.alpha_overrides, {})
                self.manager().validate_new_adapter(config, LoRARef(pinned=False))

    def test_loading_rejects_overrides_before_mutating_state(self):
        cases = {
            "rank_pattern": {"rank_pattern": {"o_proj": 8}},
            "alpha_pattern": {"alpha_pattern": {"o_proj": 32}},
            "alpha_pattern float": {"alpha_pattern": {"o_proj": 4.5}},
        }
        for name, patterns in cases.items():
            field = name.split()[0]
            for route in ("file", "tensor"):
                with (
                    self.subTest(name, route=route),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    config = self.config(**patterns)
                    (Path(directory) / "adapter_config.json").write_text(
                        json.dumps(config)
                    )
                    ref = LoRARef(
                        lora_name="per-module", lora_path=directory, pinned=False
                    )
                    manager = self.manager()
                    result = (
                        manager._load_lora_adapter(ref)
                        if route == "file"
                        else manager.load_lora_adapter_from_tensors(ref, {}, config)
                    )
                    self.assertFalse(result.success)
                    self.assertIn("per-module", result.error_message)
                    self.assertIn(field, result.error_message)
                    self.assertIn("o_proj", result.error_message)
                    self.assertEqual(manager.configs, {})
                    self.assertEqual(manager.loras, {})
                    self.assertEqual(manager.lora_refs, {})
                    self.assertEqual(manager.num_pinned_loras, 0)
                    manager.load_lora_weights.assert_not_called()
                    manager.load_lora_weights_from_tensors.assert_not_called()

    def test_initial_adapter_rejects_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "adapter_config.json").write_text(
                json.dumps(self.config(rank_pattern={"o_proj": 8}))
            )
            ref = LoRARef(lora_name="initial", lora_path=directory, pinned=False)
            manager = self.manager()
            with self.assertRaisesRegex(RuntimeError, "rank_pattern"):
                manager.init_lora_adapters([ref])
            self.assertEqual(manager.configs, {})
            self.assertEqual(manager.lora_refs, {})
            manager.load_lora_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
