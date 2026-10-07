"""Infer serving targets from PEFT ParamWrapper configs without loading a model."""

import copy
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from sglang.srt.lora import lora_config
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TARGETS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def _load_manager_class():
    # Target inference uses the real config and target-name utilities. The
    # remaining imports below initialize GPU/model runtime code; none of these
    # types is instantiated by init_lora_shapes. Restore all imports afterward.
    dependencies = {
        "sglang.srt.layers.moe.fused_moe_triton.layer": ("FusedMoE",),
        "sglang.srt.layers.vocab_parallel_embedding": (
            "ParallelLMHead",
            "VocabParallelEmbedding",
        ),
        "sglang.srt.lora.backend.lora_registry": ("get_backend_from_name",),
        "sglang.srt.lora.layers": (
            "BaseLayerWithLoRA",
            "FusedMoEWithLoRA",
            "get_lora_layer",
        ),
        "sglang.srt.lora.lora": ("LoRAAdapter",),
        "sglang.srt.lora.mem_pool": ("LoRAMemoryPool",),
        "sglang.srt.managers.io_struct": ("LoRAUpdateOutput",),
    }
    stubs = {}
    for name, attributes in dependencies.items():
        stubs[name] = ModuleType(name)
        for attribute in attributes:
            setattr(stubs[name], attribute, object)
    source = Path(lora_config.__file__).with_name("lora_manager.py")
    spec = importlib.util.spec_from_file_location("isolated_moe_target_manager", source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict("sys.modules", stubs):
        spec.loader.exec_module(module)
    return module.LoRAManager


LoRAManager = _load_manager_class()


def _config(target_modules, target_parameters=_TARGETS):
    return {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 16,
        "target_modules": target_modules,
        "target_parameters": target_parameters,
    }


def _infer_targets(config, server_targets=None):
    # Exercise the real target inference/CLI validation without constructing
    # model wrappers, allocating GPU buffers, or loading checkpoint weights.
    manager = LoRAManager.__new__(LoRAManager)
    manager.configs = {"adapter": config}
    manager.lora_refs = {"adapter": SimpleNamespace(lora_name="test-adapter")}
    manager.lora_added_tokens_size = None
    manager.init_lora_shapes(target_modules=server_targets)
    return manager.target_modules


class TestPEFTMoEConfig(CustomTestCase):
    def test_parameter_only_adapters_infer_both_expert_modules(self):
        for modules in ([], None):
            with self.subTest(target_modules=modules):
                saved = _config(modules)
                original = copy.deepcopy(saved)
                config = LoRAConfig.from_dict(saved)
                self.assertEqual(_infer_targets(config), {"gate_up_proj", "down_proj"})
                self.assertIs(config.hf_config, saved)
                self.assertEqual(saved, original)

    def test_mixed_dense_and_parameter_targets_are_combined(self):
        saved = _config(["q_proj", "down_proj"])
        original = copy.deepcopy(saved)
        config = LoRAConfig.from_dict(saved)
        self.assertEqual(
            _infer_targets(config), {"qkv_proj", "gate_up_proj", "down_proj"}
        )
        self.assertEqual(config.target_modules.count("down_proj"), 1)
        self.assertEqual(saved, original)

    def test_short_and_qualified_parameter_names_infer_targets(self):
        for target in (
            "down_proj",
            "experts.down_proj",
            "model.layers.3.mlp.experts.down_proj",
        ):
            with self.subTest(target=target):
                config = LoRAConfig.from_dict(_config([], [target]))
                self.assertEqual(_infer_targets(config), {"down_proj"})

    def test_unrelated_parameter_names_do_not_enable_expert_modules(self):
        config = LoRAConfig.from_dict(
            _config(
                ["q_proj"],
                [
                    "mlp.shared_expert.down_proj",
                    "mlp.notexperts.down_proj",
                    "mlp.experts.router",
                ],
            )
        )
        self.assertEqual(_infer_targets(config), {"qkv_proj"})

    def test_parameter_targets_are_checked_against_server_selection(self):
        config = LoRAConfig.from_dict(_config([]))
        self.assertEqual(
            _infer_targets(config, {"gate_proj", "down_proj"}),
            {"gate_up_proj", "down_proj"},
        )
        with self.assertRaisesRegex(ValueError, "gate_up_proj"):
            _infer_targets(config, {"down_proj"})

    def test_target_module_strings_keep_their_existing_semantics(self):
        for modules in ("all", "all-linear", ".*q_proj"):
            with self.subTest(target_modules=modules):
                config = LoRAConfig.from_dict(_config(modules))
                self.assertEqual(config.target_modules, modules)
                self.assertEqual(config.hf_config["target_modules"], modules)


if __name__ == "__main__":
    unittest.main()
