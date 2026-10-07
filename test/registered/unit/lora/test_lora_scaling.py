"""Exercise real rsLoRA config and scaling on CPU, without loading a model."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.lora import lora_config
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_adapter_class():
    # These imports pull in GPU backends/model loaders, but are not used by
    # the constructor or dense o_proj normalization exercised here. Keep the
    # actual config and adapter implementations, restoring imports afterwards.
    dependencies = {
        "sglang.srt.configs.load_config": "LoadConfig",
        "sglang.srt.layers.utils": "get_layer_id",
        "sglang.srt.lora.backend.base_backend": "BaseLoRABackend",
        "sglang.srt.model_loader.loader": "DefaultModelLoader",
        "sglang.srt.utils.hf_transformers_utils": "AutoConfig",
    }
    stubs = {}
    for name, attribute in dependencies.items():
        stubs[name] = ModuleType(name)
        setattr(stubs[name], attribute, object)
    source = Path(lora_config.__file__).with_name("lora.py")
    spec = importlib.util.spec_from_file_location("isolated_lora_scaling", source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict("sys.modules", stubs):
        spec.loader.exec_module(module)
    return module.LoRAAdapter


LoRAAdapter = _load_adapter_class()

_CONFIG = {
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["o_proj"],
}


def _new_adapter(config):
    return LoRAAdapter(
        "dense-scaling-test",
        config,
        SimpleNamespace(num_hidden_layers=1, hidden_size=5),
        None,
        None,
    )


class TestLoRAScaling(unittest.TestCase):
    def test_scaling_from_dict(self):
        for rank in (1, 8, 64):
            for flag in (None, False, True):
                with self.subTest(rank=rank, use_rslora=flag):
                    options = {} if flag is None else {"use_rslora": flag}
                    config = LoRAConfig.from_dict({**_CONFIG, "r": rank, **options})
                    adapter = _new_adapter(config)
                    self.assertIs(config.use_rslora, flag is True)
                    denominator = math.sqrt(rank) if flag else rank
                    self.assertEqual(adapter.scaling, 16 / denominator)

    def test_scaling_from_local_adapter_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_config.json"
            for flag in (None, False, True):
                with self.subTest(use_rslora=flag):
                    options = {} if flag is None else {"use_rslora": flag}
                    path.write_text(json.dumps({**_CONFIG, **options}))
                    config = LoRAConfig(path=directory)
                    self.assertIs(config.use_rslora, flag is True)
                    denominator = math.sqrt(8) if flag else 8
                    self.assertEqual(_new_adapter(config).scaling, 16 / denominator)

    def test_rslora_scales_dense_delta_without_changing_saved_factors(self):
        rank = 8
        a = torch.arange(rank * 5, dtype=torch.float64).reshape(rank, 5) / 16
        b = torch.arange(3 * rank, dtype=torch.float64).reshape(3, rank) / 32
        saved_a, saved_b = a.clone(), b.clone()
        input_ = torch.tensor([1.0, -2.0, 0.5, 0.25, 3.0], dtype=torch.float64)
        expected = (saved_b @ saved_a) @ input_
        prefix = "model.layers.0.self_attn.o_proj"
        deltas = []
        for flag in (False, True):
            config = LoRAConfig.from_dict({**_CONFIG, "use_rslora": flag})
            adapter = _new_adapter(config)
            weights = adapter.layers[0].weights
            weights[f"{prefix}.lora_A.weight"] = a
            weights[f"{prefix}.lora_B.weight"] = b
            adapter._normalize_weights()
            self.assertIs(weights[f"{prefix}.lora_A.weight"], a)
            self.assertIs(weights[f"{prefix}.lora_B.weight"], b)
            torch.testing.assert_close(a, saved_a, rtol=0, atol=0)
            torch.testing.assert_close(b, saved_b, rtol=0, atol=0)
            delta = (b @ (a @ input_)) * adapter.scaling
            denominator = math.sqrt(rank) if flag else rank
            torch.testing.assert_close(delta, expected * (16 / denominator))
            deltas.append(delta)
        torch.testing.assert_close(deltas[1], deltas[0] * math.sqrt(rank))


if __name__ == "__main__":
    unittest.main()
