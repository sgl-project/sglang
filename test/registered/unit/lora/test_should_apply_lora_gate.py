"""Unit tests for the should_apply_lora gate in LoRAManager.init_lora_modules.

Target modules are matched by name suffix, which is ambiguous for multimodal
models: an audio or vision tower that names its projections the same way as the
language model gets wrapped by `--lora-target-modules qkv_proj`, even though the
adapter carries no weights for it. Models veto those modules by defining
should_apply_lora.

LoRAManager is built through __new__ so the memory pool, adapter download and
CUDA setup in __init__ stay out of the way; only init_lora_modules runs.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede the lora_manager import

import re
import unittest
from unittest.mock import MagicMock

import torch.nn as nn

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Attention(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class _Layer(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.self_attn = _Attention(dim)


class _LayerStack(nn.Module):
    def __init__(self, n_layers: int = 2, dim: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(dim) for _ in range(n_layers)])


def _make_manager(base_model, target_modules, num_layers: int = 2):
    manager = LoRAManager.__new__(LoRAManager)
    manager.base_model = base_model
    manager.target_modules = set(target_modules)
    manager.base_hf_config = MagicMock(num_hidden_layers=num_layers)
    manager.experts_shared_outer_loras = False
    manager.lora_use_virtual_experts = False

    wrapped: list[str] = []
    # The real set_lora_module builds a device-bound layer; record the name and
    # hand back a mock, which tolerates the attribute writes the caller makes.
    manager.set_lora_module = lambda name, _module: (
        wrapped.append(name),
        MagicMock(),
    )[1]
    return manager, wrapped


class TestShouldApplyLoraGate(CustomTestCase):
    def test_gate_filters_encoder_tower(self):
        """A tower sharing the language model's naming must not be wrapped."""

        pattern = re.compile(
            r"^thinker\.model\.layers\.\d+\.self_attn\.(?:qkv_proj|o_proj)$"
        )

        class _Thinker(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _LayerStack()
                self.audio_tower = _LayerStack()

        class _Multimodal(nn.Module):
            def __init__(self):
                super().__init__()
                self.thinker = _Thinker()

            def should_apply_lora(self, module_name: str) -> bool:
                return bool(pattern.match(module_name))

        manager, wrapped = _make_manager(_Multimodal(), {"qkv_proj", "o_proj"})
        manager.init_lora_modules()

        self.assertEqual(
            [name for name in wrapped if "audio_tower" in name],
            [],
            msg="audio_tower modules leaked past the gate",
        )
        self.assertEqual(
            set(wrapped),
            {
                "thinker.model.layers.0.self_attn.qkv_proj",
                "thinker.model.layers.0.self_attn.o_proj",
                "thinker.model.layers.1.self_attn.qkv_proj",
                "thinker.model.layers.1.self_attn.o_proj",
            },
        )

    def test_model_without_hook_keeps_suffix_match(self):
        """Models that don't define the hook keep the plain suffix behavior."""

        class _Vanilla(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _LayerStack()

        manager, wrapped = _make_manager(_Vanilla(), {"qkv_proj"})
        manager.init_lora_modules()

        self.assertEqual(
            set(wrapped),
            {
                "model.layers.0.self_attn.qkv_proj",
                "model.layers.1.self_attn.qkv_proj",
            },
        )

    def test_deny_all_hook_wraps_nothing(self):
        class _DenyAll(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _LayerStack()

            def should_apply_lora(self, module_name: str) -> bool:
                return False

        manager, wrapped = _make_manager(_DenyAll(), {"qkv_proj", "o_proj"})
        manager.init_lora_modules()

        self.assertEqual(wrapped, [])


if __name__ == "__main__":
    unittest.main()
