"""CPU tests: a Qwen3-Omni quantization ignore list reaches the layers SGLang builds.

Checkpoints name thinker modules `thinker.visual.*` / `thinker.audio_tower.*` /
`thinker.model.*` and keep unfused names (`attn.qkv`, `self_attn.q_proj`), while
SGLang builds `visual.*` / `audio_tower.*` / `model.language_model.*` with fused
`attn.qkv_proj`. The model's quant-name mapper must bridge the two.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.test.test_utils import CustomTestCase

# Excerpt of cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit's ignore list: the
# checkpoint stores all of these BF16 and quantizes the thinker's attention and
# routed experts.
OMNI_AWQ_IGNORE = [
    "thinker.visual.blocks.0.attn.qkv",
    "thinker.visual.blocks.0.attn.proj",
    "thinker.visual.blocks.0.mlp.linear_fc1",
    "thinker.visual.blocks.0.mlp.linear_fc2",
    "thinker.visual.merger.mlp.0",
    "thinker.visual.merger.mlp.2",
    "thinker.visual.merger_list.0.mlp.0",
    "thinker.visual.merger_list.0.mlp.2",
    "thinker.audio_tower.conv_out",
    "thinker.audio_tower.proj1",
    "thinker.audio_tower.proj2",
    "thinker.audio_tower.layers.0.self_attn.q_proj",
    "thinker.audio_tower.layers.0.self_attn.k_proj",
    "thinker.audio_tower.layers.0.self_attn.v_proj",
    "thinker.audio_tower.layers.0.self_attn.out_proj",
    "thinker.audio_tower.layers.0.fc1",
    "thinker.audio_tower.layers.0.fc2",
    "thinker.model.layers.0.mlp.gate",
    "thinker.lm_head",
]

WNA16_GROUP = {
    "format": "pack-quantized",
    "targets": ["Linear"],
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": False,
    },
    "input_activations": None,
}


def _omni_scheme():
    """Scheme lookup with the model's mapper applied, as the loader applies it."""
    hf_config = SimpleNamespace(
        thinker_config=SimpleNamespace(audio_config=SimpleNamespace(encoder_layers=32))
    )
    quant_config = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": WNA16_GROUP},
            "ignore": list(OMNI_AWQ_IGNORE),
        }
    )
    quant_config.apply_weight_name_mapper(
        Qwen3OmniMoeForConditionalGeneration.get_hf_to_sglang_mapper(hf_config)
    )
    return lambda name: quant_config.get_scheme_dict(torch.nn.Linear(1, 1), name)


class TestQwen3OmniQuantNames(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.object(
            CompressedTensorsConfig, "_check_scheme_supported", return_value=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scheme = _omni_scheme()

    def test_ignored_vision_layers_are_built_bf16(self):
        """The BF16 vision tower is not built quantized (it trips group_size on main)."""
        for name in (
            "visual.blocks.0.attn.qkv_proj",
            "visual.blocks.0.attn.proj",
            "visual.blocks.0.mlp.linear_fc1",
            "visual.blocks.0.mlp.linear_fc2",
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
            "visual.merger_list.0.mlp.0",
            "visual.merger_list.0.mlp.2",
        ):
            with self.subTest(name=name):
                self.assertIsNone(self.scheme(name))

    def test_ignored_audio_layers_are_built_bf16(self):
        for name in (
            "audio_tower.conv_out",
            "audio_tower.proj1",
            "audio_tower.proj2",
            "audio_tower.layers.0.attn.qkv_proj",
            "audio_tower.layers.0.attn.proj",
            "audio_tower.layers.0.fc1",
            "audio_tower.layers.0.fc2",
        ):
            with self.subTest(name=name):
                self.assertIsNone(self.scheme(name))

    def test_quantized_thinker_layers_stay_quantized(self):
        """Only the listed layers change: the thinker's attention and other blocks keep a scheme."""
        for name in (
            "model.language_model.layers.0.self_attn.qkv_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "visual.blocks.1.attn.qkv_proj",
            "audio_tower.layers.1.attn.qkv_proj",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(self.scheme(name))

    def test_text_model_names_are_renamed_to_language_model(self):
        self.assertIsNone(self.scheme("model.language_model.layers.0.mlp.gate"))
        self.assertIsNone(self.scheme("lm_head"))


if __name__ == "__main__":
    unittest.main()
