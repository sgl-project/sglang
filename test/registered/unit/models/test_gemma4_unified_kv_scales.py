"""Real Unified Gemma constructor/loading contracts; no GPU kernels."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
import unittest
from types import SimpleNamespace

import torch
from transformers import Gemma4UnifiedConfig

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration
from sglang.srt.runtime_context import (
    get_context,
    get_parallel,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class TestUnifiedKVScales(CustomTestCase):
    def setUp(self):
        super().setUp()
        state = snapshot_context()
        state["__parallel__"] = {
            k: v.copy() if isinstance(v, dict) else v
            for k, v in state["__parallel__"].items()
        }
        self.addCleanup(restore_context, state)
        get_context().set_server_args(
            ServerArgs(model_path="unused", device="cpu", boundary_reduction="ar")
        )
        get_parallel().override_permanently(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            pp_group=SimpleNamespace(
                world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
            ),
            tp_group=None,
            pp_rank=0,
            pp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            dp_size=1,
        )

    def model(self):
        config = Gemma4UnifiedConfig()
        text = config.text_config
        for name, value in dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            vocab_size=64,
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=0,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=64,
            global_head_dim=128,
            num_kv_shared_layers=0,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=16,
        ).items():
            setattr(text, name, value)
        text.allow_global_per_layer_attribute_access = True
        quant = CompressedTensorsConfig.from_config(
            {
                "format": "dense",
                "quant_method": "compressed-tensors",
                "config_groups": {},
                "ignore": [],
                "kv_cache_scheme": {
                    "type": "float",
                    "num_bits": 8,
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
            }
        )
        with torch.device("meta"):
            model = Gemma4UnifiedForConditionalGeneration(config, quant_config=quant)
        return model

    def test_actual_loader_retains_serialized_fp8_scales(self):
        model = self.model()
        weights = []
        for index, shell in enumerate(model.language_model.layers):
            layer = shell.self_attn.attn
            for axis, value in [
                ("k", 0.125 * (index + 1)),
                ("v", 0.375 * (index + 1)),
            ]:
                setattr(
                    layer,
                    axis + "_scale",
                    torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False),
                )
                weights.append(
                    (
                        f"model.language_model.layers.{index}.self_attn.{axis}_scale",
                        torch.tensor([value]),
                    )
                )
        loaded = model.load_weights(iter(weights))
        for index, shell in enumerate(model.language_model.layers):
            layer = shell.self_attn.attn
            self.assertEqual(layer.k_scale.item(), 0.125 * (index + 1))
            self.assertEqual(layer.v_scale.item(), 0.375 * (index + 1))
            self.assertIn(
                f"language_model.layers.{index}.self_attn.attn.k_scale", loaded
            )
            self.assertIn(
                f"language_model.layers.{index}.self_attn.attn.v_scale", loaded
            )


if __name__ == "__main__":
    unittest.main()
