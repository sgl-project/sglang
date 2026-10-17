"""GLM-5.3-Flash compressed-tensors checkpoints list the KDA forget gate in ``ignore``
by the transformers module path ``self_attn.forget_gate.f_{a,b}_proj``; the checkpoint
tensors and SGLang's modules are ``self_attn.f_{a,b}_proj``. Without remapping, compressed-tensors finds
neither an ignore entry nor a target for those layers and raises at construction.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HF_LAYER = "model.language_model.layers.0"
PREFIX = "model.layers.0.self_attn"

# Layer-0 subset of RedHatAI/GLM-5.3-Flash-NVFP4@18d55bfd's quantization_config.
NVFP4_ARGS = {
    "num_bits": 4,
    "type": "float",
    "symmetric": True,
    "strategy": "tensor_group",
    "group_size": 16,
    "dynamic": False,
}
QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "config_groups": {
        "group_0": {
            "format": "nvfp4-pack-quantized",
            "targets": [
                "re:.*\\.layers\\.(?:[3-9]|[1-3][0-9]|4[0-4])\\.mlp\\.experts\\..*(gate|up|down)_proj$"
            ],
            "weights": NVFP4_ARGS,
            "input_activations": {**NVFP4_ARGS, "dynamic": "local"},
        }
    },
    "ignore": [
        f"{HF_LAYER}.self_attn.{name}"
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "forget_gate",
            "forget_gate.f_a_proj",
            "forget_gate.f_b_proj",
            "b_proj",
            "g_a_proj",
            "g_b_proj",
            "o_norm",
            "o_proj",
        )
    ],
    "packed_modules_mapping": Glm5NextForConditionalGeneration.packed_modules_mapping,
}


class TestGlm5NextCompressedTensorsIgnore(CustomTestCase):
    def setUp(self):
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)
        override = get_context().override_server_args(
            device="cpu", enable_lora=False, lora_paths=None
        )
        override.install()
        self.addCleanup(override.restore)
        parallel = get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_tp_size=1,
        )
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)

        self.quant_config = CompressedTensorsConfig.from_config(QUANT_CONFIG)
        self.quant_config.apply_weight_name_mapper(
            Glm5NextForConditionalGeneration.hf_to_sglang_mapper
        )

    @torch.no_grad()
    def test_forget_gate_ignore_entries_reach_kda_projections(self):
        hidden, heads, dim = 16, 4, 8
        attention = glm5_next.Glm5NextLinearAttention(
            layer_idx=0,
            hidden_size=hidden,
            config=SimpleNamespace(
                linear_attn_config={
                    "head_dim": dim,
                    "num_heads": heads,
                    "short_conv_kernel_size": 4,
                }
            ),
            quant_config=self.quant_config,
            prefix=PREFIX,
        )
        checkpoint = {
            "f_a_proj": torch.randn(dim, hidden),
            "f_b_proj": torch.randn(heads * dim, dim),
        }
        for name in checkpoint:
            self.assertIsInstance(
                getattr(attention, name).quant_method, UnquantizedLinearMethod
            )
            getattr(attention, name).weight.fill_(torch.nan)

        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=0),
            num_fused_shared_experts=0,
            quant_config=self.quant_config,
            named_parameters=lambda: (
                (f"{PREFIX}.{name}", param)
                for name, param in attention.named_parameters()
            ),
        )
        with patch.object(glm5_next.DeepseekV2WeightLoaderMixin, "post_load_weights"):
            Glm5NextForConditionalGeneration.load_weights(
                model,
                [
                    (f"{HF_LAYER}.self_attn.{name}.weight", weight)
                    for name, weight in checkpoint.items()
                ],
            )
        for name, weight in checkpoint.items():
            torch.testing.assert_close(
                getattr(attention, name).weight, weight, atol=0, rtol=0
            )

    def test_experts_keep_nvfp4_target(self):
        scheme = self.quant_config.get_scheme_dict(
            torch.nn.Module(), "model.layers.3.mlp.experts.0.gate_proj"
        )
        self.assertEqual(scheme["format"], "nvfp4-pack-quantized")


if __name__ == "__main__":
    unittest.main()
