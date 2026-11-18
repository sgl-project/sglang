"""Step-3.5's head-wise attention gate ``g_proj`` follows the checkpoint's quant config.

``g_proj`` is a plain linear projection next to ``q/k/v/o_proj``. Some quantized
Step-3.5 checkpoints quantize it with them (for example a compressed-tensors
W4A16 checkpoint stores ``self_attn.g_proj.weight_packed`` and
``self_attn.g_proj.weight_scale``), while the official FP8 checkpoint keeps it
in bf16 through ``modules_to_not_convert``. SGLang must therefore build
``g_proj`` with the checkpoint's ``quant_config`` and let the config decide:
building it unquantized drops the quantized tensors during loading and leaves
``g_proj.weight`` uninitialized.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.step3p5 import Step3p5Attention
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PREFIX = "model.layers.0.self_attn"


class _RecordingQuantConfig(QuantizationConfig):
    """Builds every layer unquantized and records which linear layers asked."""

    def __init__(self):
        super().__init__()
        self.linear_prefixes: list[str] = []

    def get_name(self):
        return "recording"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_scaled_act_names(self):
        return []

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            self.linear_prefixes.append(prefix)
            return UnquantizedLinearMethod()
        return None


class TestStep3p5AttentionQuantConfig(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
            )
        )
        # Only the projections are under test; the CPU rotary-embedding
        # fallback needs vllm, so stand it in.
        rope_patch = patch(
            "sglang.srt.models.step3p5.get_rope", return_value=torch.nn.Identity()
        )
        rope_patch.start()
        self.addCleanup(rope_patch.stop)

    def _build(self, quant_config):
        # Shrunk from stepfun-ai/Step-3.5-Flash so the layer builds instantly on
        # CPU, keeping the projection sizes divisible by the FP8 128x128 block.
        return Step3p5Attention(
            hidden_size=128,
            num_heads=4,
            num_kv_heads=2,
            layer_id=0,
            head_dim=64,
            max_position_embeddings=64,
            quant_config=quant_config,
            rms_norm_eps=1e-5,
            use_head_wise_attn_gate=True,
            prefix=_PREFIX,
        )

    def test_qkv_and_o_proj_use_quant_config(self):
        """Sanity check: the other attention projections already get it."""
        quant_config = _RecordingQuantConfig()
        self._build(quant_config)
        self.assertIn(f"{_PREFIX}.qkv_proj", quant_config.linear_prefixes)
        self.assertIn(f"{_PREFIX}.o_proj", quant_config.linear_prefixes)

    def test_g_proj_uses_quant_config_under_its_checkpoint_name(self):
        quant_config = _RecordingQuantConfig()
        attn = self._build(quant_config)
        self.assertIn(
            f"{_PREFIX}.g_proj",
            quant_config.linear_prefixes,
            "Step3p5Attention.g_proj was built without the checkpoint's "
            "quant_config, so a checkpoint that quantizes g_proj loads into an "
            f"uninitialized weight. Linear layers that asked: "
            f"{quant_config.linear_prefixes}",
        )
        self.assertIsNotNone(attn.g_proj.quant_method)

    def test_fp8_checkpoint_keeps_g_proj_unquantized(self):
        """The official FP8 checkpoint lists g_proj in modules_to_not_convert."""
        quant_config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "lm_head",
                    "model.embed_tokens",
                    "model.norm",
                    f"{_PREFIX}.g_proj",
                    f"{_PREFIX}.qkv_proj",
                    f"{_PREFIX}.o_proj",
                ],
            }
        )
        attn = self._build(quant_config)
        self.assertIsInstance(attn.g_proj.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
