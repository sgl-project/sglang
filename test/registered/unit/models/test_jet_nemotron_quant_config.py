"""Every JetBlock projection is built with the checkpoint's quant config.

The reference Jet-Nemotron ``JetBlock`` defines ``o_proj`` as a plain
``nn.Linear`` next to ``q/k/v/a/b/g_proj``, so quantization tools quantize it
together with them. SGLang must therefore build ``o_proj`` with the
checkpoint's ``quant_config`` and under its checkpoint name, or a quantized
checkpoint's ``o_proj`` tensors have nowhere to load and the quantization
``ignore`` list cannot name it.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.jet_nemotron import JetNemotronConfig
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.jet_nemotron import JetBlock
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


def _tiny_jet_config() -> JetNemotronConfig:
    # Field names and kinds follow jet-ai/Jet-Nemotron-2B's config.json,
    # shrunk so the block builds instantly on CPU.
    return JetNemotronConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=1e4,
        hidden_act="silu",
        layer_types=["jet"],
        efficient_attention_config={
            "jet": {
                "mode": "chunk",
                "expand_v": 2.0,
                "num_heads": 2,
                "head_dim": 8,
                "norm_eps": "1e-5",
                "conv_size": 4,
                "dconv_generator_reduction": 8,
                "dconv_implementation": "triton",
            }
        },
    )


class TestJetBlockQuantConfig(CustomTestCase):
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

    def _build(self):
        quant_config = _RecordingQuantConfig()
        block = JetBlock(
            _tiny_jet_config(), layer_id=0, quant_config=quant_config, prefix=_PREFIX
        )
        return block, quant_config

    def test_input_projection_uses_quant_config(self):
        """Sanity check: the fused input projection already gets it."""
        _, quant_config = self._build()
        self.assertIn(f"{_PREFIX}.qkvabz_proj", quant_config.linear_prefixes)

    def test_o_proj_uses_quant_config_under_its_checkpoint_name(self):
        block, quant_config = self._build()
        self.assertIn(
            f"{_PREFIX}.o_proj",
            quant_config.linear_prefixes,
            "JetBlock.o_proj was built without the checkpoint's quant_config "
            "(or without prefix), so a quantized checkpoint's o_proj tensors "
            f"cannot load. Linear layers that asked: {quant_config.linear_prefixes}",
        )
        self.assertIsNotNone(block.o_proj.quant_method)


if __name__ == "__main__":
    unittest.main()
