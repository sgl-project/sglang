"""GPT-J and Granite 4.0 hybrid load_weights under quantized checkpoints."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GPTJConfig

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.gpt_j import GPTJForCausalLM
from sglang.srt.models.granitemoehybrid import GraniteMoeHybridForCausalLM
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_EXPERTS = 4
MOE = "model.layers.0.block_sparse_moe"


class _PlainQuantConfig(QuantizationConfig):
    """A quantization config with the base interface only, as every config
    except compressed-tensors is: it has no get_cache_scale."""

    def get_name(self):
        return "plain"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

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
            return UnquantizedLinearMethod()
        return None


def _compressed_tensors_with_fp8_kv():
    return CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[],
        quant_format="float-quantized",
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
        kv_cache_scheme={
            "type": "float",
            "num_bits": 8,
            "strategy": "tensor",
            "symmetric": True,
            "dynamic": False,
        },
    )


def _gpt_j(quant_config):
    config = GPTJConfig(
        n_embd=32,
        n_head=4,
        n_layer=1,
        n_positions=32,
        rotary_dim=8,
        vocab_size=64,
        tie_word_embeddings=False,
    )
    return GPTJForCausalLM(config, quant_config=quant_config)


class _ExpertParam(torch.nn.Parameter):
    """A fused expert parameter whose loader has FusedMoE.weight_loader's
    signature and records each call."""

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        self.calls.append((weight_name, shard_id, expert_id))


def _granite_hybrid(quant_config):
    """GraniteMoeHybridForCausalLM as far as load_weights reads it: the
    parameters of one MoE block and the final norm."""
    experts = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        param = _ExpertParam(torch.zeros(1), requires_grad=False)
        param.calls = []
        experts.register_parameter(name, param)
    layer = torch.nn.Module()
    layer.block_sparse_moe = torch.nn.Module()
    layer.block_sparse_moe.experts = experts
    backbone = torch.nn.Module()
    backbone.layers = torch.nn.ModuleList([layer])
    backbone.norm = torch.nn.LayerNorm(4)

    model = object.__new__(GraniteMoeHybridForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = backbone
    model.config = SimpleNamespace(num_local_experts=NUM_EXPERTS)
    model.quant_config = quant_config
    return model


class _ContextTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
            )
        )


class TestLoadersUnderQuantConfigs(_ContextTestCase):
    def test_gpt_j_loads_under_a_config_without_cache_scale_lookup(self):
        """Only compressed-tensors maps checkpoint names to KV cache scales;
        any other quantization config still loads the checkpoint."""
        model = _gpt_j(_PlainQuantConfig())
        weight = torch.randn_like(model.transformer.wte.weight)
        model.load_weights([("transformer.wte.weight", weight)])
        self.assertTrue(torch.equal(model.transformer.wte.weight, weight))

    def test_gpt_j_loads_compressed_tensors_kv_scales(self):
        model = _gpt_j(_compressed_tensors_with_fp8_kv())
        model.load_weights(
            [
                ("transformer.h.0.attn.k_proj.output_scale", torch.tensor([0.5])),
                ("transformer.h.0.attn.v_proj.output_scale", torch.tensor([0.25])),
            ]
        )
        attn = model.transformer.h[0].attn.attn
        self.assertEqual(attn.k_scale.item(), 0.5)
        self.assertEqual(attn.v_scale.item(), 0.25)

    def test_granite_hybrid_loads_under_a_config_without_cache_scale_lookup(self):
        model = _granite_hybrid(_PlainQuantConfig())
        weight = torch.randn(4)
        model.load_weights([("model.norm.weight", weight)])
        self.assertTrue(torch.equal(model.model.norm.weight, weight))

    def test_granite_hybrid_loads_per_expert_tensors(self):
        """llmcompressor checkpoints store one tensor per expert and
        projection; each reaches the fused parameter's loader with its shard
        and expert id."""
        model = _granite_hybrid(None)
        model.load_weights(
            [
                (f"{MOE}.experts.3.gate_proj.weight", torch.zeros(2)),
                (f"{MOE}.experts.2.up_proj.weight", torch.zeros(2)),
                (f"{MOE}.experts.1.down_proj.weight", torch.zeros(2)),
            ]
        )
        experts = model.model.layers[0].block_sparse_moe.experts
        self.assertEqual(
            experts.w13_weight.calls,
            [
                (f"{MOE}.experts.w13_weight", "w1", 3),
                (f"{MOE}.experts.w13_weight", "w3", 2),
            ],
        )
        self.assertEqual(
            experts.w2_weight.calls, [(f"{MOE}.experts.w2_weight", "w2", 1)]
        )


if __name__ == "__main__":
    unittest.main()
