"""Beam FP8 checkpoints load q/k/v/gate into one packed attention projection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.beam import BeamConfig
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.beam import BeamAttention, BeamForCausalLM
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TP_SIZE, TP_RANK = 8, 5
HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 16, 8, 8, 64


def fp8_dynamic(ignore=()):
    # Scheme of reflection/Beam-1.0-501B-A23B-FP8 (llm-compressor FP8_DYNAMIC).
    scheme = {"num_bits": 8, "type": "float", "symmetric": True}
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "ignore": ["lm_head", *ignore],
            "config_groups": {
                "FP8_DYNAMIC": {
                    "format": "float-quantized",
                    "targets": ["Linear"],
                    "weights": dict(scheme, strategy="channel", dynamic=False),
                    "input_activations": dict(scheme, strategy="token", dynamic=True),
                }
            },
            "packed_modules_mapping": BeamForCausalLM.packed_modules_mapping,
        }
    )


def checkpoint_values(rows, offset):
    return ((torch.arange(rows * HIDDEN) + offset) % 29 - 14).reshape(
        rows, HIDDEN
    ).float() / 16


class TestBeamFp8AttentionPacking(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        # Compressed-tensors checks FP8 support against the device capability.
        capability = patch("torch.cuda.get_device_capability", return_value=(9, 0))
        capability.start()
        self.addCleanup(capability.stop)
        server = ServerArgs(model_path="dummy", device="cpu", tp_size=TP_SIZE)
        publish(server, role="test", ranks=SpawnRanks(world_rank=TP_RANK))
        self.config = BeamConfig(
            hidden_size=HIDDEN,
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            num_hidden_layers=4,
        )

    def build(self, layer_id, quant_config):
        model = BeamForCausalLM.__new__(BeamForCausalLM)
        nn.Module.__init__(model)
        model.config = self.config
        model.start_layer, model.end_layer = 0, self.config.num_hidden_layers
        model.model = nn.Module()
        model.model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.model.layers = nn.ModuleList(nn.Module() for _ in range(layer_id + 1))
        attn = BeamAttention(
            self.config,
            layer_id,
            quant_config=quant_config,
            prefix=f"model.layers.{layer_id}.self_attn",
        )
        model.model.layers[layer_id].self_attn = attn
        return model, attn

    def test_fp8_shards_and_scales_load_at_rank_offsets(self):
        model, attn = self.build(1, fp8_dynamic())
        packed = attn.qkv_gate_proj
        self.assertIsInstance(packed.quant_method, CompressedTensorsLinearMethod)
        self.assertEqual(packed.weight.dtype, torch.float8_e4m3fn)

        rows = {
            "q_proj": HEADS * HEAD_DIM,
            "k_proj": KV_HEADS * HEAD_DIM,
            "v_proj": KV_HEADS * HEAD_DIM,
            "gate_proj": HEADS * HEAD_DIM,
        }
        weights, expected_weight, expected_scale = [], [], []
        for offset, (name, n) in enumerate(rows.items()):
            weight = checkpoint_values(n, offset).to(torch.float8_e4m3fn)
            # The checkpoint stores per-channel scales as [rows, 1] BF16.
            scale = (torch.arange(n) + 16 * offset + 1).to(torch.bfloat16)[:, None]
            prefix = f"model.layers.1.self_attn.{name}"
            weights += [(f"{prefix}.weight", weight), (f"{prefix}.weight_scale", scale)]
            local = n // TP_SIZE
            expected_weight.append(weight[TP_RANK * local : (TP_RANK + 1) * local])
            expected_scale.append(scale[TP_RANK * local : (TP_RANK + 1) * local])
        model.load_weights(weights)

        self.assertTrue(
            torch.equal(
                packed.weight.view(torch.uint8),
                torch.cat(expected_weight).view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(packed.weight_scale.float(), torch.cat(expected_scale).float())
        )

    def test_unquantized_attention_shards_stay_split(self):
        def ignored(layer, projections):
            return [f"model.layers.{layer}.self_attn.{p}" for p in projections]

        quant_config = fp8_dynamic(
            ignored(1, ("qkv_proj",))
            + ignored(2, ("gate_proj",))
            + ignored(3, ("q_proj", "k_proj", "v_proj", "gate_proj"))
        )
        for layer_id, qkv_method, gate_method in (
            (1, UnquantizedLinearMethod, CompressedTensorsLinearMethod),
            (2, CompressedTensorsLinearMethod, UnquantizedLinearMethod),
            (3, UnquantizedLinearMethod, UnquantizedLinearMethod),
        ):
            with self.subTest(layer_id=layer_id):
                _, attn = self.build(layer_id, quant_config)
                self.assertFalse(attn.use_packed_projection)
                self.assertIsInstance(attn.qkv_proj.quant_method, qkv_method)
                self.assertIsInstance(attn.gate_proj.quant_method, gate_method)

        nvfp4 = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.1.self_attn*"],
            packed_modules_mapping=BeamForCausalLM.packed_modules_mapping,
        )
        _, separate = self.build(1, nvfp4)
        self.assertFalse(separate.use_packed_projection)
        self.assertIsInstance(separate.qkv_proj.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
