"""Compressed-tensors FP8 MoE must size w13 by the layer's gating.

Non-gated experts (NemotronH relu2) carry only up_proj in w13. Sizing it as
gate + up doubles the first GEMM and the w13 memory, with the extra half
zero-filled by the loader.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS, HIDDEN, INTERMEDIATE, BLOCK = 2, 256, 384, 128


def _quant_config(strategy: str) -> dict:
    block = strategy == "block"
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": strategy,
                    "dynamic": False,
                    "block_structure": [BLOCK, BLOCK] if block else None,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": {"tensor": "tensor", "channel": "token"}.get(
                        strategy, "group"
                    ),
                    "dynamic": strategy != "tensor",
                    "group_size": BLOCK if block else None,
                },
            }
        },
        "ignore": ["lm_head"],
    }


def _scale_shape(strategy: str, rows: int, cols: int) -> tuple:
    if strategy == "tensor":
        return (1,)
    if strategy == "channel":
        return (rows, 1)
    return (rows // BLOCK, cols // BLOCK)


class TestCompressedTensorsFp8MoENonGated(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
        from sglang.srt.runtime_context import get_context, get_flags, get_parallel

        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            )
        )
        self.enterContext(patch.object(fused_moe_layer, "_is_cpu", False))

    def _build(self, strategy: str, is_gated: bool):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=1,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            layer_id=0,
            params_dtype=torch.bfloat16,
            quant_config=CompressedTensorsConfig.from_config(_quant_config(strategy)),
            prefix="model.layers.0.mixer.experts",
            is_gated=is_gated,
            activation="silu" if is_gated else "relu2",
        )
        self.assertIs(type(layer.scheme), CompressedTensorsW8A8Fp8MoE)
        return layer

    def _load(self, layer, strategy: str, shard_id: str, expert_id: int, seed: int):
        rows, cols = (
            (HIDDEN, INTERMEDIATE) if shard_id == "w2" else (INTERMEDIATE, HIDDEN)
        )
        gen = torch.Generator().manual_seed(seed)
        weight = (torch.randn(rows, cols, generator=gen) * 16).to(torch.float8_e4m3fn)
        scale = torch.rand(_scale_shape(strategy, rows, cols), generator=gen) + 0.5
        prefix = "w2" if shard_id == "w2" else "w13"
        for suffix, value in (("weight", weight), ("weight_scale", scale)):
            name = f"{prefix}_{suffix}"
            layer.weight_loader(
                getattr(layer, name),
                value,
                name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
        return weight, scale

    def test_non_gated_w13_holds_only_up_proj(self):
        for strategy in ("tensor", "channel", "block"):
            with self.subTest(strategy=strategy):
                layer = self._build(strategy, is_gated=False)
                self.assertEqual(
                    tuple(layer.w13_weight.shape), (NUM_EXPERTS, INTERMEDIATE, HIDDEN)
                )
                for expert_id in range(NUM_EXPERTS):
                    up, up_scale = self._load(layer, strategy, "w1", expert_id, 1)
                    self._load(layer, strategy, "w2", expert_id, 2)
                    torch.testing.assert_close(
                        layer.w13_weight[expert_id].float(), up.float(), rtol=0, atol=0
                    )
                    loaded_scale = layer.w13_weight_scale[expert_id]
                    if strategy == "tensor":
                        self.assertEqual(tuple(loaded_scale.shape), (1,))
                    torch.testing.assert_close(
                        loaded_scale.reshape(up_scale.shape), up_scale
                    )

    def test_gated_w13_holds_gate_and_up(self):
        for strategy in ("tensor", "channel", "block"):
            with self.subTest(strategy=strategy):
                layer = self._build(strategy, is_gated=True)
                self.assertEqual(
                    tuple(layer.w13_weight.shape),
                    (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
                )
                gate, _ = self._load(layer, strategy, "w1", 0, 3)
                up, _ = self._load(layer, strategy, "w3", 0, 4)
                torch.testing.assert_close(
                    layer.w13_weight[0].float(),
                    torch.cat((gate, up)).float(),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
