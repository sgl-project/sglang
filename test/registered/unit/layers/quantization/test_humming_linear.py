"""Exercise Humming's online loader, schema conversion, and dense GPU forward."""

import math
import unittest

import torch
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.utils.weight import dequantize_weight

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.humming import (
    HummingLayerQuantizationConfig,
    HummingLinearMethod,
    HummingMoEMethod,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def apply_reference_hadamard(inputs, block_size):
    # Independent normalized Walsh-Hadamard transform on each input block.
    matrix = torch.ones(1, 1, device=inputs.device)
    while matrix.shape[0] < block_size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    transformed = inputs.float().reshape(-1, block_size) @ matrix
    return (transformed / math.sqrt(block_size)).reshape_as(inputs).to(inputs.dtype)


class TestHummingLinear(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise unittest.SkipTest("Humming adapter GPU regression requires SM90")

    def test_online_loading_and_forward(self):
        cases = (
            ("uint4", None, 128, None, 0, "bf16"),
            ("float4e2m1", "float8e4m3", 16, "tensor", 0, "bf16"),
            ("float4e2m1", "float8e4m3", 16, "channel", 0, "bf16"),
            ("uint4", None, 128, None, 32, "bf16"),
            ("float8e4m3", None, 0, None, 0, "fp8_static"),
            ("uint4", None, 128, None, 0, "int8"),
        )
        for (
            weight_dtype,
            scale_dtype,
            group_size,
            scale_2,
            hadamard,
            input_mode,
        ) in cases:
            with self.subTest(
                weight_dtype=weight_dtype,
                scale_2=scale_2,
                hadamard=hadamard,
                input_mode=input_mode,
            ):
                torch.manual_seed(42)
                weight_schema = HummingWeightSchema(
                    b_dtype=weight_dtype,
                    bs_dtype=scale_dtype,
                    weight_scale_group_size=group_size,
                    weight_scale_2_type=scale_2,
                    hadamard_block_size=hadamard,
                )
                if input_mode == "fp8_static":
                    input_schema = HummingInputSchema(
                        a_dtype="float8e4m3", input_quant_mode="static_tensor"
                    )
                elif input_mode == "int8":
                    input_schema = HummingInputSchema(a_dtype="int8")
                else:
                    input_schema = HummingInputSchema()
                method = HummingLinearMethod(
                    HummingLayerQuantizationConfig(
                        weight_schema=weight_schema,
                        input_schema=input_schema,
                        is_online_quant=True,
                    )
                )
                layer = torch.nn.Module()
                layer.with_bias = False
                with torch.device("cuda"):
                    method.create_weights(
                        layer,
                        input_size_per_partition=512,
                        output_partition_sizes=[256],
                        input_size=512,
                        output_size=256,
                        params_dtype=torch.bfloat16,
                    )
                    inputs = torch.randn(17, 512, dtype=torch.bfloat16)

                # Checkpoint loaders yield CPU tensors even when parameters are on CUDA.
                weights = torch.randn(256, 512, dtype=torch.bfloat16) / math.sqrt(512)
                layer.weight.weight_loader(layer.weight, weights)
                reference_weight = dequantize_weight(
                    layer.weight,
                    getattr(layer, "weight_scale", None),
                    getattr(layer, "zero_point", None),
                    getattr(layer, "weight_scale_2", None),
                    weight_schema.b_dtype,
                    packed=True,
                )
                reference_input = inputs.float()
                if hadamard:
                    reference_input = apply_reference_hadamard(inputs, hadamard).float()
                if input_mode == "fp8_static":
                    scale = torch.tensor([0.02], device="cuda")
                    layer.input_scale.weight_loader(layer.input_scale, scale)
                    reference_input = (reference_input / scale).to(
                        torch.float8_e4m3fn
                    ).float() * scale
                elif input_mode == "int8":
                    scale = reference_input.abs().amax(-1, keepdim=True) / 127.0
                    reference_input = (reference_input / scale).round().clamp(
                        -128, 127
                    ) * scale

                reference = (reference_input @ reference_weight.T).to(torch.bfloat16)
                method.process_weights_after_loading(layer)
                actual = method.apply(layer, inputs)
                self.assertTrue(torch.isfinite(actual).all().item())
                relative_error = (
                    actual.float() - reference.float()
                ).norm() / reference.float().norm()
                self.assertLess(relative_error.item(), 0.015)

    def test_moe_online_hadamard_loading_from_cpu(self):
        for parameter_device in ("cuda", "cpu"):
            with self.subTest(parameter_device=parameter_device):
                torch.manual_seed(42)
                schema = HummingWeightSchema(
                    b_dtype="uint4",
                    weight_scale_group_size=128,
                    hadamard_block_size=32,
                )
                method = HummingMoEMethod(
                    HummingLayerQuantizationConfig(
                        weight_schema=schema, is_online_quant=True
                    )
                )
                # Use the real expert checkpoint loader without distributed initialization.
                layer = FusedMoE.__new__(FusedMoE)
                torch.nn.Module.__init__(layer)
                layer.quant_method = method
                layer.quant_config = method.quant_config
                layer.moe_runner_config = MoeRunnerConfig(is_gated=True)
                layer.scheme = None
                layer._has_fused_shared = False
                layer.uses_expert_location = False
                layer._expert_storage_rank = 0
                layer._num_local_routed = 2
                layer.use_flashinfer_trtllm_moe = False
                layer.use_triton_kernels = False
                layer.use_presharded_weights = False
                layer.moe_tp_size = 1
                layer.moe_tp_rank = 0
                with torch.device(parameter_device):
                    method.create_weights(
                        layer,
                        num_experts=2,
                        hidden_size=512,
                        intermediate_size_per_partition=256,
                        params_dtype=torch.bfloat16,
                        weight_loader=layer.weight_loader,
                    )

                checkpoint_weights = {
                    "w1": torch.randn(2, 256, 512, dtype=torch.bfloat16),
                    "w3": torch.randn(2, 256, 512, dtype=torch.bfloat16),
                    "w2": torch.randn(2, 512, 256, dtype=torch.bfloat16),
                }
                for shard_id, weights in checkpoint_weights.items():
                    param = layer.w2_weight if shard_id == "w2" else layer.w13_weight
                    for expert_id, loaded_weight in enumerate(weights):
                        self.assertEqual(loaded_weight.device.type, "cpu")
                        param.weight_loader(
                            param,
                            loaded_weight,
                            weight_name=f"{shard_id}_weight",
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )

                layer.cuda()
                references = {
                    "w13": torch.cat(
                        (checkpoint_weights["w1"], checkpoint_weights["w3"]), dim=1
                    ),
                    "w2": checkpoint_weights["w2"],
                }
                for sublayer, weights in references.items():
                    actual = dequantize_weight(
                        getattr(layer, f"{sublayer}_weight"),
                        getattr(layer, f"{sublayer}_weight_scale"),
                        None,
                        None,
                        schema.b_dtype,
                        packed=True,
                    )
                    reference = apply_reference_hadamard(weights, 32).cuda().float()
                    relative_error = (actual - reference).norm() / reference.norm()
                    self.assertLess(relative_error.item(), 0.15)


if __name__ == "__main__":
    unittest.main()
