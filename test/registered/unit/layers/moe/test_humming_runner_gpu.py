"""SM90 numerical regressions for the Humming two-GEMM MoE runner."""

import json
import math
import unittest

import torch

from sglang.srt.layers.moe.moe_runner import humming as humming_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestHummingRunnerGPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() != 90:
            raise unittest.SkipTest("Humming runner GPU regression requires SM90")

    def test_indexed_two_gemms_with_different_m_tiles(self):
        self._check_indexed_two_gemms(static_scales=False)

    def test_indexed_two_gemms_with_static_scales(self):
        # Secondary input scales need block-scaled MMA (SM100+), so SM90
        # covers the static per-tensor scales of both GEMMs.
        self._check_indexed_two_gemms(static_scales=True)

    def _check_indexed_two_gemms(self, static_scales):
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.utils.weight import dequantize_weight

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 512, 256
        num_tokens, top_k = 128, 2
        layer = torch.nn.Module()
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate_size
        layer.params_dtype = torch.bfloat16
        layer.register_buffer(
            "locks", torch.zeros(2048, dtype=torch.int32, device="cuda")
        )
        weight_schema = HummingWeightSchema(
            b_dtype="float4e2m1", bs_dtype="float8e8m0", weight_scale_group_size=32
        )
        input_schema = HummingInputSchema(
            a_dtype="float8e4m3",
            input_scale_group_size=128,
            input_scale_dtype="float32",
        )
        reference_weights = {}
        for sublayer, shape_n, shape_k in (
            ("w13", intermediate_size * 2, hidden_size),
            ("w2", hidden_size, intermediate_size),
        ):
            sublayer_input_schema = input_schema
            if static_scales:
                sublayer_input_schema = HummingInputSchema(
                    a_dtype="float8e4m3",
                    input_quant_mode="static_tensor",
                    input_scale_dtype="float32",
                )
                layer.register_buffer(
                    f"{sublayer}_input_scale",
                    torch.tensor([0.02 if sublayer == "w13" else 0.125], device="cuda"),
                )
            weights = torch.randn(
                num_experts, shape_n, shape_k, device="cuda", dtype=torch.bfloat16
            )
            weights /= math.sqrt(shape_k)
            tensors = weight_schema.quant_tensor(weights, weight_schema, torch.bfloat16)
            reference_weights[sublayer] = dequantize_weight(
                tensors["weight"],
                tensors.get("weight_scale"),
                tensors.get("zero_point"),
                tensors.get("weight_scale_2"),
                weight_schema.b_dtype,
                packed=True,
            )
            for name, tensor in tensors.items():
                layer.register_parameter(
                    f"{sublayer}_{name}",
                    torch.nn.Parameter(tensor, requires_grad=False),
                )
            humming_runner.HummingMethod.prepare_layer_meta(
                layer=layer,
                shape_n=shape_n,
                shape_k=shape_k,
                weight_schema=weight_schema,
                input_schema=sublayer_input_schema,
                num_experts=num_experts,
                torch_dtype=torch.bfloat16,
                sublayer_name=sublayer,
                tensors=tensors,
            )
            humming_runner.HummingMethod.transform_humming_layer(
                layer, sublayer_name=sublayer
            )

        runner = humming_runner.HummingRunnerCore(
            MoeRunnerConfig(
                num_experts=num_experts,
                num_local_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size,
                top_k=top_k,
                params_dtype=torch.bfloat16,
                activation="silu",
                gate_up_interleaved=False,
            )
        )
        runner.layer = layer
        compute_config = {"gemm_type": "indexed", "use_f16_accum": False}
        configs = {
            "compute_config": compute_config,
            "compute_config_str": json.dumps(compute_config),
        }
        # Fix distinct legal M tiles: heuristics may choose the same tile on H100.
        for sublayer, block_m in (("w13", 32), ("w2", 48)):
            selected = {
                "mma_type": "wgmma",
                "block_shape": (block_m, 128, 128),
                "warp_shape": (block_m, 32, 128),
                "num_stages": 3,
                "use_warp_spec": True,
                "use_tma": True,
                "use_tma_a": False,
                "use_tma_as": False,
                "use_tma_as2": False,
                "use_tma_c": False,
                # Direct output gives deterministic eager/graph comparisons.
                "use_stream_k": False,
            }
            tuning = [[0, 1 << 30, selected]]
            configs[f"{sublayer}_tuning_config"] = tuning
            configs[f"{sublayer}_tuning_config_str"] = json.dumps(tuning)
        runner.humming_gemm_configs["indexed"] = configs
        inputs = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        topk_ids = torch.randint(
            num_experts, (num_tokens, top_k), device="cuda", dtype=torch.int32
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda"), dim=-1
        )

        def quantize_reference(values, sublayer):
            if static_scales:
                scale = getattr(layer, f"{sublayer}_input_scale")
                return (values.float() / scale).clamp(-448, 448).to(
                    torch.float8_e4m3fn
                ).float() * scale
            grouped = values.float().reshape(*values.shape[:-1], -1, 128)
            scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448.0
            return (
                (grouped / scales).to(torch.float8_e4m3fn).float() * scales
            ).reshape_as(values)

        quantized_inputs = quantize_reference(inputs, "w13")
        reference = torch.zeros_like(inputs, dtype=torch.float32)
        for expert in range(num_experts):
            tokens, slots = torch.where(topk_ids == expert)
            gate_up = (
                quantized_inputs[tokens] @ reference_weights["w13"][expert].T
            ).to(torch.bfloat16)
            gate, up = gate_up.chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                torch.bfloat16
            )
            down_input = quantize_reference(activated, "w2")
            down_output = (down_input @ reference_weights["w2"][expert].T).to(
                torch.bfloat16
            )
            reference.index_add_(
                0, tokens, down_output.float() * topk_weights[tokens, slots, None]
            )
        reference = reference.to(torch.bfloat16)

        actual = runner._run_indexed_gemm(inputs, topk_weights, topk_ids)
        self.assertTrue(torch.isfinite(actual).all().item())
        relative_error = (
            actual.float() - reference.float()
        ).norm() / reference.float().norm()
        self.assertLess(relative_error.item(), 0.035)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured = runner._run_indexed_gemm(inputs, topk_weights, topk_ids)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
