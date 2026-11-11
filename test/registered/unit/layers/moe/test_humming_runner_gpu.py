"""GPU numerical regressions for the Humming two-GEMM MoE runner."""

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
register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestHummingRunnerGPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_indexed_two_gemms_with_different_m_tiles(self):
        if get_device_sm() != 90:
            self.skipTest("Distinct WGMMA M tiles require SM90")
        self._check_indexed_two_gemms(static_scales=False)

    def test_indexed_two_gemms_with_static_scales(self):
        # Secondary input scales need block-scaled MMA (SM100+), so SM90
        # covers the static per-tensor scales of both GEMMs.
        if get_device_sm() != 90:
            self.skipTest("Static WGMMA regression targets SM90")
        self._check_indexed_two_gemms(static_scales=True)

    def test_indexed_two_gemms_with_secondary_scales(self):
        if get_device_sm() // 10 not in (10, 11):
            self.skipTest("Secondary-scale UMMA regression requires SM100/SM110 family")
        self._check_indexed_two_gemms(static_scales=False, secondary_scales=True)

    def test_standard_ep_with_default_tuning(self):
        if get_device_sm() != 90:
            self.skipTest("MXFP4 EP regression targets SM90")
        cases = [
            ("indexed", 0, True),
            ("indexed", 1, True),
            ("grouped_contiguous", 0, True),
            ("grouped_contiguous", 1, True),
            ("indexed", 0, False),
            ("grouped_contiguous", 1, False),
        ]
        for gemm_type, shared, fp8 in cases:
            with self.subTest(gemm_type=gemm_type, shared=shared, fp8=fp8):
                self._check_indexed_two_gemms(
                    static_scales=False,
                    gemm_type=gemm_type,
                    ep_size=8,
                    shared_experts=shared,
                    fp8_activations=fp8,
                )

    def _check_indexed_two_gemms(
        self,
        static_scales,
        secondary_scales=False,
        gemm_type="indexed",
        ep_size=1,
        shared_experts=0,
        fp8_activations=True,
    ):
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.utils.weight import dequantize_weight

        torch.manual_seed(42)
        num_experts, hidden_size, intermediate_size = 4, 512, 256
        num_tokens, top_k = 128, 2
        local_routed_experts = num_experts - shared_experts
        global_routed_experts = local_routed_experts * ep_size
        global_experts = global_routed_experts + shared_experts
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
            input_scale_group_size=0 if ep_size > 1 else 128,
            input_scale_dtype="float32",
        )
        if not fp8_activations:
            input_schema = HummingInputSchema(a_dtype="bfloat16")
        reference_weights = {}
        for sublayer, shape_n, shape_k in (
            ("w13", intermediate_size * 2, hidden_size),
            ("w2", hidden_size, intermediate_size),
        ):
            sublayer_input_schema = input_schema
            if secondary_scales:
                sublayer_input_schema = HummingInputSchema(
                    a_dtype="float8e4m3",
                    input_quant_mode="static_tensor_dynamic_group",
                    input_scale_group_size=32,
                    input_scale_dtype="float8e8m0",
                )
                layer.register_buffer(
                    f"{sublayer}_input_scale_2",
                    torch.tensor([0.7 if sublayer == "w13" else 1.3], device="cuda"),
                )
            elif static_scales:
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
                num_experts=global_experts,
                num_local_experts=num_experts,
                num_fused_shared_experts=shared_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size,
                top_k=top_k,
                params_dtype=torch.bfloat16,
                activation="silu",
                gate_up_interleaved=False,
            )
        )
        runner.layer = layer
        compute_config = {"gemm_type": gemm_type, "use_f16_accum": False}
        configs = {
            "compute_config": compute_config,
            "compute_config_str": json.dumps(compute_config),
        }
        if secondary_scales or ep_size > 1:
            for sublayer in ("w13", "w2"):
                meta = layer.humming_metas[sublayer]
                if secondary_scales:
                    self.assertTrue(meta.use_block_scaled_mma)
                    self.assertTrue(meta.has_input_scale_2)
                tuning = [
                    [
                        lo,
                        hi,
                        {**selected, "use_stream_k": False}
                        if secondary_scales
                        else selected,
                    ]
                    for lo, hi, selected in humming_runner.HummingMethod.get_default_tuning_configs(
                        layer, gemm_type=gemm_type, sublayer_name=sublayer
                    )
                ]
                configs[f"{sublayer}_tuning_config"] = tuning
                configs[f"{sublayer}_tuning_config_str"] = json.dumps(tuning)
        else:
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
        runner.humming_gemm_configs[gemm_type] = configs
        inputs = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        topk_ids = torch.randint(
            global_routed_experts,
            (num_tokens, top_k - shared_experts),
            device="cuda",
            dtype=torch.int32,
        )
        topk_ids = torch.where(topk_ids < local_routed_experts, topk_ids, -1)
        if shared_experts:
            shared_ids = torch.arange(
                local_routed_experts, num_experts, device="cuda", dtype=torch.int32
            )
            topk_ids = torch.cat((topk_ids, shared_ids.expand(num_tokens, -1)), dim=1)
        topk_weights = torch.softmax(
            torch.randn(num_tokens, top_k, device="cuda"), dim=-1
        )

        def quantize_reference(values, sublayer):
            if not fp8_activations:
                return values.float()
            if static_scales:
                scale = getattr(layer, f"{sublayer}_input_scale")
                return (values.float() / scale).clamp(-448, 448).to(
                    torch.float8_e4m3fn
                ).float() * scale
            if secondary_scales:
                grouped = values.float().reshape(*values.shape[:-1], -1, 32)
                secondary = getattr(layer, f"{sublayer}_input_scale_2")
                raw_scale = (
                    grouped.abs().amax(dim=-1, keepdim=True) / 448.0
                ).clamp_min(1e-30)
                # E8M0 group scales round up to the next power of two.
                scale = (
                    torch.exp2(torch.ceil(torch.log2(raw_scale / secondary)))
                    * secondary
                )
                return (
                    (grouped / scale).to(torch.float8_e4m3fn).float() * scale
                ).reshape_as(values)
            width = values.shape[-1] if ep_size > 1 else 128
            grouped = values.float().reshape(*values.shape[:-1], -1, width)
            scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448.0
            return (
                (grouped / scales).to(torch.float8_e4m3fn).float() * scales
            ).reshape_as(values)

        if secondary_scales:
            for sublayer, width in (("w13", hidden_size), ("w2", intermediate_size)):
                probe = inputs[:, :width].contiguous()
                _, group_scale, secondary = runner._prepare_input(
                    probe, sublayer_name=sublayer
                )
                expected_secondary = getattr(layer, f"{sublayer}_input_scale_2")
                torch.testing.assert_close(secondary, expected_secondary)
                raw_scale = (
                    probe.float().reshape(num_tokens, -1, 32).abs().amax(dim=-1) / 448.0
                ).clamp_min(1e-30)
                expected_exponents = (
                    torch.ceil(torch.log2(raw_scale / expected_secondary)) + 127
                ).to(torch.uint8)
                torch.testing.assert_close(
                    group_scale.view(torch.uint8).reshape_as(expected_exponents),
                    expected_exponents,
                    rtol=0,
                    atol=0,
                )

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

        def run():
            runner_input = humming_runner.HummingRunnerInput(
                hidden_states=inputs,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                gemm_type=humming_runner.HummingGemmType(gemm_type),
                is_standard_dispatch=True,
            )
            return runner.run(
                runner_input, humming_runner.HummingMoeQuantInfo(layer), {}
            ).hidden_states

        actual = run()
        self.assertTrue(torch.isfinite(actual).all().item())
        relative_error = (
            actual.float() - reference.float()
        ).norm() / reference.float().norm()
        self.assertLess(relative_error.item(), 0.035)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        if ep_size > 1:
            self.assertTrue(torch.isfinite(captured).all().item())
            graph_error = (
                captured.float() - reference.float()
            ).norm() / reference.float().norm()
            self.assertLess(graph_error.item(), 0.035)
        else:
            torch.testing.assert_close(captured, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
