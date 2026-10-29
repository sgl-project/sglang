"""SM90 component CI for Humming FP8 DeepEP dispatch and fused SiLU quant."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.moe.dsv4 import silu_and_mul_masked_post_quant
from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute_with_scale
from sglang.srt.layers.moe.moe_runner import humming as humming_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _silu_reference(gate_up: torch.Tensor, swiglu_limit: float | None) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is not None:
        gate = gate.clamp_max(swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    gate = gate.float()
    up = up.float()
    return gate * torch.sigmoid(gate) * up


class TestHummingFp8Dispatch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() != 90:
            raise unittest.SkipTest("Humming FP8 dispatch requires SM90")

    def test_fp8_dispatch_preserves_group_scales(self):
        from sglang.srt.layers.quantization import humming_utils

        generator = torch.Generator(device="cuda").manual_seed(0)
        hidden_states = torch.randn(
            5,
            256,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
        hidden_states_scale = torch.arange(10, device="cuda", dtype=torch.float32).view(
            5, 2
        )
        topk_ids = torch.tensor(
            [[2, 0], [1, -1], [0, 2], [1, 0], [-1, 2]],
            device="cuda",
            dtype=torch.int32,
        )

        output, output_scale, src2dst, expert_offsets = moe_permute_with_scale(
            inputs=hidden_states,
            input_scale=hidden_states_scale,
            topk_ids=topk_ids,
            num_experts=3,
            is_ep=True,
        )

        for token in range(topk_ids.size(0)):
            for slot in range(topk_ids.size(1)):
                source_index = token * topk_ids.size(1) + slot
                destination_index = int(src2dst[source_index])
                if topk_ids[token, slot] < 0:
                    self.assertLess(destination_index, 0)
                    continue
                self.assertTrue(
                    torch.equal(
                        output[destination_index].view(torch.uint8),
                        hidden_states[token].view(torch.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output_scale[destination_index], hidden_states_scale[token]
                    )
                )

        self.assertTrue(
            torch.equal(
                expert_offsets,
                torch.tensor([0, 3, 5, 8], device="cuda", dtype=torch.int32),
            )
        )

        dispatcher = SimpleNamespace(
            quant_config={"existing_key": "preserved"},
            set_quant_config=MagicMock(),
        )
        layer = SimpleNamespace(
            dispatcher=dispatcher,
            humming_metas={
                "w13": SimpleNamespace(
                    a_dtype=humming_runner.dtypes.float8e4m3,
                    input_scale_group_size=128,
                )
            },
        )
        with (
            patch.object(
                humming_utils,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_deepep=lambda: True),
            ),
            patch.object(
                humming_utils,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(deepep_dispatcher_output_dtype="auto")
                ),
            ),
            patch.object(
                humming_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
        ):
            self.assertTrue(humming_utils.configure_humming_deepep_dispatch(layer))

        self.assertTrue(layer._humming_uses_deepep_fp8_dispatch)
        dispatcher.set_quant_config.assert_called_once_with(
            {"existing_key": "preserved", "dispatcher_output_dtype": "fp8"}
        )
        humming_runner._validate_deepep_dispatch_input(output, output_scale, layer)

    def test_silu_fused_masked_quant(self):
        num_experts = 8
        max_tokens = 17
        intermediate = 768
        num_groups = intermediate // 128

        generator = torch.Generator(device="cuda").manual_seed(1)
        gate_up = (
            torch.randn(
                num_experts,
                max_tokens,
                2 * intermediate,
                generator=generator,
                device="cuda",
            )
            * 3.0
        ).to(torch.bfloat16)
        expert_num_tokens = torch.tensor(
            [17, 0, 7, 10, 17, 1, 8, 5], device="cuda", dtype=torch.int32
        )

        for swiglu_limit in (None, 10.0):
            with self.subTest(swiglu_limit=swiglu_limit):
                runner = humming_runner.HummingRunnerCore(
                    MoeRunnerConfig(
                        num_experts=num_experts,
                        num_local_experts=num_experts,
                        intermediate_size_per_partition=intermediate,
                        top_k=8,
                        activation="silu",
                        is_gated=True,
                        swiglu_limit=swiglu_limit,
                        gate_up_interleaved=False,
                    )
                )
                runner.layer = SimpleNamespace(
                    humming_metas={
                        "w2": SimpleNamespace(
                            a_dtype=humming_runner.dtypes.float8e4m3,
                            input_scale_group_size=128,
                        )
                    }
                )
                activation_output = torch.full(
                    (num_experts * max_tokens, intermediate),
                    float("nan"),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                buffers = {
                    "gate_up_output": gate_up.flatten(0, 1),
                    "activation_output": activation_output,
                }

                def run_poisoned_kernel(*args, **kwargs):
                    args[1].view(torch.uint8).fill_(0x7F)
                    args[2].fill_(float("nan"))
                    return silu_and_mul_masked_post_quant(*args, **kwargs)

                with (
                    patch.object(
                        runner,
                        "apply_activation",
                        side_effect=AssertionError("SiLU fused path was not selected"),
                    ) as fallback_activation,
                    patch(
                        "sglang.kernels.ops.moe.dsv4.silu_and_mul_masked_post_quant",
                        side_effect=run_poisoned_kernel,
                    ) as fused_silu,
                ):
                    down_input, down_scale = runner._grouped_masked_act_quant(
                        gate_up, expert_num_tokens, buffers
                    )

                fused_silu.assert_called_once()
                fallback_activation.assert_not_called()
                call = fused_silu.call_args
                self.assertIs(call.args[0], gate_up)
                self.assertIs(call.args[4], expert_num_tokens)
                self.assertEqual(call.args[3], 128)
                self.assertEqual(call.kwargs["topk"], 8)
                self.assertEqual(call.kwargs["swiglu_limit"], swiglu_limit)
                self.assertFalse(call.kwargs["scale_ue8m0"])
                self.assertEqual(call.args[1].data_ptr(), down_input.data_ptr())
                self.assertEqual(call.args[2].data_ptr(), down_scale.data_ptr())
                self.assertTrue(bool(activation_output.isnan().all()))

                actual = down_input.view(num_experts, max_tokens, intermediate)
                actual_scale = down_scale.view(num_experts, max_tokens, num_groups)
                reference = _silu_reference(gate_up, swiglu_limit)
                grouped_reference = reference.view(
                    num_experts, max_tokens, num_groups, 128
                )
                expected_scale = (
                    grouped_reference.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
                )

                for expert in range(num_experts):
                    valid = int(expert_num_tokens[expert])
                    if valid:
                        torch.testing.assert_close(
                            actual_scale[expert, :valid],
                            expected_scale[expert, :valid],
                            rtol=5e-3,
                            atol=1e-6,
                        )
                        expanded_scale = actual_scale[expert, :valid].repeat_interleave(
                            128, dim=-1
                        )
                        dequantized = actual[expert, :valid].float() * expanded_scale
                        self.assertTrue(
                            bool(
                                (
                                    (dequantized - reference[expert, :valid]).abs()
                                    <= expanded_scale * 17.0 + 1e-4
                                ).all()
                            )
                        )

                    if valid < max_tokens:
                        self.assertTrue(
                            bool(
                                (actual[expert, valid:].view(torch.uint8) == 0x7F).all()
                            )
                        )
                        self.assertTrue(
                            bool(actual_scale[expert, valid:].isnan().all())
                        )


class TestHummingInactiveRoutes(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_indexed_finalize_skips_poisoned_routes_without_ep(self):
        from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum

        ids = torch.tensor(
            [[0, 1], [1, -1], [-1, -1], [1, 0]], device="cuda", dtype=torch.int32
        )
        weights = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2]], device="cuda"
        )
        # A legal leading storage record makes a negative map lookup observable
        # without making the test depend on an out-of-allocation device read.
        map_storage = torch.tensor([0, 0, -1], device="cuda", dtype=torch.int32)
        for dtype in (torch.bfloat16, torch.float16):
            for expert_map in (None, map_storage[1:]):
                for poison_weights in (False, True):
                    with self.subTest(
                        dtype=dtype,
                        mapped=expert_map is not None,
                        poison_weights=poison_weights,
                    ):
                        valid = ids >= 0
                        if expert_map is not None:
                            valid &= expert_map[ids.clamp_min(0)] >= 0
                        values = (
                            torch.arange(4 * 2 * 128, device="cuda")
                            .reshape(4, 2, 128)
                            .remainder(11)
                            .to(dtype)
                        )
                        values[~valid] = float("nan")
                        route_weights = weights.clone()
                        if poison_weights:
                            route_weights[~valid] = float("nan")
                        expected = (
                            (
                                torch.where(valid.unsqueeze(-1), values.float(), 0)
                                * torch.where(valid, route_weights, 0).unsqueeze(-1)
                            )
                            .sum(dim=1)
                            .to(dtype)
                        )
                        actual = moe_fused_mul_sum(
                            values,
                            route_weights,
                            topk_ids=ids,
                            expert_map=expert_map,
                            is_ep=False,
                        )
                        self.assertTrue(torch.isfinite(actual).all().item())
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)

    def test_grouped_permutation_skips_inactive_routes_without_ep(self):
        from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute, moe_unpermute

        ids = torch.tensor(
            [[0, 1], [1, -1], [-1, -1], [1, 0]], device="cuda", dtype=torch.int32
        )
        weights = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2]], device="cuda"
        )
        inputs = (
            torch.arange(4 * 128, device="cuda")
            .reshape(4, 128)
            .remainder(7)
            .to(torch.bfloat16)
        )
        output = torch.full(
            (ids.numel(), 128), float("nan"), device="cuda", dtype=inputs.dtype
        )
        permuted, src2dst, offsets = moe_permute(
            inputs, ids, num_experts=2, is_ep=False, outputs=output
        )
        self.assertTrue((src2dst.view_as(ids)[ids < 0] < 0).all().item())
        torch.testing.assert_close(
            offsets, torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
        )
        down = torch.full_like(permuted, float("nan"))
        reference = torch.zeros_like(inputs)
        for expert in range(2):
            start, end = int(offsets[expert]), int(offsets[expert + 1])
            down[start:end] = permuted[start:end] * (expert + 1)
        for token in range(ids.shape[0]):
            for slot in range(ids.shape[1]):
                expert = int(ids[token, slot])
                if expert >= 0:
                    reference[token] += (
                        inputs[token] * (expert + 1) * weights[token, slot]
                    )
        actual = moe_unpermute(down, src2dst, ids, weights)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
        self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)


if __name__ == "__main__":
    unittest.main()
