"""CPU regression tests for ModelOpt NVFP4 gated-MoE TP padding.

The regression test drives the real ``process_weights_after_loading`` path used
by ``flashinfer_cutlass``. It keeps Gemma-4's relevant dimensions exactly as
observed in the TP=2 failure: 704 fused rows, 352 channels per rank, hidden size
2816, and a 22-column w2 block scale. Only the expert count is reduced to keep
CPU CI small.

The GPU-only block-scale swizzle is replaced with a CPU test double that keeps
its padding contract. This makes the test fail at the original gated-padding
assertion if per-half padding is removed or moved after swizzling, while marker
values catch a naive tail pad that shifts the two logical w13 projections.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization import modelopt_quant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GROUP_SIZE = 16
GEMMA4_MOE_INTERMEDIATE_SIZE = 704
TP_SIZE = 2
TP_INTERMEDIATE_SIZE = GEMMA4_MOE_INTERMEDIATE_SIZE // TP_SIZE
PADDED_INTERMEDIATE_SIZE = 384
GEMMA4_HIDDEN_SIZE = 2816
NUM_TEST_EXPERTS = 1


def _parameter(tensor: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(tensor, requires_grad=False)


def _make_gemma4_tp2_layer():
    projection_shape = (
        NUM_TEST_EXPERTS,
        TP_INTERMEDIATE_SIZE,
        GEMMA4_HIDDEN_SIZE // 2,
    )
    projection_scale_shape = (
        NUM_TEST_EXPERTS,
        TP_INTERMEDIATE_SIZE,
        GEMMA4_HIDDEN_SIZE // GROUP_SIZE,
    )

    layer = torch.nn.Module()
    layer.w13_weight = _parameter(
        torch.cat(
            (
                torch.full(projection_shape, 0x11, dtype=torch.uint8),
                torch.full(projection_shape, 0x77, dtype=torch.uint8),
            ),
            dim=1,
        )
    )
    layer.w13_weight_scale = _parameter(
        torch.cat(
            (
                torch.full(projection_scale_shape, 1.0, dtype=torch.float8_e4m3fn),
                torch.full(projection_scale_shape, 2.0, dtype=torch.float8_e4m3fn),
            ),
            dim=1,
        )
    )

    w2_columns = torch.arange(1, TP_INTERMEDIATE_SIZE // 2 + 1, dtype=torch.uint8).view(
        1, 1, -1
    )
    layer.w2_weight = _parameter(
        w2_columns.expand(NUM_TEST_EXPERTS, GEMMA4_HIDDEN_SIZE, -1).contiguous()
    )
    w2_scale_columns = torch.arange(
        1, TP_INTERMEDIATE_SIZE // GROUP_SIZE + 1, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    layer.w2_weight_scale = _parameter(
        w2_scale_columns.view(1, 1, -1)
        .expand(NUM_TEST_EXPERTS, GEMMA4_HIDDEN_SIZE, -1)
        .contiguous()
    )

    layer.w13_input_scale = _parameter(torch.full((NUM_TEST_EXPERTS, 2), 0.5))
    layer.w2_input_scale = _parameter(torch.full((NUM_TEST_EXPERTS,), 0.25))
    layer.w13_weight_scale_2 = _parameter(
        torch.tensor([[0.125, 0.25]]).expand(NUM_TEST_EXPERTS, -1).contiguous()
    )
    layer.w2_weight_scale_2 = _parameter(torch.full((NUM_TEST_EXPERTS,), 0.5))

    layer.moe_runner_config = SimpleNamespace(
        is_gated=True,
        swiglu_limit=None,
        intermediate_size_per_partition=TP_INTERMEDIATE_SIZE,
    )
    layer.num_experts = NUM_TEST_EXPERTS
    layer.dispatcher = MagicMock()

    original_params = {
        name: getattr(layer, name)
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
        )
    }
    return layer, original_params


class TestNvfp4MoeIntermediatePadding(CustomTestCase):
    def test_process_weights_after_loading_pads_gemma4_tp2_before_swizzle(self):
        """Exercise the exact flashinfer_cutlass post-load regression path."""
        layer, original_params = _make_gemma4_tp2_layer()
        method = modelopt_quant.ModelOptNvFp4FusedMoEMethod.__new__(
            modelopt_quant.ModelOptNvFp4FusedMoEMethod
        )
        method.quant_config = SimpleNamespace(
            group_size=GROUP_SIZE,
            use_per_token_activation=False,
        )
        method.enable_flashinfer_trtllm_moe = False
        backend = modelopt_quant.MoeRunnerBackend.FLASHINFER_CUTLASS
        swizzle_inputs = []

        def record_swizzle_input(scale):
            swizzle_inputs.append(scale.detach().clone())
            padded_rows = (scale.shape[1] + 127) // 128 * 128
            padded_columns = (scale.shape[2] + 3) // 4 * 4
            return torch.zeros(
                scale.shape[0],
                padded_rows,
                padded_columns,
                dtype=scale.dtype,
                device=scale.device,
            )

        with (
            patch.object(
                modelopt_quant, "get_moe_runner_backend", return_value=backend
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=backend,
            ),
            patch.object(modelopt_quant, "MOE_NVFP4_DISPATCH", False),
            patch.object(
                modelopt_quant,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
            patch.object(
                modelopt_quant,
                "swizzle_blockscale",
                side_effect=record_swizzle_input,
            ),
        ):
            method.process_weights_after_loading(layer)

        self.assertEqual(len(swizzle_inputs), 2)
        self.assertEqual(
            swizzle_inputs[0].shape,
            (1, 768, 176),
        )
        self.assertEqual(
            swizzle_inputs[1].shape,
            (1, 2816, 24),
        )
        self.assertEqual(layer.w13_weight.shape, (1, 768, 1408))
        self.assertEqual(layer.w2_weight.shape, (1, 2816, 192))

        for name, padded, first_marker, second_marker in (
            ("w13_weight", layer.w13_weight, 0x11, 0x77),
            ("w13_weight_scale", swizzle_inputs[0], 1.0, 2.0),
        ):
            with self.subTest(tensor=name):
                self.assertTrue(
                    torch.equal(
                        padded[:, :TP_INTERMEDIATE_SIZE],
                        torch.full_like(padded[:, :TP_INTERMEDIATE_SIZE], first_marker),
                    )
                )
                first_padding = padded[:, TP_INTERMEDIATE_SIZE:PADDED_INTERMEDIATE_SIZE]
                self.assertTrue(
                    torch.equal(first_padding, torch.zeros_like(first_padding))
                )
                self.assertTrue(
                    torch.equal(
                        padded[
                            :,
                            PADDED_INTERMEDIATE_SIZE : PADDED_INTERMEDIATE_SIZE
                            + TP_INTERMEDIATE_SIZE,
                        ],
                        torch.full_like(
                            padded[
                                :,
                                PADDED_INTERMEDIATE_SIZE : PADDED_INTERMEDIATE_SIZE
                                + TP_INTERMEDIATE_SIZE,
                            ],
                            second_marker,
                        ),
                    )
                )
                second_padding = padded[
                    :, PADDED_INTERMEDIATE_SIZE + TP_INTERMEDIATE_SIZE :
                ]
                self.assertTrue(
                    torch.equal(second_padding, torch.zeros_like(second_padding))
                )

        expected_w2 = torch.arange(1, 177, dtype=torch.uint8).view(1, 1, -1)
        self.assertTrue(
            torch.equal(
                layer.w2_weight[..., : TP_INTERMEDIATE_SIZE // 2],
                expected_w2.expand(1, GEMMA4_HIDDEN_SIZE, -1),
            )
        )
        w2_padding = layer.w2_weight[..., TP_INTERMEDIATE_SIZE // 2 :]
        self.assertTrue(torch.equal(w2_padding, torch.zeros_like(w2_padding)))

        padded_w2_scale = swizzle_inputs[1]
        expected_w2_scale = torch.arange(1, 23, dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        self.assertTrue(
            torch.equal(
                padded_w2_scale[..., : TP_INTERMEDIATE_SIZE // GROUP_SIZE],
                expected_w2_scale.view(1, 1, -1).expand(1, GEMMA4_HIDDEN_SIZE, -1),
            )
        )
        w2_scale_padding = padded_w2_scale[..., TP_INTERMEDIATE_SIZE // GROUP_SIZE :]
        self.assertTrue(
            torch.equal(w2_scale_padding, torch.zeros_like(w2_scale_padding))
        )

        for name, original_param in original_params.items():
            with self.subTest(parameter_identity=name):
                self.assertIs(getattr(layer, name), original_param)

        for source_name, derived_name, expected_shape in (
            (
                "w13_weight_scale",
                "w13_blockscale_swizzled",
                (1, 768, 176),
            ),
            ("w2_weight_scale", "w2_blockscale_swizzled", (1, 2816, 24)),
        ):
            source = getattr(layer, source_name)
            derived = getattr(layer, derived_name)
            with self.subTest(blockscale_alias=derived_name):
                self.assertIs(derived, source)
                self.assertEqual(derived.shape, expected_shape)
                self.assertEqual(derived.dtype, torch.float8_e4m3fn)

        self.assertEqual(
            layer.moe_runner_config.intermediate_size_per_partition,
            TP_INTERMEDIATE_SIZE,
        )
        self.assertEqual(
            layer.cutlass_moe_params.intermediate_size_per_partition,
            PADDED_INTERMEDIATE_SIZE,
        )
        self.assertEqual(layer.cutlass_moe_params.hidden_size, GEMMA4_HIDDEN_SIZE)

    def test_gemma4_tp1_intermediate_padding_is_noop(self):
        """Each TP=1 projection has 704 rows, already a multiple of 64."""
        hidden_size = 16
        intermediate_size = GEMMA4_MOE_INTERMEDIATE_SIZE
        tensors = (
            torch.zeros(1, 2 * intermediate_size, hidden_size // 2, dtype=torch.uint8),
            torch.zeros(
                1,
                2 * intermediate_size,
                hidden_size // GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
            ),
            torch.zeros(1, hidden_size, intermediate_size // 2, dtype=torch.uint8),
            torch.zeros(
                1,
                hidden_size,
                intermediate_size // GROUP_SIZE,
                dtype=torch.float8_e4m3fn,
            ),
        )

        padded = modelopt_quant._pad_nvfp4_gated_moe_weights_for_swizzle(
            *tensors, group_size=GROUP_SIZE
        )

        for source, result in zip(tensors, padded):
            self.assertIs(result, source)


if __name__ == "__main__":
    unittest.main()
