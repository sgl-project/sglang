"""MXFP8 TRT-LLM MoE alignment must match the per-expert reference byte for byte."""

import unittest
from types import SimpleNamespace

import torch

# Import quantization before flashinfer_trtllm; the reverse is a circular import.
# isort: off
from sglang.srt.layers.quantization import fp8  # noqa: F401
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    _align_mxfp8_moe_weights,
    align_mxfp8_moe_weights_for_flashinfer_trtllm,
    clear_mxfp8_shuffle_index_cache,
)

# isort: on
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEVICE = torch.device("cuda")
BLOCK = 32
TILE = 128


def _make_layer(num_experts, hidden, intermediate, is_gated, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand_u8(*shape):
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device=DEVICE, generator=gen
        )

    # Non-gated layers keep the gated w13 allocation and use its first half.
    gate_up = 2 * intermediate
    layer = torch.nn.Module()
    layer.moe_runner_config = SimpleNamespace(is_gated=is_gated)
    for name, value in (
        (
            "w13_weight",
            rand_u8(num_experts, gate_up, hidden).view(torch.float8_e4m3fn),
        ),
        (
            "w2_weight",
            rand_u8(num_experts, hidden, intermediate).view(torch.float8_e4m3fn),
        ),
        ("w13_weight_scale_inv", rand_u8(num_experts, gate_up, hidden // BLOCK)),
        ("w2_weight_scale_inv", rand_u8(num_experts, hidden, intermediate // BLOCK)),
    ):
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    return layer


def _reference(layer, is_gated):
    """The alignment, one expert at a time."""
    from flashinfer import block_scale_interleave
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )
    from flashinfer.utils import (
        get_shuffle_matrix_a_row_indices,
        get_shuffle_matrix_sf_a_row_indices,
    )

    w13 = layer.w13_weight.data.clone()
    w2 = layer.w2_weight.data.clone()
    w13_scale = layer.w13_weight_scale_inv.data.clone()
    w2_scale = layer.w2_weight_scale_inv.data.clone()
    if not is_gated:
        intermediate = w2.shape[2]
        w13 = w13[:, :intermediate, :].contiguous()
        w13_scale = w13_scale[:, :intermediate, :].contiguous()
    w13, w13_scale, w2, w2_scale, _ = _align_mxfp8_moe_weights(
        w13, w13_scale, w2, w2_scale, is_gated, 16 if is_gated else 128
    )
    num_experts, gate_up, _ = w13.shape
    hidden = w2.shape[1]
    w13_u8, w2_u8 = w13.view(torch.uint8), w2.view(torch.uint8)

    if is_gated:
        reorder = get_reorder_rows_for_gated_act_gemm_row_indices(w13_u8[0]).to(DEVICE)
    else:
        reorder = torch.arange(gate_up, device=DEVICE)
    w13_rows = get_shuffle_matrix_a_row_indices(w13_u8[0], TILE).to(DEVICE)
    w2_rows = get_shuffle_matrix_a_row_indices(w2_u8[0], TILE).to(DEVICE)
    w13_scale_rows = get_shuffle_matrix_sf_a_row_indices(
        w13_scale[0].reshape(gate_up, -1), TILE
    ).to(DEVICE)
    w2_scale_rows = get_shuffle_matrix_sf_a_row_indices(
        w2_scale[0].reshape(hidden, -1), TILE
    ).to(DEVICE)

    out_w13, out_w2 = torch.empty_like(w13_u8), torch.empty_like(w2_u8)
    out_w13_scale, out_w2_scale = (
        torch.empty_like(w13_scale),
        torch.empty_like(w2_scale),
    )
    for i in range(num_experts):
        out_w13[i] = w13_u8[i].index_select(0, reorder).index_select(0, w13_rows)
        out_w2[i] = w2_u8[i].index_select(0, w2_rows)
        out_w13_scale[i] = block_scale_interleave(
            w13_scale[i]
            .index_select(0, reorder)
            .reshape(gate_up, -1)
            .index_select(0, w13_scale_rows)
        ).reshape_as(out_w13_scale[i])
        out_w2_scale[i] = block_scale_interleave(
            w2_scale[i].reshape(hidden, -1).index_select(0, w2_scale_rows)
        ).reshape_as(out_w2_scale[i])
    return {
        "w13_weight": out_w13,
        "w2_weight": out_w2,
        "w13_weight_scale_inv": out_w13_scale,
        "w2_weight_scale_inv": out_w2_scale,
    }


class TestMxfp8MoeTrtllmAlign(CustomTestCase):
    def test_matches_per_expert_reference(self):
        cases = [
            # (num_experts, hidden, intermediate, is_gated)
            (8, 256, 128, True),
            (256, 512, 256, True),
            (16, 384, 256, True),
            (8, 256, 128, False),
            (8, 256, 96, False),  # padded to 128
        ]
        for seed, (num_experts, hidden, intermediate, is_gated) in enumerate(cases):
            with self.subTest(
                num_experts=num_experts,
                hidden=hidden,
                intermediate=intermediate,
                is_gated=is_gated,
            ):
                clear_mxfp8_shuffle_index_cache()
                layer = _make_layer(num_experts, hidden, intermediate, is_gated, seed)
                expected = _reference(layer, is_gated)
                align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)
                for name, value in expected.items():
                    actual = getattr(layer, name).data
                    self.assertEqual(actual.shape, value.shape, name)
                    self.assertTrue(
                        torch.equal(actual.view(torch.uint8), value.view(torch.uint8)),
                        name,
                    )


if __name__ == "__main__":
    unittest.main()
