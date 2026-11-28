"""Ascend A5: serialized block-FP8 MoE experts must load as MXFP8.

A block-FP8 checkpoint carries ``[128, 128]`` fp32 block scales, which the A5
grouped matmul cannot consume — it wants 1x32 UE8M0 scales. The load path
requantizes every expert once; this checks the resulting layout and numerics.
"""

import unittest

import torch

# Import the quantization package first: moe_methods is part of a pre-existing
# import cycle when imported directly from a cold interpreter.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_utils import (
    _dequant_e4m3fn_to_float32,
    requant_block_fp8_to_mxfp8,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    process_npu_arch35_block_fp8_moe_weights,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="nightly-1-npu-a3", nightly=True)

BLOCK_SIZE = 128
MXFP8_BLOCK_SIZE = 32
NUM_EXPERTS = 2
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512


def _device() -> torch.device:
    return torch.device("npu", torch.npu.current_device())


def _block_fp8_experts(out_dim: int, in_dim: int):
    """Return (e4m3 payload, fp32 block scales) shaped [E, N, K] / [E, N//128, K//128]."""
    torch.manual_seed(0)
    device = _device()
    weight = torch.randn(NUM_EXPERTS, out_dim, in_dim, device=device) * 0.02
    scale = (
        weight.reshape(
            NUM_EXPERTS,
            out_dim // BLOCK_SIZE,
            BLOCK_SIZE,
            in_dim // BLOCK_SIZE,
            BLOCK_SIZE,
        )
        .abs()
        .amax(dim=(2, 4))
        / 448.0
    )
    payload = (
        weight.reshape(
            NUM_EXPERTS,
            out_dim // BLOCK_SIZE,
            BLOCK_SIZE,
            in_dim // BLOCK_SIZE,
            BLOCK_SIZE,
        )
        / scale[:, :, None, :, None]
    ).clamp(-448.0, 448.0)
    return payload.to(torch.float8_e4m3fn).reshape(NUM_EXPERTS, out_dim, in_dim), scale


class _FakeMoELayer(torch.nn.Module):
    """Only the parameters the MoE weight-loading hook touches."""

    def __init__(self):
        super().__init__()
        for prefix, out_dim, in_dim in (
            ("w13", 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ("w2", HIDDEN_SIZE, INTERMEDIATE_SIZE),
        ):
            payload, scale = _block_fp8_experts(out_dim, in_dim)
            self.register_parameter(
                f"{prefix}_weight", torch.nn.Parameter(payload, requires_grad=False)
            )
            self.register_parameter(
                f"{prefix}_weight_scale_inv",
                torch.nn.Parameter(scale, requires_grad=False),
            )


@unittest.skipIf(not is_npu_arch35(), "Ascend 950-class (arch35) only")
class TestBlockFp8MoEHasMXFP8Layout(unittest.TestCase):
    def test_experts_are_requantized_to_mxfp8_layout(self):
        layer = _FakeMoELayer()
        process_npu_arch35_block_fp8_moe_weights(layer, [BLOCK_SIZE, BLOCK_SIZE])

        for prefix, k_dim, n_dim in (
            ("w13", HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE),
            ("w2", INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")

            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(tuple(weight.shape), (NUM_EXPERTS, k_dim, n_dim))
            # Strided transpose view: npu_quant_matmul scans it K-major and
            # rejects a contiguous copy.
            self.assertFalse(weight.is_contiguous())

            self.assertEqual(scale.dtype, torch.uint8)
            self.assertEqual(
                tuple(scale.shape),
                (NUM_EXPERTS, k_dim // (2 * MXFP8_BLOCK_SIZE), n_dim, 2),
            )
            self.assertFalse(scale.is_contiguous())

            # The fp32 block scales are consumed at load time.
            self.assertIsNone(getattr(layer, f"{prefix}_weight_scale_inv"))

    def test_requantization_error_stays_quantization_bound(self):
        payload, block_scale = _block_fp8_experts(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
        expert, scale = payload[0], block_scale[0]

        qw, qs = requant_block_fp8_to_mxfp8(expert, scale, [BLOCK_SIZE, BLOCK_SIZE])
        dequantized = _dequant_e4m3fn_to_float32(qw) * torch.pow(
            2.0, qs.reshape(qw.shape[0], -1).to(torch.int32) - 127
        ).repeat_interleave(MXFP8_BLOCK_SIZE, dim=1)
        reference = _dequant_e4m3fn_to_float32(expert) * scale.repeat_interleave(
            BLOCK_SIZE, dim=0
        ).repeat_interleave(BLOCK_SIZE, dim=1)

        self.assertFalse(bool(torch.isnan(dequantized).any()))
        rel_frobenius = ((dequantized - reference).norm() / reference.norm()).item()
        # Measured 0.029 on Ascend 950PR; the gap to block-FP8 is the e8m0
        # 1x32 scale granularity, not a decoding bug.
        self.assertLess(rel_frobenius, 0.05)

    def test_pruned_experts_avoid_dead_scale_blocks(self):
        # Pruned checkpoints (Drama 3 zeroes 55% of the w13 blocks) hand back an
        # e8m0 scale of 0 for a fully zero e4m3 block, which the A5 grouped
        # matmul reads as a dead block instead of a zero one.
        weight = torch.zeros(256, 1024, device=_device(), dtype=torch.float8_e4m3fn)
        scale = torch.full((2, 8), 1.0, device=_device())
        qw, qs = requant_block_fp8_to_mxfp8(weight, scale, [BLOCK_SIZE, BLOCK_SIZE])

        self.assertEqual(int(qw.view(torch.uint8).abs().sum()), 0)
        self.assertGreater(int(qs.view(torch.uint8).min()), 0)

    def test_dispatcher_is_switched_to_mxfp8(self):
        # The MXFP8 kernels expect routing to emit e4m3 activations plus e8m0
        # scales, so going through NPUMXFP8MoEMethod is what leaves the layer in
        # a servable state. Hand-building the layout instead leaves the
        # dispatcher on BF16: every weight still matches the checkpoint, but the
        # experts lose most of their contribution (measured on Drama 3: mean CER
        # 64 % against 10 % for the offline export of the same checkpoint).
        class _RecordingDispatcher:
            def __init__(self):
                self.config = None

            def set_quant_config(self, config):
                self.config = config

        layer = _FakeMoELayer()
        layer.dispatcher = _RecordingDispatcher()
        process_npu_arch35_block_fp8_moe_weights(layer, [BLOCK_SIZE, BLOCK_SIZE])

        self.assertEqual(layer.dispatcher.config["dispatcher_output_dtype"], "mxfp8")


if __name__ == "__main__":
    unittest.main()
