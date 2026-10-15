"""Weight checker on checkpoint-layout NVFP4 experts (Muse Spark on Marlin), on CPU.

The checker compares the NVFP4 experts in dequantized space,
e2m1 * e4m3 block scale * fp32 global scale, so a re-quantized update that
encodes the same values differently compares equal, and one that rounds an
element to a neighbouring code stays within allow_quant_error. The Marlin
repack derived from those experts is never compared or hashed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.muse_spark_nvfp4_moe import (
    MuseSparkNvFp4FusedMoEMethod,
)
from sglang.srt.utils.weight_checker import (
    QuantizedWeight,
    _build_check_entries,
    _build_quantized_set,
    _check_tensors,
    _hash_tensor,
)
from sglang.srt.utils.weight_checker_comparator import (
    NvFp4Comparable,
    select_comparable_weight,
)
from sglang.test.test_utils import CustomTestCase

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NUM_EXPERTS, N, K = 2, 8, 32


@contextmanager
def _cpu_chunks():
    """The comparator moves chunks to CUDA; keep them on CPU."""
    with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
        yield


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """4-bit codes (..., K) -> bytes (..., K/2), low nibble first."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _random_experts(seed: int):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (NUM_EXPERTS, N, K), generator=gen, dtype=torch.uint8)
    block = (torch.rand(NUM_EXPERTS, N, K // 16, generator=gen) * 8 + 0.5).to(
        torch.float8_e4m3fn
    )
    # gate rows [0, N/2) and up rows [N/2, N) carry their own global scale
    global_scale = torch.rand(NUM_EXPERTS, 2, generator=gen) * 0.01 + 0.001
    return codes, block, global_scale


def _reference(codes, block, global_scale):
    magnitude = E2M1[(codes & 0x7).long()]
    values = torch.where((codes & 0x8).bool(), -magnitude, magnitude)
    rows = torch.arange(N) // (N // 2)
    scale = block.float().repeat_interleave(16, dim=-1)
    return values * (scale * global_scale[:, rows].unsqueeze(-1))


def _entries(name, comparable):
    return [(name, True, comparable)]


class TestNvFp4Comparable(CustomTestCase):
    def test_dequantizes_low_nibble_first_with_per_half_global_scales(self):
        codes, block, global_scale = _random_experts(seed=0)
        comparable = NvFp4Comparable(_pack(codes), block, global_scale)
        torch.testing.assert_close(
            comparable.dequantize(torch.float32),
            _reference(codes, block, global_scale),
            rtol=0,
            atol=0,
        )

    def test_equivalent_encodings_compare_and_hash_equal(self):
        """Doubling a block scale while halving its codes, or writing -0 for +0,
        must not read as a weight change."""
        codes = torch.full((NUM_EXPERTS, N, K), 4, dtype=torch.uint8)  # 2.0
        codes[0, 0, 0] = 0  # +0
        block = torch.ones(NUM_EXPERTS, N, K // 16).to(torch.float8_e4m3fn)
        global_scale = torch.full((NUM_EXPERTS, 2), 0.01)
        other_codes = torch.full_like(codes, 2)  # 1.0
        other_codes[0, 0, 0] = 8  # -0
        expect = NvFp4Comparable(_pack(codes), block, global_scale)
        actual = NvFp4Comparable(
            _pack(other_codes),
            (block.float() * 2).to(torch.float8_e4m3fn),
            global_scale,
        )
        with _cpu_chunks():
            _check_tensors(_entries("w", expect), _entries("w", actual))
        self.assertEqual(
            _hash_tensor(expect.dequantize()), _hash_tensor(actual.dequantize())
        )

    def test_one_code_step_is_quant_error_but_a_large_change_is_not(self):
        codes, block, global_scale = _random_experts(seed=1)
        codes[..., 0] = 2  # 1.0
        expect = NvFp4Comparable(_pack(codes), block, global_scale)
        neighbour = codes.clone()
        neighbour[1, N - 1, 0] = 3  # 1.5, in the up half
        far = codes.clone()
        far[1, N - 1, 0] = 7  # 6.0
        with _cpu_chunks():
            actual = NvFp4Comparable(_pack(neighbour), block, global_scale)
            _check_tensors(
                _entries("w", expect), _entries("w", actual), allow_quant_error=True
            )
            with self.assertRaisesRegex(Exception, "check tensor equality failed"):
                _check_tensors(_entries("w", expect), _entries("w", actual))
            actual = NvFp4Comparable(_pack(far), block, global_scale)
            with self.assertRaisesRegex(Exception, "num_exceed=1 "):
                _check_tensors(
                    _entries("w", expect), _entries("w", actual), allow_quant_error=True
                )
            swapped = NvFp4Comparable(_pack(codes), block, global_scale.flip(-1))
            with self.assertRaisesRegex(Exception, "check tensor equality failed"):
                _check_tensors(
                    _entries("w", expect),
                    _entries("w", swapped),
                    allow_quant_error=True,
                )


def _muse_marlin_layer(keep_checkpoint_layout: bool) -> nn.Module:
    method = MuseSparkNvFp4FusedMoEMethod.__new__(MuseSparkNvFp4FusedMoEMethod)
    method._moe_runner_backend = MoeRunnerBackend.MARLIN
    method.keep_checkpoint_layout = keep_checkpoint_layout
    codes, block, global_scale = _random_experts(seed=2)
    layer = nn.Module()
    layer.quant_method = method
    tensors = {
        "gate_up_proj": _pack(codes),
        "gate_up_proj_weight_scale": block,
        "gate_up_proj_weight_scale_2": global_scale,
        "gate_up_proj_input_scale": torch.ones(NUM_EXPERTS, 2),
        "down_proj": _pack(codes),
        "down_proj_weight_scale": block,
        "down_proj_weight_scale_2": global_scale[:, 0].clone(),
        "down_proj_input_scale": torch.ones(NUM_EXPERTS),
        # Marlin repack: int32 tiles, e4m3 scales, bf16 global scales
        "w13_weight": torch.zeros(NUM_EXPERTS, K // 16, 2 * N, dtype=torch.int32),
        "w13_weight_scale": block.clone(),
        "w13_weight_scale_2": torch.ones(NUM_EXPERTS, dtype=torch.bfloat16),
        "w2_weight": torch.zeros(NUM_EXPERTS, K // 16, 2 * N, dtype=torch.int32),
        "w2_weight_scale": block.clone(),
        "w2_weight_scale_2": torch.ones(NUM_EXPERTS, dtype=torch.bfloat16),
    }
    for name, tensor in tensors.items():
        layer.register_parameter(name, nn.Parameter(tensor, requires_grad=False))
    model = nn.Module()
    model.experts = layer
    return model


class TestMuseSparkMarlinRouting(CustomTestCase):
    def test_compares_checkpoint_layout_and_ignores_marlin_repack(self):
        model = _muse_marlin_layer(keep_checkpoint_layout=True)
        quantized_set = _build_quantized_set(model)
        self.assertEqual(
            quantized_set["experts.gate_up_proj"],
            QuantizedWeight(
                NvFp4Comparable,
                "experts.gate_up_proj_weight_scale",
                global_scale_name="experts.gate_up_proj_weight_scale_2",
                derived_names=(
                    "experts.w13_weight",
                    "experts.w13_weight_scale",
                    "experts.w13_weight_scale_2",
                ),
            ),
        )
        self.assertEqual(
            set(quantized_set), {"experts.gate_up_proj", "experts.down_proj"}
        )

        expect = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        actual = dict(model.named_parameters())
        with torch.no_grad():
            for name in ("w13_weight", "w2_weight"):
                getattr(model.experts, name).fill_(-1)
        entries = list(_build_check_entries(actual, set(), quantized_set))
        self.assertEqual(
            [name for name, _, _ in entries],
            [
                "experts.gate_up_proj",
                "experts.gate_up_proj_input_scale",
                "experts.down_proj",
                "experts.down_proj_input_scale",
            ],
        )
        with _cpu_chunks():
            _check_tensors(
                _build_check_entries(expect, set(), quantized_set), iter(entries)
            )

    def test_released_checkpoint_layout_is_rejected(self):
        model = _muse_marlin_layer(keep_checkpoint_layout=False)
        with self.assertRaisesRegex(
            NotImplementedError, "SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE"
        ):
            select_comparable_weight(model.experts.quant_method)


if __name__ == "__main__":
    unittest.main()
