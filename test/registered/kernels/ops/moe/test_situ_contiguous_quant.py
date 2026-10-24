"""Contiguous SiTU quantization must use the scale that reaches the GEMM."""

import math
import unittest

import torch
import triton

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_BETA = 4.0
_LINEAR_BETA = 25.0
_CASES = (
    "zero",
    "ones",
    "random_seed_3",
    "signed",
    "below_power_of_two",
    "near_power_of_two",
    "above_power_of_two",
)


def _situ_reference(gate_up: torch.Tensor) -> torch.Tensor:
    # Keep the oracle on CPU and independent of the kernel's sigmoid-based tanh.
    assert gate_up.device.type == "cpu"
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (
        _BETA
        * torch.tanh(gate / _BETA)
        * torch.sigmoid(gate)
        * _LINEAR_BETA
        * torch.tanh(up / _LINEAR_BETA)
    )


def _make_inputs(group_size: int, num_groups: int):
    width = group_size * num_groups
    values = torch.zeros(len(_CASES), 2 * width, dtype=torch.bfloat16, device="cpu")
    values[1].fill_(1)

    generator = torch.Generator(device="cpu").manual_seed(3)
    values[2] = torch.randn(2 * width, generator=generator, device="cpu") * 2.0
    signed = torch.rand(2 * width, generator=generator, device="cpu") * 3.0 + 0.5
    signed[width::2].neg_()
    values[3] = signed

    # Solve for up at three scale boundaries, then use neighboring BF16 inputs
    # on either side. Distinct group scales also catch packing-order mistakes.
    target_scales = torch.tensor(
        [2.0 ** (-9 + 2 * (group % 3)) for group in range(num_groups)],
        dtype=torch.float64,
        device="cpu",
    )
    gate_value = 4.0
    gate_factor = _BETA * math.tanh(gate_value / _BETA) / (1.0 + math.exp(-gate_value))
    boundary_up = (
        _LINEAR_BETA * torch.atanh(448.0 * target_scales / (gate_factor * _LINEAR_BETA))
    ).to(torch.bfloat16)
    below = torch.nextafter(boundary_up, torch.full_like(boundary_up, -math.inf))
    above = torch.nextafter(boundary_up, torch.full_like(boundary_up, math.inf))
    for row, up in enumerate((below, boundary_up, above), start=4):
        values[row, :width].fill_(gate_value)
        values[row, width:] = up.repeat_interleave(group_size)
    return values, target_scales.float()


def _unpack_contiguous_scales(packed: torch.Tensor, num_groups: int) -> torch.Tensor:
    # The contiguous runner packs [1, rows, groups] into [1, rows, groups / 4].
    # Decode words after moving to CPU; do not reuse the rounding implementation.
    words = packed.detach().cpu().to(torch.int64)
    shifts = torch.arange(4, dtype=torch.int64, device="cpu") * 8
    exponents = ((words.unsqueeze(-1) >> shifts) & 0xFF).reshape(
        words.shape[1], num_groups
    )
    return torch.exp2(exponents.float() - 127.0)


class TestSituContiguousQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("the UE8M0 SiTU runner requires Blackwell")

    def test_quantization_with_packed_scales(self):
        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            _cast_to_e8m0_with_rounding_up,
            _situ_mul_quant_contig_kernel,
        )

        # Seven input cases share one launch per shape. KG=12 exercises the
        # masked groups in KG_POW2=16 without extending the packing contract.
        for group_size in (32, 128):
            for num_groups in (4, 8, 12):
                with self.subTest(group_size=group_size, num_groups=num_groups):
                    inputs_cpu, target_scales = _make_inputs(group_size, num_groups)
                    expected = _situ_reference(inputs_cpu)
                    reference_scales = (
                        expected.reshape(len(_CASES), num_groups, group_size)
                        .abs()
                        .amax(dim=-1)
                        / 448.0
                    )
                    self.assertTrue(bool((reference_scales[4] < target_scales).all()))
                    self.assertTrue(bool((reference_scales[6] > target_scales).all()))

                    inputs = inputs_cpu.to("cuda")
                    rows, twice_width = inputs.shape
                    width = twice_width // 2
                    output = torch.empty(
                        (rows, width), dtype=torch.float8_e4m3fn, device="cuda"
                    )
                    raw_scales = torch.empty(
                        (rows, num_groups), dtype=torch.float32, device="cuda"
                    )
                    _situ_mul_quant_contig_kernel[(rows,)](
                        inputs,
                        output,
                        raw_scales,
                        width,
                        num_groups,
                        _BETA,
                        _LINEAR_BETA,
                        GROUP=group_size,
                        KG_POW2=triton.next_power_of_2(num_groups),
                        num_warps=8,
                    )
                    # Call the production torch.compile wrapper, not __wrapped__.
                    packed = _cast_to_e8m0_with_rounding_up(raw_scales.unsqueeze(0))
                    self.assertEqual(packed.dtype, torch.int32)
                    self.assertEqual(tuple(packed.shape), (1, rows, num_groups // 4))
                    scales = _unpack_contiguous_scales(packed, num_groups)
                    raw_scales_cpu = raw_scales.cpu()
                    expanded_scales = scales.repeat_interleave(group_size, dim=-1)
                    actual = output.cpu().float() * expanded_scales
                    error = (actual - expected).abs()
                    # Same bound as the existing SiTU kernel test. FP8 rounding
                    # is lossy; exact equality with the SiTU reference is wrong.
                    bound = expanded_scales * 17.0

                    for row, case in enumerate(_CASES):
                        with self.subTest(case=case):
                            self.assertTrue(bool(torch.isfinite(scales[row]).all()))
                            self.assertTrue(bool((scales[row] > 0).all()))
                            # Pin the expected bin on both sides so inflating a
                            # scale cannot hide error by widening the bound.
                            if case == "below_power_of_two":
                                self.assertTrue(torch.equal(scales[row], target_scales))
                            elif case == "above_power_of_two":
                                self.assertTrue(
                                    torch.equal(scales[row], target_scales * 2.0)
                                )
                            self.assertTrue(bool(torch.isfinite(actual[row]).all()))
                            self.assertTrue(
                                bool((error[row] <= bound[row]).all()),
                                f"{case}: max error={error[row].max().item()}, "
                                "max error/bound="
                                f"{(error[row] / bound[row]).max().item()}",
                            )
                            self.assertTrue(
                                torch.equal(raw_scales_cpu[row], scales[row]),
                                "packing changed the scale used for FP8 quantization",
                            )


if __name__ == "__main__":
    unittest.main()
