"""CUDA unit tests for the MXFP4 expert-pack kernels."""

import unittest

import torch

from sglang.kernels.ops.moe.expert_pack_mxfp4 import (
    mxfp4_matvec,
    mxfp4_matvec_dual,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _matvec_reference(
    input_cpu,
    cache_cpu,
    slots_cpu,
    role_offset,
    input_size,
    output_size,
    records_per_input,
):
    blocks = input_size // 32
    row_bytes = blocks * 17
    result = torch.zeros((slots_cpu.numel(), output_size), dtype=torch.float32)
    for record, slot in enumerate(slots_cpu.tolist()):
        input_row = record // records_per_input
        for output_row in range(output_size):
            total = 0.0
            row_offset = (
                int(slot) * cache_cpu.stride(0) + role_offset + output_row * row_bytes
            )
            for block in range(blocks):
                quant_offset = row_offset + block * 17
                scale = 2.0 ** (int(cache_cpu.view(-1)[quant_offset]) - 127)
                block_sum = 0.0
                for index in range(16):
                    packed = int(cache_cpu.view(-1)[quant_offset + index + 1])
                    block_sum += (
                        float(input_cpu[input_row, block * 32 + index])
                        * _FP4_VALUES[packed & 0xF]
                    )
                    block_sum += (
                        float(input_cpu[input_row, block * 32 + index + 16])
                        * _FP4_VALUES[packed >> 4]
                    )
                total += block_sum * scale
            result[record, output_row] = total
    return result


def _fill_mxfp4_cache(slots, role_bytes, input_size, device):
    blocks = input_size // 32
    cache_cpu = torch.zeros((slots, 3 * role_bytes), dtype=torch.uint8)
    for slot in range(slots):
        for role in range(3):
            role_rows = role_bytes // (blocks * 17)
            for row in range(role_rows):
                for block in range(blocks):
                    offset = (
                        slot * cache_cpu.stride(0)
                        + role * role_bytes
                        + row * blocks * 17
                        + block * 17
                    )
                    cache_cpu.view(-1)[offset] = 126 + ((slot + role + row + block) % 3)
                    for byte in range(16):
                        low = (slot + role * 3 + row + block + byte) % 16
                        high = (15 + slot + role + row + block - byte) % 16
                        cache_cpu.view(-1)[offset + byte + 1] = low | (high << 4)
    return cache_cpu.to(device=device)


@unittest.skipUnless(torch.cuda.is_available(), "MXFP4 kernel tests require CUDA")
class TestExpertPackMxfp4(unittest.TestCase):
    def test_matvec_fp16_and_bf16(self):
        input_size, output_size, records_per_input = 64, 5, 2
        blocks = input_size // 32
        role_bytes = output_size * blocks * 17
        cache = _fill_mxfp4_cache(2, role_bytes, input_size, "cuda")
        slots = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device="cuda")
        for dtype in (torch.float16, torch.bfloat16):
            if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            input_tensor = torch.arange(
                2 * input_size, dtype=torch.float32, device="cuda"
            ).reshape(2, input_size)
            input_tensor = ((input_tensor % 19) - 9).to(dtype)
            output = mxfp4_matvec(
                input_tensor,
                cache,
                slots,
                role_offset=role_bytes,
                role_bytes=role_bytes,
                input_size=input_size,
                output_size=output_size,
                records_per_input=records_per_input,
            )
            reference = _matvec_reference(
                input_tensor.cpu(),
                cache.cpu(),
                slots.cpu(),
                role_bytes,
                input_size,
                output_size,
                records_per_input,
            ).to(dtype)
            torch.testing.assert_close(output.cpu(), reference, rtol=0.03, atol=0.25)

    def test_matvec_dual_matches_two_roles(self):
        input_size, output_size, records_per_input = 64, 17, 2
        blocks = input_size // 32
        role_bytes = output_size * blocks * 17
        cache = _fill_mxfp4_cache(2, role_bytes, input_size, "cuda")
        slots = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda")
        input_tensor = (torch.randn(2, input_size, device="cuda") * 0.5).to(
            torch.float16
        )
        output_a, output_b = mxfp4_matvec_dual(
            input_tensor,
            cache,
            slots,
            gate_role_offset=0,
            up_role_offset=role_bytes,
            role_bytes=role_bytes,
            input_size=input_size,
            output_size=output_size,
            records_per_input=records_per_input,
        )
        expected_a = mxfp4_matvec(
            input_tensor,
            cache,
            slots,
            role_offset=0,
            role_bytes=role_bytes,
            input_size=input_size,
            output_size=output_size,
            records_per_input=records_per_input,
        )
        expected_b = mxfp4_matvec(
            input_tensor,
            cache,
            slots,
            role_offset=role_bytes,
            role_bytes=role_bytes,
            input_size=input_size,
            output_size=output_size,
            records_per_input=records_per_input,
        )
        torch.testing.assert_close(output_a, expected_a)
        torch.testing.assert_close(output_b, expected_b)

    def test_invalid_dimensions_are_rejected(self):
        input_tensor = torch.zeros((1, 31), dtype=torch.float16, device="cuda")
        cache = torch.zeros((1, 17), dtype=torch.uint8, device="cuda")
        slots = torch.zeros((1,), dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(RuntimeError, "divisible by 32"):
            mxfp4_matvec(
                input_tensor,
                cache,
                slots,
                role_offset=0,
                role_bytes=17,
                input_size=31,
                output_size=1,
                records_per_input=1,
            )


if __name__ == "__main__":
    unittest.main()
