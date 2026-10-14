"""CUDA unit tests for the MXFP4 expert-pack kernels."""

import unittest

import torch

from sglang.kernels.ops.moe.expert_pack_mxfp4 import (
    mxfp4_marlin_repack,
    mxfp4_matvec,
    mxfp4_matvec_dual,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")


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
_PACK_INDEX = (0, 2, 4, 6, 1, 3, 5, 7)


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


def _load_raw_word(raw_cpu, slot, role_offset, row, blocks_per_row, packed_word):
    block = packed_word // 4
    word_in_block = packed_word & 3
    offset = (
        slot * raw_cpu.stride(0)
        + role_offset
        + row * blocks_per_row * 17
        + block * 17
        + 1
        + word_in_block * 4
    )
    word = 0
    for byte in range(4):
        word |= int(raw_cpu.view(-1)[offset + byte]) << (8 * byte)
    return word


def _marlin_nibble(word, value_index):
    return (word >> ((value_index & 7) * 4)) & 0xF


def _marlin_scale_perm(index):
    local_perm = (0, 2, 1, 3)
    interleaved = (index // 4) * 4 + local_perm[index & 3]
    return ((interleaved & 7) * 8) + (interleaved >> 3)


def _repack_reference(
    raw_cpu, source_slot, role_bytes, input_size, output_size, gate_up
):
    blocks = input_size // 32
    total_words = (input_size // 16) * output_size * 2
    output = torch.empty(total_words, dtype=torch.int32)
    tile_span = (output_size // 64) * 128
    rows_per_role = output_size // 2 if gate_up else output_size
    for index in range(total_words):
        tile_k, tile_rem = divmod(index, tile_span)
        tile_n, local = divmod(tile_rem, 128)
        warp, thread = local & 3, local >> 2
        cur_n = warp * 16 + thread // 4
        tc_row = (thread & 3) * 2
        values = []
        for high in (False, True):
            source_row = tile_n * 64 + cur_n + (8 if high else 0)
            role = (
                1 if gate_up and source_row >= rows_per_role else (0 if gate_up else 2)
            )
            row = source_row % rows_per_role if gate_up else source_row
            for offset in (0, 1, 8, 9):
                value_index = tc_row + offset
                word = _load_raw_word(
                    raw_cpu,
                    source_slot,
                    role * role_bytes,
                    row,
                    blocks,
                    tile_k * 2 + value_index // 8,
                )
                values.append(_marlin_nibble(word, value_index))
        packed = 0
        for output_index, value_index in enumerate(_PACK_INDEX):
            packed |= values[value_index] << (output_index * 4)
        # The CUDA kernel stores the packed uint32 bit pattern in int32.
        output[index] = packed if packed < (1 << 31) else packed - (1 << 32)
    return output


def _scale_reference(
    raw_cpu, source_slot, role_bytes, input_size, output_size, gate_up
):
    blocks = input_size // 32
    output = torch.empty(blocks * output_size, dtype=torch.uint8)
    rows_per_role = output_size // 2 if gate_up else output_size
    for index in range(output.numel()):
        group, column = divmod(index, output_size)
        source_column = (column // 64) * 64 + _marlin_scale_perm(column & 63)
        role = (
            1 if gate_up and source_column >= rows_per_role else (0 if gate_up else 2)
        )
        row = source_column % rows_per_role if gate_up else source_column
        offset = (
            source_slot * raw_cpu.stride(0)
            + role * role_bytes
            + row * blocks * 17
            + group * 17
        )
        output[index] = raw_cpu.view(-1)[offset]
    return output


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

    def test_marlin_repack_weights_and_scales(self):
        hidden_size, intermediate_size = 64, 32
        w13_n, w2_n = 2 * intermediate_size, hidden_size
        role_bytes = intermediate_size * (hidden_size // 32) * 17
        raw_cpu = torch.zeros((4, 3 * role_bytes), dtype=torch.uint8)
        for slot in range(4):
            for role, rows, size in (
                (0, intermediate_size, hidden_size),
                (1, intermediate_size, hidden_size),
                (2, hidden_size, intermediate_size),
            ):
                groups = size // 32
                for row in range(rows):
                    for group in range(groups):
                        offset = (
                            slot * raw_cpu.stride(0)
                            + role * role_bytes
                            + row * groups * 17
                            + group * 17
                        )
                        raw_cpu.view(-1)[offset] = (
                            100 + slot * 7 + role * 11 + row + group
                        ) % 256
                        for byte in range(16):
                            raw_cpu.view(-1)[offset + byte + 1] = (
                                (byte + row + role) % 16
                            ) | (((15 - byte + slot + group) % 16) << 4)
        raw = raw_cpu.cuda()
        source_slots = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
        target_slots = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
        w13_words = (hidden_size // 16) * w13_n * 2
        w2_words = (intermediate_size // 16) * w2_n * 2
        w13_scales = (hidden_size // 32) * w13_n
        w2_scales = (intermediate_size // 32) * w2_n
        w13 = torch.full((4, w13_words), -1, dtype=torch.int32, device="cuda")
        w2 = torch.full((4, w2_words), -1, dtype=torch.int32, device="cuda")
        w13_scale = torch.full((4, w13_scales), 255, dtype=torch.uint8, device="cuda")
        w2_scale = torch.full((4, w2_scales), 255, dtype=torch.uint8, device="cuda")
        mxfp4_marlin_repack(
            raw,
            source_slots,
            target_slots,
            role_bytes=role_bytes,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )
        torch.cuda.synchronize()
        for source_slot, target_slot in zip(
            source_slots.cpu().tolist(), target_slots.cpu().tolist()
        ):
            expected_w13 = _repack_reference(
                raw_cpu, source_slot, role_bytes, hidden_size, w13_n, True
            )
            expected_w2 = _repack_reference(
                raw_cpu, source_slot, role_bytes, intermediate_size, w2_n, False
            )
            expected_w13_scale = _scale_reference(
                raw_cpu, source_slot, role_bytes, hidden_size, w13_n, True
            )
            expected_w2_scale = _scale_reference(
                raw_cpu, source_slot, role_bytes, intermediate_size, w2_n, False
            )
            torch.testing.assert_close(w13[target_slot].cpu(), expected_w13)
            torch.testing.assert_close(w2[target_slot].cpu(), expected_w2)
            torch.testing.assert_close(w13_scale[target_slot].cpu(), expected_w13_scale)
            torch.testing.assert_close(w2_scale[target_slot].cpu(), expected_w2_scale)
        self.assertTrue(torch.all(w13[0] == -1))
        self.assertTrue(torch.all(w2[0] == -1))
        self.assertTrue(torch.all(w13_scale[0] == 255))
        self.assertTrue(torch.all(w2_scale[0] == 255))

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
