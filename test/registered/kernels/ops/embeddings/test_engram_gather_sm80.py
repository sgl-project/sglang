"""SM80 engram gather: GPU byte tables, independent CPU bit-format oracle.

Run directly with unittest on CUDA SM80. No direct SRT imports, native FP8 views,
experimental flags, or host-table support claims. NaN payloads are unspecified;
all other BF16 bits (including signed zeros and infinities) must match exactly.
"""

import unittest

import torch
from sglang.kernels.ops.embeddings.engram_gather import engram_gather


def _is_sm80():
    return (
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0)
    )


def _decode_e4m3_cpu(raw):
    """Build FP32 bits with integer significands, not a native FP8 cast."""
    assert raw.device.type == "cpu" and raw.dtype == torch.uint8
    table = []
    for code in range(256):
        sign = (code >> 7) << 31
        exponent, fraction = (code >> 3) & 15, code & 7
        if exponent == 15 and fraction == 7:
            magnitude = 0x7FC00000
        elif exponent == 0 and fraction == 0:
            magnitude = 0
        else:
            # value = integer_significand * 2**power, exactly. Normalize
            # using bit_length rather than the kernel's piecewise decoder.
            significand = fraction if exponent == 0 else 8 + fraction
            power = -9 if exponent == 0 else exponent - 10
            top = significand.bit_length() - 1
            magnitude = ((power + top + 127) << 23) | (
                (significand - (1 << top)) << (23 - top)
            )
        table.append(sign | magnitude)
    bits = torch.tensor(table, dtype=torch.int64).to(torch.int32)
    return bits[raw.long()].contiguous().view(torch.float32)


def _decode_ue8m0_cpu(raw):
    """Exact FP32 bit construction, including the two boundary codes."""
    assert raw.device.type == "cpu" and raw.dtype == torch.uint8
    bits = raw.to(torch.int64) << 23
    bits[raw == 0] = 0x00400000  # 2**-127, not zero
    bits[raw == 255] = 0x7FC00000  # NaN, not infinity
    return bits.to(torch.int32).contiguous().view(torch.float32)


def _reference(weight, scale, ids, block_size, row_lo, row_hi):
    assert weight.device.type == scale.device.type == ids.device.type == "cpu"
    values = _decode_e4m3_cpu(weight)
    scales = _decode_ue8m0_cpu(scale).repeat_interleave(block_size, dim=1)
    # Multiply in CPU FP32 first (including overflow/underflow and 0 * NaN),
    # then perform exactly one BF16 RNE conversion. Never use exp/exp2.
    decoded = (values * scales).to(torch.bfloat16)
    expected = torch.zeros(ids.numel(), weight.shape[1], dtype=torch.bfloat16)
    owned = (ids >= row_lo) & (ids < row_hi)
    expected[owned] = decoded[(ids[owned] - row_lo).long()]
    return expected


@unittest.skipUnless(_is_sm80(), "requires CUDA SM80 (8, 0)")
class TestEngramGatherSM80(unittest.TestCase):
    def _assert_exact(self, actual, expected):
        actual = actual.cpu()
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.shape, expected.shape)
        # equal_nan comparison checks overflow classifications; bit comparison
        # additionally distinguishes signed zero. Ignore only NaN payloads.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        not_nan = ~torch.isnan(expected)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.int16)[not_nan],
                expected.contiguous().view(torch.int16)[not_nan],
            )
        )

    def _run(self, weight, scale, ids, row_lo, row_hi):
        dim, block_size = weight.shape[1], 32
        # Nonzero source offsets ensure the original supplied raw addresses
        # are honored; neither allocation nor pointer is reinterpreted as FP8.
        weight_storage = torch.full(
            (weight.numel() + 64,), 127, dtype=torch.uint8, device="cuda"
        )
        scale_storage = torch.full(
            (scale.numel() + 64,), 255, dtype=torch.uint8, device="cuda"
        )
        weight_gpu = weight_storage[32 : 32 + weight.numel()].view_as(weight)
        scale_gpu = scale_storage[32 : 32 + scale.numel()].view_as(scale)
        weight_gpu.copy_(weight)
        scale_gpu.copy_(scale)
        ids_gpu = ids.cuda()
        out = torch.full((ids.numel(), dim), float("nan"), dtype=torch.bfloat16).cuda()
        got = engram_gather(
            weight_gpu.data_ptr(),
            scale_gpu.data_ptr(),
            ids_gpu,
            out,
            dim,
            block_size,
            row_lo,
            row_hi,
        )
        self.assertIs(got, out)
        self._assert_exact(
            got, _reference(weight, scale, ids, block_size, row_lo, row_hi)
        )

    def test_all_fp8_codes_cross_all_scale_codes(self):
        # Every one of the 65,536 (E4M3 byte, UE8M0 byte) pairs, all owned.
        # In particular: +/-0 * NaN, E4M3 +/-NaN, byte-0 subnormals,
        # BF16 underflow ties and high scales overflowing FP32/BF16.
        weight = torch.arange(256, dtype=torch.uint8).repeat(256, 1)
        scale = torch.arange(256, dtype=torch.uint8)[:, None].repeat(1, 8)
        row_lo = 19
        for dtype in (torch.int32, torch.int64):
            with self.subTest(index_dtype=dtype):
                ids = torch.arange(row_lo + 255, row_lo - 1, -1, dtype=dtype)
                self._run(weight, scale, ids, row_lo, row_lo + 256)

    def test_dimensions_shards_duplicates_and_bounds(self):
        for dim in (128, 256, 512):
            weight = (torch.arange(5 * dim).reshape(5, dim) % 256).to(torch.uint8)
            boundary_scales = torch.tensor(
                [0, 1, 120, 127, 128, 247, 254, 255], dtype=torch.uint8
            )
            scale = boundary_scales[
                torch.arange(5 * (dim // 32)).reshape(5, dim // 32) % 8
            ]
            for dtype in (torch.int32, torch.int64):
                for row_lo in (37,) if dtype == torch.int32 else (37, 2**40 + 37):
                    row_hi = row_lo + 5
                    ids = torch.tensor(
                        [
                            -1,
                            row_lo - 1,
                            row_lo,
                            row_hi - 1,
                            row_lo + 2,
                            row_lo,
                            row_hi,
                            row_hi + 17,
                            torch.iinfo(dtype).min,
                            torch.iinfo(dtype).max,
                        ],
                        dtype=dtype,
                    )
                    with self.subTest(dim=dim, index_dtype=dtype, row_lo=row_lo):
                        self._run(weight, scale, ids, row_lo, row_hi)

    def test_unowned_rows_do_not_dereference_raw_pointers(self):
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor([-1, 0, 36, 42, 1000], dtype=dtype, device="cuda")
            out = torch.full((ids.numel(), 128), float("nan"), device="cuda").to(
                torch.bfloat16
            )
            got = engram_gather(0, 0, ids, out, 128, 32, 37, 42)
            self._assert_exact(got, torch.zeros_like(out, device="cpu"))

    def test_empty(self):
        for dtype in (torch.int32, torch.int64):
            for dim in (128, 256, 512):
                ids = torch.empty(0, dtype=dtype, device="cuda")
                out = torch.empty((0, dim), dtype=torch.bfloat16, device="cuda")
                self.assertIs(engram_gather(0, 0, ids, out, dim, 32, 37, 42), out)

    def test_cuda_graph_capture_and_changed_ids(self):
        dim, row_lo, row_hi = 256, 2**40 + 13, 2**40 + 17
        weight = torch.arange(256, dtype=torch.uint8).repeat(4, 1)
        scale = torch.tensor([0, 127, 254, 255], dtype=torch.uint8)[:, None].repeat(
            1, dim // 32
        )
        weight_gpu, scale_gpu = weight.cuda(), scale.cuda()
        ids = torch.tensor([row_lo, row_hi - 1, -1, row_lo], dtype=torch.int64)
        ids_gpu = ids.cuda()
        out = torch.empty((ids.numel(), dim), dtype=torch.bfloat16, device="cuda")

        def gather():
            return engram_gather(
                weight_gpu.data_ptr(),
                scale_gpu.data_ptr(),
                ids_gpu,
                out,
                dim,
                32,
                row_lo,
                row_hi,
            )

        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                gather()
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather()
        graph.replay()
        self._assert_exact(out, _reference(weight, scale, ids, 32, row_lo, row_hi))
        ids = torch.tensor([row_lo + 1, row_hi, row_lo + 2, row_lo + 1])
        ids_gpu.copy_(ids)
        graph.replay()
        self._assert_exact(out, _reference(weight, scale, ids, 32, row_lo, row_hi))


if __name__ == "__main__":
    unittest.main()
