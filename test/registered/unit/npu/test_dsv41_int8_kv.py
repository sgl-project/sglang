"""Independent byte-layout and page-table checks for the 910B CSA cache."""

import struct
import unittest

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import decode_int8, encode_int8
from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8_kv import (
    materialize_kv_pages,
    pack_kv,
    unpack_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestDSV41Int8KV(unittest.TestCase):
    def test_signed_levels_and_independent_scale_decode(self):
        gen = torch.Generator().manual_seed(11419)
        rows = torch.randn((3, 512), generator=gen).bfloat16()
        rows[0].zero_()
        rows[1, :5] = torch.tensor([0, -1.125, 2**-19, -(2**-25), 2**-42])
        records = pack_kv(rows)
        self.assertEqual((records.shape, records.dtype), ((3, 896), torch.uint8))
        self.assertTrue(
            torch.equal(records[:, :128].view(torch.bfloat16), rows[:, 448:])
        )
        self.assertTrue((records[:, 866:] == 0).all())
        actual = unpack_kv(records)
        for i in range(rows.shape[0]):
            data = bytes(records[i].tolist())
            levels = struct.unpack_from("<448b", data, 128)
            factor_bits = struct.unpack_from("<H", data, 864)[0]
            factor = struct.unpack("<f", struct.pack("<I", factor_bits << 16))[0]
            reconstructed = []
            for channel, q in enumerate(levels):
                halfbyte = data[640 + channel // 2]
                slot = (halfbyte >> (4 * (channel % 2))) & 15
                offset = 576 + (16 * (channel // 224) + slot) * 2
                bits = struct.unpack_from("<H", data, offset)[0]
                if bits & 0x8000:
                    scale = struct.unpack(
                        "<f", struct.pack("<I", (bits & 0x7FFF) << 16)
                    )[0]
                else:
                    scale = struct.unpack_from("<e", data, offset)[0]
                reconstructed.append(q * scale * factor)
            expected = torch.tensor(reconstructed).bfloat16()
            torch.testing.assert_close(actual[i, :448], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, rows, atol=1e-5, rtol=1e-5)

    def test_empty_and_strided_rows(self):
        storage = torch.randn(
            6, 256, generator=torch.Generator().manual_seed(27)
        ).bfloat16()
        rows = storage[::2, ::2]
        q, metadata = encode_int8(rows, 128)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(metadata.shape, (3, 98))
        torch.testing.assert_close(
            decode_int8(q, metadata, 128), rows, atol=1e-5, rtol=1e-5
        )
        empty = torch.empty((0, 1, 512), dtype=torch.bfloat16)
        self.assertEqual(unpack_kv(pack_kv(empty)).shape, empty.shape)

    def test_active_and_static_physical_pages(self):
        rows = torch.randn(
            7, 4, 1, 512, generator=torch.Generator().manual_seed(23)
        ).bfloat16()
        rows[0].zero_()
        packed = pack_kv(rows)
        table = torch.tensor([[5, 2, -1, 87], [5, 4, 1, -1]], dtype=torch.int32)
        lengths = torch.tensor([7, 10])
        for static in (False, True):
            dense, mapping = materialize_kv_pages(
                packed, table, lengths, fixed_shape=static
            )
            self.assertEqual(mapping[0, 2:].tolist(), [-1, 0])
            for row, count in enumerate((2, 3)):
                expected = unpack_kv(packed[table[row, :count].long()])
                torch.testing.assert_close(
                    dense[mapping[row, :count].long()], expected, atol=0, rtol=0
                )
        ref, ref_table = materialize_kv_pages(rows, table, lengths)
        self.assertIs(ref, rows)
        self.assertIs(ref_table, table)


if __name__ == "__main__":
    unittest.main()
