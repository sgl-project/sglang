"""Check captured DSv4.1 page reconstruction on changing logical and physical pages."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=40, suite="extra-b-test-1-npu-a3")


class TestDSV41Int8Pages(CustomTestCase):
    def test_small_eager_table_keeps_duplicate_logical_slots(self):
        from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import physical_page_map

        table = torch.tensor([[4, 4, 99], [4, 6, -1]], device="npu", dtype=torch.int32)
        lengths = torch.tensor([200, 150], device="npu", dtype=torch.int32)
        physical, logical = physical_page_map(
            table, 128, lengths, fixed_shape=False, pool_pages=16
        )
        self.assertEqual(physical.cpu().tolist(), [0, 4, 4, 0, 4, 6, 0])
        self.assertEqual(logical.cpu().tolist(), [[1, 2, 0], [4, 5, -1]])

        # Beyond the eager scratch budget, a padded table still deduplicates.
        padded = torch.cat((table, table.new_full((2, 2), 99)), dim=1)
        compact, remapped = physical_page_map(
            padded, 128, lengths, fixed_shape=False, pool_pages=16
        )
        self.assertEqual(compact.cpu().tolist(), [0, 4, 6])
        self.assertEqual(remapped.cpu().tolist(), [[1, 1, 0, 0, 0], [1, 2, -1, 0, 0]])

    def test_logical_decode_at_exact_physical_capacity(self):
        from sglang.kernels.ops.kvcache.dsv41_int8_pages import decode_logical_pages
        from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import (
            decode_int8,
            encode_int8,
        )

        pool, page_size = 8, 4
        gen = torch.Generator().manual_seed(19841)
        for width, group_size, rotary in ((128, 128, 0), (448, 224, 64)):
            with self.subTest(width=width):
                src = torch.randn(pool, page_size, 1, width, generator=gen).bfloat16()
                codes, meta = encode_int8(src, group_size)
                expected = decode_int8(codes, meta, group_size)
                rope = None
                if rotary:
                    rope = torch.randn(
                        pool, page_size, 1, rotary, generator=gen
                    ).bfloat16()
                    expected = torch.cat((expected, rope), dim=-1)
                    rope = rope.npu()
                codes, meta = codes.npu(), meta.npu()
                # Non-contiguous input views also occur with sliced capture buckets.
                table = torch.zeros(2, 6, device="npu", dtype=torch.int32)[:, :4]
                lengths = torch.zeros(2, 2, device="npu", dtype=torch.int32)[:, 0]

                def reconstruct():
                    return decode_logical_pages(
                        codes, meta, table, lengths, group_size, 16, rope
                    )

                reconstruct()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output, remapped = reconstruct()

                for ids, counts in (
                    ([[4, 4, 99, -1], [4, 6, 7, 99]], [7, 11]),
                    ([[7, 2, -1, 99], [2, 3, 99, -1]], [8, 5]),
                    ([[1, 2, 3, 99], [0, -1, -1, -1]], [11, 0]),
                ):
                    table.copy_(torch.tensor(ids, dtype=torch.int32))
                    lengths.copy_(torch.tensor(counts, dtype=torch.int32))
                    graph.replay()
                    torch.npu.synchronize()
                    actual, logical = output.cpu(), remapped.cpu()
                    self.assertEqual(actual.shape[0], pool + 1)
                    for row, count in enumerate(counts):
                        for column, physical in enumerate(ids[row]):
                            if column * page_size < count and physical > 0:
                                torch.testing.assert_close(
                                    actual[logical[row, column]],
                                    expected[physical],
                                    rtol=0,
                                    atol=0,
                                )
                            else:
                                self.assertEqual(
                                    int(logical[row, column]), min(physical, 0)
                                )

    def test_graph_replay_reconstructs_referenced_rows_and_rotary_channels(self):
        from sglang.kernels.ops.kvcache.dsv41_int8_pages import decode_pages
        from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import (
            SCALES_PER_GROUP,
            decode_int8,
            encode_int8,
            physical_page_map,
        )

        pages, page_size = 7, 4
        gen = torch.Generator().manual_seed(89423)
        for width, group, rotary in ((128, 128, 0), (448, 224, 64)):
            with self.subTest(width=width):
                rows = torch.randn(pages, page_size, 1, width, generator=gen).bfloat16()
                rows[..., 0] *= 2**-25
                codes, metadata = encode_int8(rows, group)
                reference = decode_int8(codes, metadata, group)
                # Values and metadata in the compressed KV record have a
                # larger row stride than their last dimension.
                record_width = 2 * rotary + width + metadata.shape[-1]
                storage = torch.zeros(
                    pages,
                    page_size,
                    1,
                    record_width + 2,
                    dtype=torch.uint8,
                    device="npu",
                )
                values = storage[..., 2 * rotary : 2 * rotary + width].view(torch.int8)
                data = storage[..., 2 * rotary + width : record_width]
                values.copy_(codes)
                data.copy_(metadata)
                rope = None
                if rotary:
                    rope = storage[..., : 2 * rotary].view(torch.bfloat16)
                    expected_rope = torch.randn(
                        pages, page_size, 1, rotary, generator=gen
                    ).bfloat16()
                    rope.copy_(expected_rope)
                    reference = torch.cat((reference, expected_rope), dim=-1)

                table = torch.zeros(2, 4, device="npu", dtype=torch.int32)
                lengths = torch.zeros(2, device="npu", dtype=torch.int32)

                def reconstruct():
                    physical, logical = physical_page_map(
                        table, page_size, lengths, fixed_shape=True, pool_pages=pages
                    )
                    decoded = decode_pages(
                        values, data, physical, group, SCALES_PER_GROUP, rope
                    )
                    return decoded, logical

                reconstruct()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output, remapped = reconstruct()

                for ids, counts in (
                    ([[4, 3, 99, -1], [4, 5, -1, -1]], [7, 5]),
                    ([[6, -1, -1, -1], [2, 1, 3, 99]], [3, 11]),
                    ([[1, 2, 99, -1], [0, -1, -1, -1]], [8, 0]),
                ):
                    table.copy_(torch.tensor(ids, dtype=torch.int32))
                    lengths.copy_(torch.tensor(counts, dtype=torch.int32))
                    graph.replay()
                    torch.npu.synchronize()
                    actual, mapping = output.cpu(), remapped.cpu()
                    self.assertEqual(actual.shape[0], pages)
                    for req, count in enumerate(counts):
                        for k in range((count + page_size - 1) // page_size):
                            torch.testing.assert_close(
                                actual[mapping[req, k]],
                                reference[ids[req][k]],
                                rtol=0,
                                atol=0,
                            )


if __name__ == "__main__":
    unittest.main()
