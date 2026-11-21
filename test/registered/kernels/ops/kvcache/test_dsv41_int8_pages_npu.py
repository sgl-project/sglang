"""Check captured DSv4.1 page reconstruction on changing logical and physical pages."""

import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=40, suite="extra-b-test-1-npu-a3")


class TestDSV41Int8Pages(CustomTestCase):
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
