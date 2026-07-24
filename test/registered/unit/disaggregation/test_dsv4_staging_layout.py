"""CPU tests for the experimental DeepSeek-V4 block staging layout."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.staging_buffer import (
    compute_address_block_staging_size,
    compute_dsv4_staging_upper_bound,
    split_address_blocks,
)
from sglang.srt.disaggregation.common.staging_handler import (
    handle_staging_req,
    handle_staging_rsp,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsv4StagingLayout(CustomTestCase):
    def test_split_address_blocks_preserves_addresses_and_bytes(self):
        fragments = split_address_blocks([(0x1000, 0x8000, 2050)])

        self.assertEqual(
            fragments,
            [
                (0x1000, 0x8000, 1024),
                (0x1400, 0x8400, 1024),
                (0x1800, 0x8800, 2),
            ],
        )

    def test_staging_size_is_aligned_and_includes_payload(self):
        blocks = [(0x1000, 0x8000, 2050), (0x2000, 0x9000, 17)]
        required, num_fragments = compute_address_block_staging_size(blocks)

        self.assertEqual(num_fragments, 4)
        self.assertEqual(required % 256, 0)
        self.assertGreaterEqual(required, 2050 + 17 + num_fragments * 3 * 8)

    def test_dsv4_upper_bound_covers_fully_fragmented_plan(self):
        num_tokens = 257
        c4_layers = 2
        c128_layers = 1
        dcp_size = 2
        c4_tokens = (num_tokens + 3) // 4
        c4_local = (c4_tokens + dcp_size - 1) // dcp_size
        c128_tokens = (num_tokens + 127) // 128
        c128_local = (c128_tokens + dcp_size - 1) // dcp_size

        blocks = []
        next_addr = 0x1000

        def append_records(count, segments, layers):
            nonlocal next_addr
            for _ in range(layers * count):
                for length in segments:
                    blocks.append((next_addr, next_addr + 0x100000, length))
                    next_addr += 0x2000

        append_records(c4_local, (576, 8), c4_layers)
        append_records(c4_tokens, (128, 4), c4_layers)
        append_records(c128_local, (576, 8), c128_layers)

        actual, _ = compute_address_block_staging_size(blocks)
        upper_bound = compute_dsv4_staging_upper_bound(
            num_tokens, c4_layers, c128_layers, dcp_size
        )
        self.assertLessEqual(actual, upper_bound)

    def test_staging_allocations_are_isolated_by_prefill_writer(self):
        class FakeAllocator:
            ALLOC_OVERSIZED = -2

            def __init__(self):
                self.total_size = 1 << 20
                self.next_offset = 0
                self.next_id = 0

            def assign(self, required):
                result = (self.next_id, self.next_offset, 0)
                self.next_id += 1
                self.next_offset += required
                return result

        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            dsv4_chunk_staging_infos={},
        )
        allocator = FakeAllocator()
        common = [b"STAGING_REQ", b"7", b"0", b"48", b"session", b"16384"]
        for writer in (b"0", b"2"):
            handle_staging_req(
                common + [writer],
                allocator,
                SimpleNamespace(),
                1,
                1,
                None,
                {7: receiver},
                {},
            )

        infos = receiver.dsv4_chunk_staging_infos
        self.assertEqual(set(infos), {(0, "0"), (0, "2")})
        self.assertNotEqual(infos[(0, "0")][1], infos[(0, "2")][1])

    def test_staging_response_is_filtered_by_prefill_writer(self):
        tinfo = SimpleNamespace(staging=None)
        transfer_infos = {7: {"session": tinfo}}
        response = [
            b"STAGING_RSP",
            b"7",
            b"0",
            b"4096",
            b"0",
            b"8192",
            b"session",
            b"2",
        ]

        handle_staging_rsp(response, transfer_infos, expected_writer_id="0")
        self.assertIsNone(tinfo.staging)
        handle_staging_rsp(response, transfer_infos, expected_writer_id="2")
        self.assertEqual(tinfo.staging.offsets[0], 4096)


if __name__ == "__main__":
    unittest.main()
