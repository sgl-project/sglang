import concurrent.futures
import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch_npu  # noqa: F401

from sglang.srt.disaggregation.common.dcp_pack import try_pack_dcp_src
from sglang.srt.hardware_backend.npu.dcp_pack import (
    NPUDCPPackBuffer,
    init_npu_dcp_pack_buffers,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=20, suite="extra-b-test-1-npu-a3")


class TestNpuDcpPack(CustomTestCase):
    def setUp(self):
        torch.npu.set_device(0)

    def test_heterogeneous_rows_strides_tail_and_pack_offset(self):
        widths, strides = [1024, 128, 656, 1152], [1040, 160, 688, 1152]
        sources = [
            torch.arange(1200 * stride, device="npu", dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .view(1200, stride)
            for stride in strides
        ]
        torch.npu.synchronize()
        pack = NPUDCPPackBuffer(128 + 529 * sum(widths), 0)
        for count in (1, 17, 257, 529):
            with self.subTest(count=count):
                rows = np.arange(count, dtype=np.int64)[::-1].copy() * 2 + 3
                pack.buffer.fill_(165)
                pointers, packed_rows = try_pack_dcp_src(
                    pack_buffer=pack,
                    kv_data_ptrs=[x.data_ptr() for x in sources],
                    src_token_indices=rows,
                    token_item_lens=widths,
                    src_token_item_lens=strides,
                    pack_offset_bytes=64,
                )
                np.testing.assert_array_equal(packed_rows, np.arange(count))
                offset = 64
                for ptr, source, width in zip(pointers, sources, widths):
                    self.assertEqual(ptr, pack.get_ptr() + offset)
                    size = count * width
                    expected = source[torch.tensor(rows, device="npu"), :width]
                    got = pack.buffer[offset : offset + size].view(count, width)
                    torch.testing.assert_close(
                        got.cpu(), expected.cpu(), rtol=0, atol=0
                    )
                    offset += size
                self.assertTrue(bool(torch.all(pack.buffer[:64] == 165)))
                self.assertTrue(bool(torch.all(pack.buffer[offset:] == 165)))
        replacement = [torch.full_like(source, 113) for source in sources]
        try_pack_dcp_src(
            pack_buffer=pack,
            kv_data_ptrs=[x.data_ptr() for x in replacement],
            src_token_indices=np.arange(17, dtype=np.int64),
            token_item_lens=widths,
            src_token_item_lens=strides,
            pack_offset_bytes=64,
        )
        self.assertTrue(bool(torch.all(pack.buffer[64 : 64 + 17 * sum(widths)] == 113)))

    def test_send_thread_waits_for_kv_and_orders_index_copy(self):
        source = torch.zeros((1024, 128), dtype=torch.uint8, device="npu")
        values = (
            torch.arange(source.numel(), dtype=torch.int32, device="npu")
            .remainder(251)
            .to(torch.uint8)
            .view_as(source)
        )
        pack = NPUDCPPackBuffer(529 * 128, 0)
        producer = torch.npu.Stream()
        matrix = torch.ones((256, 256), device="npu")
        producer.wait_stream(torch.npu.current_stream())
        barrier = threading.Barrier(2)

        def send(ready, rows):
            torch.npu.set_device(0)
            # Same boundary as TransferKVChunk.wait_event in the real sender.
            ready.synchronize()
            barrier.wait(timeout=30)
            try_pack_dcp_src(
                pack_buffer=pack,
                kv_data_ptrs=[source.data_ptr()],
                src_token_indices=rows,
                token_item_lens=[128],
            )
            return pack.buffer[: rows.size * 128].view(rows.size, 128).cpu()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for iteration, count in enumerate((17, 529, 257)):
                rows = np.arange(count, dtype=np.int64)[::-1].copy()
                current_values = (
                    (values.to(torch.int32) + iteration * 17)
                    .remainder(251)
                    .to(torch.uint8)
                )
                producer.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(producer):
                    source.copy_(current_values)
                    ready = torch.npu.Event()
                    ready.record()
                future = executor.submit(send, ready, rows)
                barrier.wait(timeout=30)
                with torch.npu.stream(producer):
                    for _ in range(8):
                        output = matrix @ matrix
                actual = future.result(timeout=60)
                expected = current_values[torch.tensor(rows, device="npu")].cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                producer.synchronize()
                torch.testing.assert_close(output.cpu(), torch.full((256, 256), 256.0))

    def test_registers_target_capacity_and_keeps_small_buffer_fallback(self):
        registrations = []
        args = SimpleNamespace(
            kv_item_lens=[128 * 1024, 128 * 128, 128 * 1024],
            page_size=128,
            num_draft_entries=1,
            gpu_id=0,
        )
        buffers = init_npu_dcp_pack_buffers(
            lambda *region: registrations.append(region),
            args,
            2,
            2,
            257,
            device_type="npu",
            include_draft=True,
        )
        self.assertEqual([x.get_size() for x in buffers], [258 * 1152] * 2)
        self.assertEqual(registrations, [(x.get_ptr(), x.get_size()) for x in buffers])
        small = NPUDCPPackBuffer(16, 0)
        small.buffer.fill_(165)
        self.assertIsNone(
            try_pack_dcp_src(
                pack_buffer=small,
                kv_data_ptrs=[0],
                src_token_indices=np.array([0, 1], dtype=np.int64),
                token_item_lens=[1024],
            )
        )
        self.assertTrue(bool(torch.all(small.buffer == 165)))


if __name__ == "__main__":
    unittest.main()
