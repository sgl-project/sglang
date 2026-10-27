"""Real CUDA L2 copies: sharded GQA KV and unsharded GDN checkpoints."""

import unittest

import torch

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGqaDcpL2Copies(CustomTestCase):
    def test_kv_roundtrip_all_owners(self):
        for layout, io in (("page_first_direct", "direct"), ("page_first", "kernel")):
            for dtype in (torch.bfloat16, torch.float16):
                for size in (1, 2, 4):
                    for rank in range(size):
                        with self.subTest(
                            layout=layout, dtype=dtype, size=size, rank=rank
                        ):
                            self._kv_roundtrip(layout, io, size, rank, dtype)

    def _kv_roundtrip(self, layout, io, size, rank, dtype):
        pool = MHATokenToKVPool(
            size=64,
            page_size=4,
            dtype=dtype,
            head_num=2,
            head_dim=8,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        host = MHATokenToKVPoolHost(
            pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2,
            host_size=0,
            page_size=4 * size,
            layout=layout,
            device_capacity_tokens=pool.size,
            start_layer=pool.start_layer,
            end_layer=pool.end_layer,
            dcp_size=size,
            dcp_rank=rank,
        )
        try:
            buffers = [pool.get_key_buffer(i) for i in range(2)] + [
                pool.get_value_buffer(i) for i in range(2)
            ]
            original = []
            for i, buffer in enumerate(buffers):
                values = (
                    (
                        torch.arange(buffer.numel(), device="cuda").reshape(
                            buffer.shape
                        )
                        + rank * 19
                        + i * 7
                    )
                    % 127
                ).to(buffer.dtype)
                buffer.copy_(values)
                original.append(values.clone())
            page = 4 * size
            source = torch.cat(
                [torch.arange(p * page, (p + 1) * page) for p in (3, 1, 15)]
            )
            destination = torch.cat(
                [torch.arange(p * page, (p + 1) * page) for p in (0, 5, 7)]
            )
            host_ids = torch.cat(
                [
                    torch.arange(p * page, (p + 1) * page)
                    for p in (host.page_num - 1, 2, 9)
                ]
            )
            host.kv_buffer.zero_()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                write_host = host_ids
                if io == "kernel":
                    source, destination = source.cuda(), destination.cuda()
                    if not host.can_use_write_back_jit:
                        write_host = host_ids.cuda()
                host.backup_from_device_all_layer(pool, write_host, source, io)
                for buffer in buffers:
                    buffer.fill_(-1)
                read_host = host_ids.cuda() if io == "kernel" else host_ids
                for layer in range(2):
                    host.load_to_device_per_layer(
                        pool, read_host, destination, layer, io
                    )
            torch.cuda.synchronize()
            src = source[source % size == rank].cuda() // size
            dst = destination[destination % size == rank].cuda() // size
            for buffer, values in zip(buffers, original):
                expected = torch.full_like(buffer, -1)
                expected[dst] = values[src]
                torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
        finally:
            torch.cuda.synchronize()
            host.destroy()

    def test_gdn_checkpoint_ids_are_not_context_sharded(self):
        for layout, io in (("page_first_direct", "direct"), ("page_first", "kernel")):
            with self.subTest(layout=layout):
                temporal = torch.arange(
                    2 * 16 * 32, device="cuda", dtype=torch.float32
                ).reshape(2, 16, 4, 8)
                conv = torch.arange(
                    2 * 16 * 16, device="cuda", dtype=torch.bfloat16
                ).reshape(2, 16, 4, 4)
                originals = [temporal.clone(), conv.clone()]
                host = MambaPoolHost(
                    MambaStateBufferInfo(temporal=temporal, conv=(conv,)),
                    host_to_device_ratio=2,
                    host_size=0,
                    device_capacity=16,
                    layout=layout,
                )
                try:
                    src = torch.tensor([1, 3, 8, 15], dtype=torch.int64)
                    dst = torch.tensor([4, 7, 10, 12], dtype=torch.int64)
                    host_ids = torch.tensor([2, 9, 20, 32], dtype=torch.int64)
                    if io == "kernel":
                        src, dst, host_ids = (x.cuda() for x in (src, dst, host_ids))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        host.backup_from_device_all_layer(None, host_ids, src, io)
                        temporal.fill_(-1)
                        conv.fill_(-1)
                        for layer in range(2):
                            host.load_to_device_per_layer(
                                None, host_ids, dst, layer, io
                            )
                    torch.cuda.synchronize()
                    self.assertEqual(host.dcp_size, 1)
                    for buffer, original in zip((temporal, conv), originals):
                        expected = torch.full_like(buffer, -1)
                        expected[:, dst] = original[:, src]
                        torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
                finally:
                    host.destroy()


if __name__ == "__main__":
    unittest.main()
