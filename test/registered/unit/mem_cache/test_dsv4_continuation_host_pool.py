import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestDSV4ContinuationHostPool(unittest.TestCase):
    def test_page_first_staging_respects_explicit_chunk(self):
        pool = object.__new__(DeepSeekV4PagedHostPool)
        pool.layout = "page_first"
        pool.item_bytes = 9_617_408
        pool.dtype = torch.uint8
        pool.num_host_pages = 256
        pool.layer_num = 1
        pool.gpu_device = "cpu"
        pool.write_back_staging_page_chunk = 1

        with (
            patch("sglang.srt.mem_cache.memory_pool_host._is_cuda", True),
            patch("sglang.srt.mem_cache.memory_pool_host._is_npu", False),
            patch("sglang.srt.mem_cache.memory_pool_host._is_xpu", False),
            patch("sglang.srt.mem_cache.memory_pool_host._is_mps", False),
            patch(
                "sglang.srt.mem_cache.memory_pool_host.can_use_write_back_jit_kernel",
                return_value=True,
            ),
        ):
            pool._init_write_back_staging_buffers()

        self.assertEqual(pool.staging_buffer.shape, (1, 1, 9_617_408))

    def test_direct_round_trip_maps_host_zero_to_device_one(self):
        payload_bytes = 64
        device_buffer = torch.zeros(
            (3, payload_bytes), dtype=torch.uint8, device="cuda"
        )
        device_buffer[1] = torch.arange(payload_bytes, dtype=torch.uint8, device="cuda")
        device_pool = MagicMock()
        device_pool.wait_ready_indices = MagicMock()
        device_pool.record_ready_indices = MagicMock()
        host_pool = DeepSeekV4PagedHostPool(
            pool_name="dsv4_continuation",
            device_buffers=[device_buffer],
            item_bytes=payload_bytes,
            num_host_pages=2,
            slot_page_size=1,
            layout="layer_first",
            pin_memory=True,
        )
        host_indices = torch.tensor([0], dtype=torch.int64)
        device_indices = torch.tensor([1], dtype=torch.int64, device="cuda")

        host_pool.backup_from_device_all_layer(
            device_pool,
            host_indices,
            device_indices,
            io_backend="direct",
        )
        torch.cuda.synchronize()
        device_buffer[1].zero_()
        host_pool.load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id=0,
            io_backend="direct",
        )
        torch.cuda.synchronize()

        self.assertEqual(device_buffer[1].cpu().tolist(), list(range(payload_bytes)))
        self.assertEqual(device_pool.wait_ready_indices.call_count, 2)
        self.assertEqual(device_pool.record_ready_indices.call_count, 2)


if __name__ == "__main__":
    unittest.main()
