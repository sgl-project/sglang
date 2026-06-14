import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.shared_mla_host_kv_cache import SharedMLATokenToKVPoolHost
from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")


class TestSharedMLAHost(unittest.TestCase):
    """Single-process (TP=1, owner) coverage of SharedMLATokenToKVPoolHost.

    Multi-rank behaviour (follower attach, NCCL host_value broadcast,
    writing_check sync) and NUMA interleave (Linux, >=2 nodes) are validated by
    end-to-end / micro-benchmarks, not here, since they need multiple processes.
    """

    PAGE_SIZE = 1
    LAYER_NUM = 2
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for SharedMLA host tests.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")
        self.host = None

    def tearDown(self):
        if self.host is not None:
            self.host.shutdown()
            self.host = None

    def _make_device_pool(self, size):
        return MLATokenToKVPool(
            size=size,
            page_size=self.PAGE_SIZE,
            dtype=torch.bfloat16,
            kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            layer_num=self.LAYER_NUM,
            device="cuda",
            enable_memory_saver=False,
        )

    def _make_host(self, device_pool, host_size=1):
        return SharedMLATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=2.0,
            host_size=host_size,
            page_size=self.PAGE_SIZE,
            layout="layer_first",
            tp_rank=0,
            tp_size=1,
        )

    def test_construction(self):
        device_pool = self._make_device_pool(size=256)
        self.host = self._make_host(device_pool)
        self.assertEqual(
            tuple(self.host.kv_buffer.shape),
            (self.LAYER_NUM, self.host.size, 1, self.host.kv_cache_dim),
        )
        self.assertGreater(self.host.mem_usage, 0)
        self.assertEqual(self.host.available_size(), self.host.size)

    def test_alloc_free_clear(self):
        device_pool = self._make_device_pool(size=256)
        self.host = self._make_host(device_pool)
        total = self.host.available_size()

        indices = self.host.alloc(8)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 8)
        self.assertEqual(self.host.available_size(), total - 8)

        self.host.free(indices)
        self.assertEqual(self.host.available_size(), total)

        self.host.clear()
        self.assertEqual(self.host.available_size(), self.host.size)

    def test_round_trip_transfer(self):
        size = 256
        device_pool = self._make_device_pool(size=size)
        self.host = self._make_host(device_pool)

        # Fill the device pool with known per-layer values.
        for layer_id in range(self.LAYER_NUM):
            buf = device_pool.kv_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=buf.dtype
            ).view_as(buf)
            buf.copy_(data + layer_id)

        n = 16
        host_indices = self.host.alloc(n)
        device_indices = torch.arange(n, device="cuda", dtype=torch.int64)

        # The transfer kernels need both index tensors on the GPU (the cache
        # controller moves host_indices to the device before calling these);
        # mirror that here.
        host_indices_dev = host_indices.to("cuda", non_blocking=True)

        # device -> shared slab
        self.host.backup_from_device_all_layer(
            device_pool, host_indices_dev, device_indices, io_backend="kernel"
        )

        # zero the device pool, then shared slab -> device
        for layer_id in range(self.LAYER_NUM):
            device_pool.kv_buffer[layer_id].zero_()
        for layer_id in range(self.LAYER_NUM):
            self.host.load_to_device_per_layer(
                device_pool,
                host_indices_dev,
                device_indices,
                layer_id,
                io_backend="kernel",
            )
        torch.cuda.synchronize()

        for layer_id in range(self.LAYER_NUM):
            buf = device_pool.kv_buffer[layer_id]
            expected = (
                torch.arange(buf.numel(), device=buf.device, dtype=buf.dtype).view_as(
                    buf
                )
                + layer_id
            )
            got = buf[device_indices]
            self.assertTrue(torch.equal(got, expected[device_indices]))

    def test_stale_slab_is_reclaimed(self):
        # Construct once, leak the slab (skip shutdown), then construct again
        # with the same computed name and assert it succeeds (stale unlinked)
        # rather than raising FileExistsError.
        device_pool = self._make_device_pool(size=256)
        first = self._make_host(device_pool)
        shm_name = first._shm_name
        # Drop the python handle without unlinking (simulate a crash).
        first._shm.close()
        first._shm = None

        self.host = self._make_host(device_pool)
        self.assertEqual(self.host._shm_name, shm_name)
        self.assertEqual(self.host.available_size(), self.host.size)

    def test_buffer_meta_lengths(self):
        device_pool = self._make_device_pool(size=256)
        self.host = self._make_host(device_pool)

        data_ptrs, data_lens, item_lens = self.host.get_contiguous_buf_infos()
        self.assertEqual(len(data_ptrs), self.LAYER_NUM)
        self.assertEqual(len(data_lens), self.LAYER_NUM)
        self.assertEqual(len(item_lens), self.LAYER_NUM)

        indices = self.host.alloc(self.PAGE_SIZE * 4)
        ptr_list, size_list = self.host.get_page_buffer_meta(indices)
        expected = (len(indices) // self.PAGE_SIZE) * self.LAYER_NUM
        self.assertEqual(len(ptr_list), expected)
        self.assertEqual(len(size_list), expected)


if __name__ == "__main__":
    unittest.main()
