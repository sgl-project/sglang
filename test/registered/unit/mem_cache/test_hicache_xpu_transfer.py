import unittest

import torch

from sglang.srt.hardware_backend.xpu.hicache import backup_to_host, load_to_device
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCacheXPUTransfer(unittest.TestCase):
    def test_indexed_round_trip(self):
        host = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3)
        device = torch.full_like(host, -1)
        host_indices = torch.tensor([6, 1, 4], dtype=torch.int64)
        device_indices = torch.tensor([0, 5, 3], dtype=torch.int64)

        load_to_device(
            host_tensors=[host],
            device_tensors=[device],
            host_indices=host_indices,
            device_indices=device_indices,
        )
        torch.testing.assert_close(
            device.index_select(0, device_indices), host.index_select(0, host_indices)
        )

        restored = torch.zeros_like(host)
        restored_indices = host_indices.flip(0)
        backup_to_host(
            device_tensors=[device],
            host_tensors=[restored],
            device_indices=device_indices,
            host_indices=restored_indices,
        )
        torch.testing.assert_close(
            restored.index_select(0, restored_indices),
            device.index_select(0, device_indices),
        )

    def test_mha_pair_and_empty_transfer(self):
        host_k = torch.randn(5, 2, 4, dtype=torch.bfloat16)
        host_v = torch.randn(5, 2, 4, dtype=torch.bfloat16)
        device_k = torch.zeros_like(host_k)
        device_v = torch.zeros_like(host_v)
        indices = torch.tensor([4, 0, 2], dtype=torch.int64)

        load_to_device(
            host_tensors=[host_k, host_v],
            device_tensors=[device_k, device_v],
            host_indices=indices,
            device_indices=indices.flip(0),
        )
        torch.testing.assert_close(
            device_k.index_select(0, indices.flip(0)), host_k.index_select(0, indices)
        )
        torch.testing.assert_close(
            device_v.index_select(0, indices.flip(0)), host_v.index_select(0, indices)
        )

        load_to_device(
            host_tensors=[host_k],
            device_tensors=[device_k],
            host_indices=torch.empty(0, dtype=torch.int64),
            device_indices=torch.empty(0, dtype=torch.int64),
        )

    def test_rejects_mismatched_inputs(self):
        tensor = torch.empty(2, 2)
        with self.assertRaisesRegex(ValueError, "same length"):
            load_to_device(
                host_tensors=[tensor],
                device_tensors=[],
                host_indices=torch.tensor([0]),
                device_indices=torch.tensor([0]),
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            backup_to_host(
                device_tensors=[tensor],
                host_tensors=[tensor],
                device_indices=torch.tensor([0]),
                host_indices=torch.tensor([0, 1]),
            )

    def test_xpu_uses_pytorch_pinned_allocator(self):
        self.assertIs(ALLOC_MEMORY_FUNCS["xpu"], alloc_with_pin_memory)

    def test_xpu_controller_preserves_host_and_device_indices(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.io_backend = "xpu"
        host_indices = torch.tensor([3, 1], dtype=torch.int64)
        device_indices = torch.tensor([5, 2], dtype=torch.int64)

        moved_host_indices, moved_device_indices = controller.move_indices(
            host_indices, device_indices
        )

        self.assertIs(moved_host_indices, host_indices)
        self.assertIs(moved_device_indices, device_indices)


if __name__ == "__main__":
    unittest.main()
