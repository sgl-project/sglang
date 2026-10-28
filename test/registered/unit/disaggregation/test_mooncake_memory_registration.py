"""Keep fused K/V offsets intact in native HIP IPC registrations."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_manager():
    storage = torch.zeros((2, 64), dtype=torch.uint8)
    base = storage.data_ptr()
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[base, base + 64],
        kv_data_lens=[64, 64],
        aux_data_ptrs=[],
        aux_data_lens=[],
        state_data_ptrs=[],
        state_data_lens=[],
    )
    snapshot = [{"address": base, "total_size": 128, "is_expandable": False}]
    return manager, storage, snapshot


class NativeHipMapping:
    """Model the pinned Mooncake relocation after HIP IPC opens a slab base.

    Its registered descriptor address is subtracted from the requested address;
    HIP's mapped address is the allocation base, including for an interior view.
    This arithmetic model needs no GPU and does not claim to execute HIP IPC.
    """

    def __init__(self, size):
        self.data = bytearray(size)
        self.regions = []

    def batch_register(self, ptrs, lengths):
        self.regions = list(zip(ptrs, lengths))
        return 0

    def write(self, destination, payload):
        for address, length in self.regions:
            if (
                address <= destination
                and destination + len(payload) <= address + length
            ):
                offset = destination - address
                self.data[offset : offset + len(payload)] = payload
                return
        raise AssertionError("Destination is not registered")


class TestMooncakeMemoryRegistration(unittest.TestCase):
    def test_fused_kv_transfer_does_not_overwrite_keys_with_values(self):
        manager, storage, snapshot = make_manager()
        manager.engine = NativeHipMapping(128)
        original_ptrs = list(manager.kv_args.kv_data_ptrs)

        with (
            patch.object(torch.version, "hip", "7.2.0"),
            patch.object(torch.cuda.memory, "memory_snapshot", return_value=snapshot),
        ):
            manager.register_buffer_to_engine()

        manager.engine.write(original_ptrs[0], bytes([17]) * 64)
        manager.engine.write(original_ptrs[1], bytes([34]) * 64)

        self.assertEqual(manager.engine.data[:64], bytes([17]) * 64)
        self.assertEqual(manager.engine.data[64:], bytes([34]) * 64)
        self.assertEqual(manager.kv_args.kv_data_ptrs, original_ptrs)
        self.assertEqual(manager.kv_args.kv_data_lens, [64, 64])

    def test_cuda_keeps_subregions_and_does_not_read_allocator_snapshot(self):
        manager, storage, _ = make_manager()
        base = storage.data_ptr()
        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda.memory, "memory_snapshot") as snapshot,
        ):
            regions = manager._registerable_regions()
        self.assertEqual(regions, [(base, 64), (base + 64, 64)])
        snapshot.assert_not_called()

    def test_expandable_hip_segments_keep_virtual_subregions(self):
        manager, storage, snapshot = make_manager()
        base = storage.data_ptr()
        snapshot[0]["is_expandable"] = True
        with (
            patch.object(torch.version, "hip", "7.2.0"),
            patch.object(torch.cuda.memory, "memory_snapshot", return_value=snapshot),
        ):
            regions = manager._registerable_regions()
        self.assertEqual(regions, [(base, 64), (base + 64, 64)])

    def test_host_aux_and_untracked_state_regions_remain_registered(self):
        manager, storage, snapshot = make_manager()
        base = storage.data_ptr()
        # Host aux and external state allocations are not in the GPU snapshot.
        manager.kv_args.aux_data_ptrs = [101]
        manager.kv_args.aux_data_lens = [32]
        manager.kv_args.state_data_ptrs = [[base + 64, 202, 303]]
        manager.kv_args.state_data_lens = [[64, 16, 0]]
        with (
            patch.object(torch.version, "hip", "7.2.0"),
            patch.object(torch.cuda.memory, "memory_snapshot", return_value=snapshot),
        ):
            regions = manager._registerable_regions()
        self.assertEqual(regions, [(base, 128), (101, 32), (202, 16)])

    def test_deregistration_uses_the_same_allocation_base(self):
        manager, storage, snapshot = make_manager()
        base = storage.data_ptr()
        manager.engine = Mock()
        with (
            patch.object(torch.version, "hip", "7.2.0"),
            patch.object(torch.cuda.memory, "memory_snapshot", return_value=snapshot),
        ):
            manager.register_buffer_to_engine()
            manager.deregister_buffer_to_engine()
        manager.engine.batch_register.assert_called_once_with([base], [128])
        manager.engine.batch_deregister.assert_called_once_with([base])


if __name__ == "__main__":
    unittest.main()
