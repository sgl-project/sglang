from __future__ import annotations

import unittest
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

if sys.platform == "win32" and "sglang" not in sys.modules:
    sglang_stub = types.ModuleType("sglang")
    sglang_stub.__path__ = [
        str(Path(__file__).resolve().parents[3] / "python" / "sglang")
    ]
    sys.modules["sglang"] = sglang_stub

    verify_stub = types.ModuleType("sglang.jit_kernel.kv_canary.verify")
    verify_stub.RealKvSource = object
    sys.modules["sglang.jit_kernel.kv_canary.verify"] = verify_stub

    hicache_storage_stub = types.ModuleType("sglang.srt.mem_cache.hicache_storage")

    class _PoolName(str, Enum):
        KV = "kv"
        SWA = "swa"

    @dataclass
    class _PoolTransfer:
        name: _PoolName
        host_indices: torch.Tensor | None = None
        device_indices: torch.Tensor | None = None

    hicache_storage_stub.PoolName = _PoolName
    hicache_storage_stub.PoolTransfer = _PoolTransfer
    sys.modules["sglang.srt.mem_cache.hicache_storage"] = hicache_storage_stub

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.hicache.bridge import CanaryHiCacheBridge
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer


def _make_group(*, kind: PoolKind, num_slots: int) -> CanaryBufferGroup:
    buffers = []
    for offset in range(4):
        values = torch.arange(num_slots * 32, dtype=torch.int64).reshape(num_slots, 32)
        buffers.append(((values + offset * 17) % 251).to(torch.uint8))
    return CanaryBufferGroup(
        kind=kind,
        k_head=buffers[0],
        k_tail=buffers[1],
        v_head=buffers[2],
        v_tail=buffers[3],
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
        kv_token_id_vs_position_offset=0,
    )


class TestCanaryHiCacheBridge(unittest.TestCase):
    def test_factory_uses_controller_host_pool_size(self) -> None:
        group = _make_group(kind=PoolKind.FULL, num_slots=8)

        class FakeHostPool:
            size = 13
            pin_memory = False

        class FakeController:
            enable_storage = False
            mem_pool_host = FakeHostPool()

        bridge = CanaryHiCacheBridge.from_cache_controller(
            buffer_groups=(group,), cache_controller=FakeController()
        )
        original = group.k_head[3].clone()
        bridge.backup(
            host_indices=torch.tensor([11]),
            device_indices=torch.tensor([3]),
            io_backend="direct",
        )
        group.k_head[6].zero_()
        bridge.restore(
            host_indices=torch.tensor([11]),
            device_indices=torch.tensor([6]),
            io_backend="direct",
        )
        torch.testing.assert_close(group.k_head[6], original)

    def test_factory_rejects_l3_storage(self) -> None:
        group = _make_group(kind=PoolKind.FULL, num_slots=8)

        class FakeController:
            enable_storage = True

        with self.assertRaisesRegex(NotImplementedError, "L3 storage"):
            CanaryHiCacheBridge.from_cache_controller(
                buffer_groups=(group,), cache_controller=FakeController()
            )

    def test_full_group_round_trip_preserves_index_mapping(self) -> None:
        group = _make_group(kind=PoolKind.FULL, num_slots=8)
        bridge = CanaryHiCacheBridge(
            buffer_groups=(group,),
            host_sizes={PoolKind.FULL: 12},
            pin_memory=False,
        )
        device_buffers = (group.k_head, group.k_tail, group.v_head, group.v_tail)
        original_slot_1 = tuple(buffer[1].clone() for buffer in device_buffers)
        original_slot_4 = tuple(buffer[4].clone() for buffer in device_buffers)

        bridge.backup(
            host_indices=torch.tensor([7, 2]),
            device_indices=torch.tensor([1, 4]),
            io_backend="direct",
        )
        group.k_head[[5, 6]].zero_()
        group.k_tail[[5, 6]].zero_()
        group.v_head[[5, 6]].zero_()
        group.v_tail[[5, 6]].zero_()
        bridge.restore(
            host_indices=torch.tensor([2, 7]),
            device_indices=torch.tensor([5, 6]),
            io_backend="direct",
        )

        for buffer, expected_4, expected_1 in zip(
            device_buffers, original_slot_4, original_slot_1
        ):
            torch.testing.assert_close(buffer[5], expected_4)
            torch.testing.assert_close(buffer[6], expected_1)

    def test_swa_group_uses_pool_transfer_indices(self) -> None:
        full = _make_group(kind=PoolKind.FULL, num_slots=8)
        swa = _make_group(kind=PoolKind.SWA, num_slots=6)
        bridge = CanaryHiCacheBridge(
            buffer_groups=(full, swa),
            host_sizes={PoolKind.FULL: 12, PoolKind.SWA: 10},
            pin_memory=False,
        )
        original_swa_slot = swa.v_tail[3].clone()
        pool_transfers = [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.tensor([8]),
                device_indices=torch.tensor([3]),
            )
        ]

        bridge.backup(
            host_indices=torch.tensor([4]),
            device_indices=torch.tensor([2]),
            pool_transfers=pool_transfers,
            io_backend="direct",
        )
        swa.v_tail[5].zero_()
        restore_transfers = [
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=torch.tensor([8]),
                device_indices=torch.tensor([5]),
            )
        ]
        bridge.restore(
            host_indices=torch.tensor([4]),
            device_indices=torch.tensor([6]),
            pool_transfers=restore_transfers,
            io_backend="direct",
        )

        torch.testing.assert_close(swa.v_tail[5], original_swa_slot)

    def test_missing_swa_transfer_is_rejected(self) -> None:
        full = _make_group(kind=PoolKind.FULL, num_slots=8)
        swa = _make_group(kind=PoolKind.SWA, num_slots=6)
        bridge = CanaryHiCacheBridge(
            buffer_groups=(full, swa),
            host_sizes={PoolKind.FULL: 12, PoolKind.SWA: 10},
            pin_memory=False,
        )

        with self.assertRaisesRegex(RuntimeError, "missing SWA PoolTransfer"):
            bridge.backup(
                host_indices=torch.tensor([4]),
                device_indices=torch.tensor([2]),
                pool_transfers=None,
                io_backend="direct",
            )


if __name__ == "__main__":
    unittest.main()
