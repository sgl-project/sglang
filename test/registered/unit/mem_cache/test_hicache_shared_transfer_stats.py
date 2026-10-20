"""Shared D2H metrics count submitted pools without changing logical ACKs."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mixed_controller(*, owner):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mla_dedup = SimpleNamespace(
        is_shared=True,
        producer_stream=None,
        is_dummy_rank=False,
        validate_host_rows=mock.Mock(),
        synchronize_error=mock.Mock(),
    )
    pools = {
        PoolName.KV: SimpleNamespace(
            allocator=SimpleNamespace(is_shared=True, is_owner=owner),
            size_per_token=16,
        ),
        PoolName.INDEXER: SimpleNamespace(
            allocator=SimpleNamespace(is_shared=True, is_owner=owner),
            size_per_token=4,
        ),
        PoolName.MAMBA: SimpleNamespace(allocator=object(), size_per_token=64),
    }
    entries = {
        name: PoolEntry(name, host, object(), lambda layer: layer)
        for name, host in pools.items()
    }
    controller.mem_pool_host = SimpleNamespace(
        anchor_entry=entries[PoolName.KV],
        entry_map=entries,
    )
    controller.mem_pool_device_allocator = mock.Mock()
    controller._move_write_operation = lambda op: (
        op.host_indices,
        op.device_indices,
        op.pool_transfers,
    )
    finish = SimpleNamespace(synchronize=mock.Mock(), query=lambda: True)
    completion = TransferCompletion(object(), finish, False)
    controller.l2_transfer_engine = mock.Mock()
    controller.l2_transfer_engine.submit_device_to_host.return_value = completion
    controller.ack_write_queue = []
    op = CacheOperation(
        torch.arange(128),
        torch.arange(128),
        node_id=99,
        pool_transfers=[
            PoolTransfer(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                host_indices=torch.arange(128),
                device_indices=torch.arange(128),
            ),
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.tensor([7]),
                device_indices=torch.tensor([3]),
            ),
        ],
    )
    controller.write_queue = [op]
    return controller, op


class TestSharedTransferStats(CustomTestCase):
    def test_owner_d2h_counts_shared_kv_indexer_and_private_state(self):
        controller, _ = _mixed_controller(owner=True)
        controller.start_writing()
        ack = controller.ack_write_queue[0]
        self.assertEqual(ack.num_bytes, 128 * 16 + 128 * 4 + 64)
        self.assertEqual(ack.num_tokens_by_pool, {"kv": 128, "mamba": 1})
        self.assertEqual(ack.num_tokens, 128)
        self.assertEqual(ack.node_ids, [99])
        self.assertEqual(
            len(controller.l2_transfer_engine.submit_device_to_host.call_args.args[0]),
            3,
        )

    def test_peer_d2h_counts_only_private_state_but_keeps_logical_ack(self):
        controller, _ = _mixed_controller(owner=False)
        controller.start_writing()
        ack = controller.ack_write_queue[0]
        self.assertEqual(ack.num_bytes, 64)
        self.assertEqual(ack.num_tokens_by_pool, {"mamba": 1})
        self.assertEqual(ack.num_tokens, 128)
        self.assertEqual(ack.node_ids, [99])
        self.assertEqual(
            len(controller.l2_transfer_engine.submit_device_to_host.call_args.args[0]),
            1,
        )

    def test_h2d_unfiltered_counts_all_pools_on_every_reader(self):
        for owner in (True, False):
            with self.subTest(owner=owner):
                controller, op = _mixed_controller(owner=owner)
                self.assertEqual(
                    controller._transfer_num_bytes(op), 128 * 16 + 128 * 4 + 64
                )
                self.assertEqual(
                    controller._num_tokens_by_pool(op), {"kv": 128, "mamba": 1}
                )
                self.assertEqual(controller._transfer_num_bytes(op, transfers=[]), 0)
                self.assertEqual(controller._num_tokens_by_pool(op, transfers=[]), {})

    def test_flat_mla_peer_empty_d2h_has_zero_physical_volume(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.mem_pool_host = SimpleNamespace(size_per_token=16)
        op = CacheOperation(torch.arange(128), torch.arange(128), node_id=99)
        self.assertEqual(controller._transfer_num_bytes(op), 128 * 16)
        self.assertEqual(controller._num_tokens_by_pool(op), {"kv": 128})
        self.assertEqual(controller._transfer_num_bytes(op, transfers=[]), 0)
        self.assertEqual(controller._num_tokens_by_pool(op, transfers=[]), {})


if __name__ == "__main__":
    unittest.main()
