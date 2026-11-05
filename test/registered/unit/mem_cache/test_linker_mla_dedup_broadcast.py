"""CPU regression tests for packed physical layers in linker MLA broadcasts."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    _build_dsa_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _broadcaster(layer_mapping, *, components=1):
    # Dispatch-only tests avoid constructing a real pool or process group.
    broadcaster = LinkerMLADedupBroadcaster.__new__(LinkerMLADedupBroadcaster)
    broadcaster.pools = {"kv": SimpleNamespace(layer_mapping=layer_mapping)}
    broadcaster.buffers = {
        "kv": [
            [
                torch.empty((4, 128), dtype=torch.uint8),
                torch.empty((4, 256), dtype=torch.uint8),
                torch.empty((4, 512), dtype=torch.uint8),
            ]
            for _ in range(components)
        ]
    }
    broadcaster.kv_staging = torch.empty(4096, dtype=torch.uint8)
    calls = []

    def record(buffers, staging, indices, elem, layer_id):
        calls.append((elem, layer_id))

    broadcaster._bcast_layer = record
    return broadcaster, calls


def _dsa_pool_group(*, source):
    def buffers(rows, width, layers, offset):
        result = []
        for layer in range(layers):
            values = torch.arange(rows * width) + offset + layer * 31
            result.append(
                values.remainder(251).to(torch.uint8).reshape(rows, 1, width)
                if source
                else torch.full((rows, 1, width), 255, dtype=torch.uint8)
            )
        return result

    target = SimpleNamespace(
        page_size=2,
        layer_num=2,
        kv_buffer=buffers(10, 4, 2, 1),
        index_k_with_scale_buffer=buffers(5, 6, 2, 71),
    )
    draft = SimpleNamespace(
        page_size=2,
        kv_buffer=buffers(10, 4, 1, 131),
        index_k_with_scale_buffer=buffers(5, 6, 1, 191),
    )
    return _build_dsa_device_pool_group(target, 2, (draft,))


def _data_broadcaster(pool_group, *, source):
    # Use the real constructor, preparation and _bcast_layer. Only rank discovery
    # and the communication primitive are mocked; no NCCL group is initialized.
    with (
        envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.override(1),
        mock.patch(
            "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
            return_value=(0, 2),
        ),
    ):
        broadcaster = LinkerMLADedupBroadcaster(
            pool_group, group=object(), src_global_rank=0
        )
    broadcaster.is_src = source
    return broadcaster


class TestLinkerMLADedupBroadcast(unittest.TestCase):
    def test_int_layer_mapping_broadcasts_one_physical_layer(self):
        broadcaster, calls = _broadcaster({0: 1})
        broadcaster.broadcast_loaded_layer(0, {"kv": (torch.tensor([1, 2]), None)})
        self.assertEqual(calls, [(256, 1)])

    def test_packed_draft_tuple_broadcasts_every_physical_layer(self):
        broadcaster, calls = _broadcaster({0: (1, 2)})
        broadcaster.broadcast_loaded_layer(0, {"kv": (torch.tensor([1, 2]), None)})
        self.assertEqual(calls, [(256, 1), (512, 2)])

    def test_unmapped_layer_broadcasts_nothing(self):
        broadcaster, calls = _broadcaster({1: 0})
        broadcaster.broadcast_loaded_layer(0, {"kv": (torch.tensor([1, 2]), None)})
        self.assertEqual(calls, [])

    def test_pool_missing_from_prepared_is_skipped(self):
        broadcaster, calls = _broadcaster({0: (0, 1)})
        broadcaster.broadcast_loaded_layer(0, {})
        self.assertEqual(calls, [])

    def test_physical_layer_zero_is_not_skipped(self):
        broadcaster, calls = _broadcaster({0: 0})
        broadcaster.broadcast_loaded_layer(0, {"kv": (torch.tensor([1]), None)})
        self.assertEqual(calls, [(128, 0)])

    def test_sequence_mapping_broadcasts_every_physical_layer(self):
        broadcaster, calls = _broadcaster({0: [0, 1, 2]})
        broadcaster.broadcast_loaded_layer(0, {"kv": (torch.tensor([1]), None)})
        self.assertEqual(calls, [(128, 0), (256, 1), (512, 2)])

    def test_every_component_broadcasts_every_mapped_layer(self):
        broadcaster, calls = _broadcaster({0: (1, 2)}, components=2)
        broadcaster.broadcast_loaded_layer(0, {"kv": (torch.tensor([1]), None)})
        self.assertEqual(calls, [(256, 1), (512, 2)] * 2)

    def test_dsa_prepare_preserves_rank_local_token_and_page_order(self):
        broadcaster = _data_broadcaster(_dsa_pool_group(source=True), source=True)
        slots = torch.tensor([6, 7, 0, 1])
        prepared = broadcaster.prepare_broadcast(
            {PoolName.KV: slots, PoolName.INDEXER: slots}, None
        )
        torch.testing.assert_close(prepared[PoolName.KV][0], slots)
        torch.testing.assert_close(prepared[PoolName.INDEXER][0], torch.tensor([3, 0]))

    def test_dsa_broadcast_uses_each_pools_indices_and_row_width(self):
        broadcaster = _data_broadcaster(_dsa_pool_group(source=True), source=True)
        slots = torch.tensor([6, 7, 0, 1])
        prepared = broadcaster.prepare_broadcast(
            {PoolName.KV: slots, PoolName.INDEXER: slots}, None
        )
        with mock.patch.object(broadcaster, "_bcast_layer") as broadcast:
            broadcaster.broadcast_loaded_layer(0, prepared)

        self.assertEqual(broadcast.call_count, 4)
        for call, (name, width, layer) in zip(
            broadcast.call_args_list,
            [
                (PoolName.KV, 4, 0),
                (PoolName.KV, 4, 2),
                (PoolName.INDEXER, 6, 0),
                (PoolName.INDEXER, 6, 2),
            ],
        ):
            self.assertIs(call.args[0], broadcaster.buffers[name][0])
            self.assertIs(call.args[1], broadcaster.kv_staging)
            self.assertIs(call.args[2], prepared[name][0])
            self.assertEqual(call.args[3:], (width, layer))

    def _assert_data_round_trip(self, layer_id):
        source = _data_broadcaster(_dsa_pool_group(source=True), source=True)
        receiver = _data_broadcaster(_dsa_pool_group(source=False), source=False)
        source_slots = torch.tensor([6, 7, 0, 1])
        receiver_slots = torch.tensor([2, 3, 8, 9])
        source_prepared = source.prepare_broadcast(
            {name: source_slots for name in source.pools}, None
        )
        receiver_prepared = receiver.prepare_broadcast(
            {name: receiver_slots for name in receiver.pools}, None
        )
        snapshots = {
            name: [buf.clone() for buf in components[0]]
            for name, components in receiver.buffers.items()
        }
        payloads = deque()

        def send(tensor, *, src, group):
            self.assertEqual(src, source.src_global_rank)
            self.assertIs(group, source.group)
            payloads.append(tensor.clone())

        def receive(tensor, *, src, group):
            self.assertEqual(src, receiver.src_global_rank)
            self.assertIs(group, receiver.group)
            payload = payloads.popleft()
            self.assertEqual(tensor.shape, payload.shape)
            tensor.copy_(payload)

        # Eight staging bytes hold two KV rows but only one indexer row. Both
        # pools need two chunks per physical layer, reusing the same staging.
        self.assertEqual(source.kv_staging.numel(), 8)
        with mock.patch.object(torch.distributed, "broadcast", side_effect=send):
            source.broadcast_loaded_layer(layer_id, source_prepared)
        chunk_count = len(payloads)
        with mock.patch.object(torch.distributed, "broadcast", side_effect=receive):
            receiver.broadcast_loaded_layer(layer_id, receiver_prepared)
        self.assertEqual(len(payloads), 0)

        for name, pool in source.pools.items():
            mapped = pool.layer_mapping[layer_id]
            layers = [mapped] if isinstance(mapped, int) else list(mapped)
            for layer, received in enumerate(receiver.buffers[name][0]):
                expected = snapshots[name][layer]
                if layer in layers:
                    expected.index_copy_(
                        0,
                        receiver_prepared[name][0],
                        source.buffers[name][0][layer].index_select(
                            0, source_prepared[name][0]
                        ),
                    )
                # Compare the entire buffer: non-selected rows and unmapped
                # layers must retain their sentinel values.
                torch.testing.assert_close(received, expected, rtol=0, atol=0)
        self.assertEqual(chunk_count, 8 if layer_id == 0 else 4)

    def test_packed_dsa_data_round_trip_with_rank_local_slots_and_chunking(self):
        self._assert_data_round_trip(0)

    def test_int_dsa_data_round_trip_with_rank_local_slots_and_chunking(self):
        self._assert_data_round_trip(1)

    def test_empty_indices_issue_no_collectives_and_leave_buffers_untouched(self):
        broadcaster = _data_broadcaster(_dsa_pool_group(source=False), source=False)
        empty = torch.empty(0, dtype=torch.int64)
        prepared = broadcaster.prepare_broadcast(
            {name: empty for name in broadcaster.pools}, None
        )
        with mock.patch.object(torch.distributed, "broadcast") as broadcast:
            broadcaster.broadcast_loaded_layer(0, prepared)
        broadcast.assert_not_called()
        for components in broadcaster.buffers.values():
            for buffer in components[0]:
                self.assertTrue(torch.all(buffer == 255).item())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
