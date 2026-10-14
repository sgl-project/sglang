"""Unit tests for the NPU IPC multimodal feature pool bookkeeping.

The Ascend IPC calls themselves need hardware; everything tested here is the
lease/acknowledgement state machine, which runs the same way on CPU.
"""

import os
import pickle
import threading
import unittest
from contextlib import suppress
from unittest.mock import patch

import torch

from sglang.srt.multimodal.transport import npu_ipc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

POOL_SIZE = 1 << 20
LEASE_BYTES = 4096
TP_SIZE = 4


class TestNpuIpcPoolBookkeeping(CustomTestCase):
    def setUp(self):
        self.pool = self._make_pool()
        self._proxies = []

    def tearDown(self):
        # Consumer mappings go first: unlinking a segment that is still mapped
        # fails on some platforms.
        for proxy in self._proxies:
            shm = getattr(proxy, "_sync_buffer", None)
            if shm is not None:
                with suppress(Exception):
                    shm.close()
                proxy._sync_buffer = None
                proxy._sync_buffer_data = None
        buffers = list(self.pool.sync_buffer_list)
        for chunk in self.pool.available_chunks + list(
            self.pool.occupied_chunks.values()
        ):
            buffers.append(chunk.sync_buffer)
        for buffer in buffers:
            with suppress(Exception):
                buffer.close()

    @staticmethod
    def _make_pool(tp_size=TP_SIZE) -> npu_ipc.MmItemMemoryPool:
        """Build a pool over host memory; only the device allocation is skipped."""
        pool = npu_ipc.MmItemMemoryPool.__new__(npu_ipc.MmItemMemoryPool)
        pool.tp_size = tp_size
        pool.memory_pool = torch.empty(POOL_SIZE, dtype=torch.int8)
        pool._pool_ipc_handle = (b"pool-handle", 0)
        pool._pool_device_index = 0
        pool.sync_buffer_list = []
        pool.available_chunks = [
            npu_ipc.MmItemMemoryChunk((0, POOL_SIZE), pool.pop_sync_buffer())
        ]
        pool.occupied_chunks = {}
        pool._next_lease_id = 0
        pool._lock = threading.Lock()
        pool._pool_full_warned = False
        pool._lease_failed_warned = False
        return pool

    def _make_proxy(
        self, sync_buffer, byte_offset: int, nbytes: int, *, origin_pid=None
    ):
        proxy = npu_ipc.NpuIpcTensorTransportProxy.__new__(
            npu_ipc.NpuIpcTensorTransportProxy
        )
        proxy.proxy_state = {
            "ipc_extra": {
                "pool_handle": self.pool._pool_ipc_handle,
                "pool_byte_offset": byte_offset,
                "lease_id": self.pool.occupied_chunks[byte_offset].lease_id,
                "pool_device_index": 0,
                "shape": (nbytes,),
                "dtype": torch.int8,
                "stride": (1,),
                "recons_shape": (nbytes,),
                "recons_dtype": torch.int8,
                "sync_buffer_meta": dict(sync_buffer.meta_data),
                "use_pool_handle_cache": True,
            }
        }
        proxy.reconstruct_tensor = None
        proxy._consumer_acknowledged = False
        proxy._origin_pid = os.getpid() + 1 if origin_pid is None else origin_pid
        proxy.total_consumer_count = self.pool.tp_size
        proxy._pool_tensor = None
        self._proxies.append(proxy)
        return proxy

    def _lease(self, nbytes=LEASE_BYTES):
        chunk = self.pool._lease_chunk_for_bytes(nbytes)
        self.assertIsNotNone(chunk)
        return chunk.sync_buffer, chunk.start

    def _lease_after_a_warmup(self, nbytes=LEASE_BYTES):
        """Lease twice so the tested chunk's id is not the trivial first one.

        A slot has to hold the lease id; a bare acknowledgement flag would make
        the first lease (id 1) indistinguishable from "everyone acknowledged".
        """
        self._lease(nbytes)
        return self._lease(nbytes)

    def test_leases_split_chunks_and_return_adjacent_space(self):
        _, byte_offset = self._lease()
        self.assertEqual(len(self.pool.occupied_chunks), 1)
        self.assertEqual(self.pool.occupied_chunks[byte_offset].start, 0)
        self.assertEqual(self.pool.occupied_chunks[byte_offset].mem_size, LEASE_BYTES)
        self.assertEqual(self.pool.available_chunks[0].start, LEASE_BYTES)

        for chunk in list(self.pool.occupied_chunks.values()):
            self.assertTrue(self.pool._return_lease_locked(chunk.start, chunk.lease_id))

        self.assertEqual(self.pool.occupied_chunks, {})
        self.assertEqual(len(self.pool.available_chunks), 1)
        self.assertEqual(self.pool.available_chunks[0].mem_size, POOL_SIZE)

    def test_adjacent_leases_merge_back_into_one_chunk(self):
        for _ in range(3):
            self._lease()
        self.assertEqual(len(self.pool.occupied_chunks), 3)
        self.assertEqual(len(self.pool.available_chunks), 1)

        for chunk in list(self.pool.occupied_chunks.values()):
            self.pool.reclaim_chunk(chunk)
        self.pool.merge_chunks()

        self.assertEqual(len(self.pool.available_chunks), 1)
        self.assertEqual(self.pool.available_chunks[0].mem_size, POOL_SIZE)

    def test_exact_fit_lease_recycles_the_previous_sync_buffer(self):
        previous_buffer = self.pool.available_chunks[0].sync_buffer

        self._lease(POOL_SIZE)

        self.assertEqual(self.pool.available_chunks, [])
        self.assertTrue(
            any(buffer is previous_buffer for buffer in self.pool.sync_buffer_list),
            "an exactly consumed chunk must return its shm buffer to the free list",
        )

    def test_exhausted_pool_warns_once_and_reports_no_slice(self):
        self._lease(POOL_SIZE)

        self.assertIsNone(self.pool._lease_chunk_for_bytes(1))
        self.assertTrue(self.pool._pool_full_warned)

    def test_empty_tensor_degrades_to_cpu_transport(self):
        self.assertIsNone(
            self.pool.wrap_tensor(torch.empty(0), use_pool_handle_cache=True)
        )
        self.assertEqual(len(self.pool.occupied_chunks), 0)

    def test_stale_lease_id_cannot_release_a_released_offset(self):
        _, byte_offset = self._lease()
        stale_id = self.pool.occupied_chunks[byte_offset].lease_id
        self.assertTrue(self.pool._return_lease_locked(byte_offset, stale_id))

        # The merged-back range is re-leased at the same offset with a fresh
        # id; a late cancel carrying the stale id must be rejected.
        _, byte_offset = self._lease()
        self.assertEqual(byte_offset, 0)
        self.assertFalse(self.pool._return_lease_locked(byte_offset, stale_id))
        self.assertEqual(len(self.pool.occupied_chunks), 1)
        self.assertTrue(
            self.pool._return_lease_locked(
                byte_offset, self.pool.occupied_chunks[byte_offset].lease_id
            )
        )
        self.assertEqual(len(self.pool.occupied_chunks), 0)

    def test_released_lease_is_found_by_its_own_nonzero_offset(self):
        _, first_offset = self._lease()
        self.assertEqual(first_offset, 0)
        second_sync, second_offset = self._lease()
        self.assertNotEqual(second_offset, 0)

        proxy = self._make_proxy(second_sync, second_offset, LEASE_BYTES)
        self.pool.cancel_proxy(proxy)

        self.assertEqual(list(self.pool.occupied_chunks), [first_offset])
        self.assertEqual(
            [(c.start, c.end) for c in self.pool.available_chunks],
            [(second_offset, POOL_SIZE)],
        )

    def test_cancel_proxy_returns_an_undispatched_slice(self):
        sync_buffer, byte_offset = self._lease()
        proxy = self._make_proxy(sync_buffer, byte_offset, LEASE_BYTES)

        self.pool.cancel_proxy(proxy)

        self.assertEqual(self.pool.occupied_chunks, {})
        self.assertEqual(len(self.pool.available_chunks), 1)
        self.assertEqual(self.pool.available_chunks[0].mem_size, POOL_SIZE)

    def test_recycled_sync_buffer_starts_from_zero(self):
        sync_buffer, _ = self._lease()
        sync_buffer.buffer_wrapper[:] = 1
        self.pool.push_sync_buffer(sync_buffer)

        reused = self.pool.pop_sync_buffer()

        self.assertIs(reused, sync_buffer)
        self.assertEqual(reused.buffer_wrapper.tolist(), [0] * TP_SIZE)

    def test_acknowledgements_use_one_slot_per_consumer_rank(self):
        sync_buffer, byte_offset = self._lease_after_a_warmup()
        chunk = self.pool.occupied_chunks[byte_offset]
        slots = self._make_proxy(sync_buffer, byte_offset, LEASE_BYTES).sync_buffer_data
        self.assertEqual(len(slots), TP_SIZE)

        # Two copies of the same proxy stand in for two deliveries to one rank;
        # a duplicate acknowledgement must not count twice. Each slot holds the
        # lease id, so a recycled lease's board cannot be completed by an old
        # acknowledgement either.
        with patch.object(npu_ipc, "resolve_consumer_rank", return_value=1):
            self._make_proxy(
                sync_buffer, byte_offset, LEASE_BYTES
            ).acknowledge_consumption()
            self._make_proxy(
                sync_buffer, byte_offset, LEASE_BYTES
            ).acknowledge_consumption()
        self.assertEqual(slots.tolist(), [0, chunk.lease_id, 0, 0])
        self.assertFalse(chunk.try_to_recycle())

        # The remaining ranks each add exactly one slot.
        for rank in (0, 2, 3):
            with patch.object(npu_ipc, "resolve_consumer_rank", return_value=rank):
                self._make_proxy(
                    sync_buffer, byte_offset, LEASE_BYTES
                ).acknowledge_consumption()
        self.assertEqual(slots.tolist(), [chunk.lease_id] * TP_SIZE)
        self.assertTrue(chunk.try_to_recycle())
        self.assertEqual(slots.tolist(), [0, 0, 0, 0])

    def test_proxy_state_shape_is_pinned(self):
        """The serialized proxy envelope must not grow unreviewed payload keys."""
        sync_buffer, byte_offset = self._lease(64)
        data = torch.empty(64, dtype=torch.int8)
        proxy = npu_ipc.NpuIpcTensorTransportProxy(
            data=data,
            info_data=data,
            sync_buffer=sync_buffer,
            pool_ipc_handle=self.pool._pool_ipc_handle,
            pool_byte_offset=byte_offset,
            pool_lease_id=self.pool.occupied_chunks[byte_offset].lease_id,
            pool_device_index=0,
            tp_size=TP_SIZE,
            use_pool_handle_cache=True,
        )
        self._proxies.append(proxy)
        self.assertEqual(set(proxy.proxy_state), {"ipc_extra"})
        self.assertEqual(
            set(proxy.proxy_state["ipc_extra"]),
            {
                "pool_handle",
                "pool_byte_offset",
                "lease_id",
                "pool_device_index",
                "shape",
                "dtype",
                "stride",
                "recons_shape",
                "recons_dtype",
                "sync_buffer_meta",
                "use_pool_handle_cache",
            },
        )
        self.assertEqual(
            set(proxy.proxy_state["ipc_extra"]["sync_buffer_meta"]),
            {"handle", "shape", "dtype"},
        )

    def test_group_acknowledgement_releases_every_slot(self):
        sync_buffer, byte_offset = self._lease_after_a_warmup()
        chunk = self.pool.occupied_chunks[byte_offset]
        proxy = self._make_proxy(sync_buffer, byte_offset, LEASE_BYTES)

        proxy.acknowledge_consumption(TP_SIZE)

        self.assertEqual(proxy.sync_buffer_data.tolist(), [chunk.lease_id] * TP_SIZE)
        self.assertTrue(chunk.try_to_recycle())

    def test_partial_group_acknowledgement_is_rejected(self):
        sync_buffer, byte_offset = self._lease()
        proxy = self._make_proxy(sync_buffer, byte_offset, LEASE_BYTES)

        with self.assertRaisesRegex(ValueError, "complete consumer group"):
            proxy.acknowledge_consumption(TP_SIZE - 2)

    def test_origin_process_read_does_not_consume_acknowledgement_quota(self):
        sync_buffer, byte_offset = self._lease()
        chunk = self.pool.occupied_chunks[byte_offset]
        proxy = self._make_proxy(
            sync_buffer, byte_offset, LEASE_BYTES, origin_pid=os.getpid()
        )

        proxy.acknowledge_consumption(TP_SIZE)

        self.assertEqual(chunk.sync_buffer.buffer_wrapper.tolist(), [0] * TP_SIZE)
        self.assertFalse(chunk.try_to_recycle())

        # An explicit abandon from the pool owner releases the whole group.
        proxy.release_without_reconstruction(TP_SIZE)

        self.assertTrue(chunk.try_to_recycle())

    def test_abandon_from_a_consumer_marks_only_that_rank(self):
        sync_buffer, byte_offset = self._lease_after_a_warmup()
        chunk = self.pool.occupied_chunks[byte_offset]
        proxy = self._make_proxy(sync_buffer, byte_offset, LEASE_BYTES)

        with patch.object(npu_ipc, "resolve_consumer_rank", return_value=2):
            proxy.release_without_reconstruction()

        self.assertEqual(
            chunk.sync_buffer.buffer_wrapper.tolist(), [0, 0, chunk.lease_id, 0]
        )
        self.assertFalse(chunk.try_to_recycle())

    def test_stale_acknowledgement_cannot_recycle_a_re_leased_chunk(self):
        """A board outlives its lease, so an old acknowledgement must be inert.

        Boards travel with the free range they guard, so after a couple of
        leases the segment that carried lease 1 guards lease 3 while a proxy
        from lease 1 may still hold a mapping of it. Matching slots against the
        lease id keeps that late write from completing the new lease's board.
        """
        first = self.pool._lease_chunk_for_bytes(LEASE_BYTES)
        stale_board, stale_id = first.sync_buffer, first.lease_id
        stale_board.buffer_wrapper[:] = stale_id  # every consumer of lease 1 acked
        self.pool.recycle_chunks()

        second = self.pool._lease_chunk_for_bytes(LEASE_BYTES)
        second.sync_buffer.buffer_wrapper[:] = second.lease_id
        self.pool.recycle_chunks()

        # The regular recycle path rotates the segment back to an occupied
        # chunk; no test-only state is needed to reach this.
        third = self.pool._lease_chunk_for_bytes(LEASE_BYTES)
        self.assertIs(third.sync_buffer, stale_board)
        self.assertNotEqual(third.lease_id, stale_id)

        stale_board.buffer_wrapper[:] = stale_id  # late ack of the first lease
        self.assertFalse(third.try_to_recycle())

        stale_board.buffer_wrapper[:] = third.lease_id
        self.assertTrue(third.try_to_recycle())

    def test_acknowledgement_without_a_lease_id_fails_loudly(self):
        sync_buffer, byte_offset = self._lease()
        proxy = self._make_proxy(sync_buffer, byte_offset, LEASE_BYTES)
        proxy.proxy_state["ipc_extra"].pop("lease_id")

        with self.assertRaisesRegex(ValueError, "needs the pool lease id"):
            proxy.acknowledge_consumption()

    def test_lease_id_exhaustion_degrades_to_cpu_transport(self):
        # The lease counter is checked before a board is popped, so a rejected
        # allocation must leave the free list untouched and report "no chunk"
        # rather than raise into a request.
        spare = self.pool.pop_sync_buffer()
        self.pool.push_sync_buffer(spare)
        self.pool._next_lease_id = npu_ipc._MAX_LEASE_ID

        self.assertIsNone(self.pool._lease_chunk_for_bytes(LEASE_BYTES))

        self.assertEqual(self.pool.occupied_chunks, {})
        self.assertIs(self.pool.pop_sync_buffer(), spare)

    def test_board_allocation_failure_degrades_to_cpu_transport(self):
        """A pool that cannot hand out a board must not fail the request."""
        with patch.object(npu_ipc, "ShmSyncBuffer", side_effect=OSError("no shm")):
            self.assertIsNone(self.pool._lease_chunk_for_bytes(LEASE_BYTES))
            self.assertIsNone(
                self.pool.wrap_tensor(
                    torch.empty(LEASE_BYTES, dtype=torch.int8),
                    use_pool_handle_cache=True,
                )
            )

        self.assertEqual(self.pool.occupied_chunks, {})

    def test_pickle_round_trip_drops_owner_local_state(self):
        """A proxy the pool owner reconstructed must reach the scheduler clean.

        The owner hashes a proxy before dispatch, so its reconstructed tensor is
        a process-local read. Shipping it would copy the whole feature over the
        IPC hop and, on the device that matches, make the receiver take the
        early return in ``reconstruct_on_target_device`` and never acknowledge
        the lease. The msgpack codec drops the same state.
        """
        sync_buffer, byte_offset = self._lease()
        proxy = self._make_proxy(sync_buffer, byte_offset, LEASE_BYTES)
        proxy.reconstruct_tensor = torch.zeros(LEASE_BYTES, dtype=torch.int8)
        proxy._sync_buffer = proxy.sync_buffer
        proxy._sync_buffer_data = proxy.sync_buffer_data
        proxy._consumer_acknowledged = True

        revived = pickle.loads(pickle.dumps(proxy))
        self._proxies.append(revived)

        self.assertIsNone(revived.reconstruct_tensor)
        self.assertFalse(revived._consumer_acknowledged)
        self.assertIsNone(revived._sync_buffer)
        self.assertIsNone(revived._sync_buffer_data)
        self.assertIsNone(revived._pool_tensor)
        # The lease coordinates travel, so the receiver can still acknowledge.
        first_chunk = self.pool.occupied_chunks[byte_offset]
        self.assertEqual(
            revived.proxy_state["ipc_extra"]["lease_id"], first_chunk.lease_id
        )
        self.assertEqual(revived._origin_pid, proxy._origin_pid)

        with patch.object(npu_ipc, "resolve_consumer_rank", return_value=0):
            revived.acknowledge_consumption()

        self.assertEqual(
            revived.sync_buffer_data.tolist(),
            [first_chunk.lease_id, 0, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
