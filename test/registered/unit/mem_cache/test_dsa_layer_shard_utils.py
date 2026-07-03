from types import SimpleNamespace

import torch

from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range
from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSALayerShardUtils(CustomTestCase):
    def test_balanced_layer_ranges_cover_all_layers_once(self):
        ranges = [get_layer_shard_range(rank, 4, 10) for rank in range(4)]
        self.assertEqual(ranges, [(0, 3), (3, 6), (6, 8), (8, 10)])

        covered = [layer_id for start, end in ranges for layer_id in range(start, end)]
        self.assertEqual(covered, list(range(10)))

    def test_owner_matches_uneven_layer_ranges(self):
        self.assertEqual(
            [get_layer_owner(i, 4, 10) for i in range(10)],
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        )

    def test_empty_tail_shards_have_empty_ranges(self):
        ranges = [get_layer_shard_range(rank, 4, 2) for rank in range(4)]
        self.assertEqual(ranges, [(0, 1), (1, 2), (2, 2), (2, 2)])

    def test_prefetch_uses_sync_fallback_without_dedicated_communicator(self):
        broadcasts = []
        counter = SimpleNamespace(wait_until=lambda _: self.fail("unexpected wait"))
        pool = SimpleNamespace(
            remote_kv_layer_id=None,
            pending_remote_kv_broadcast=False,
            pending_remote_kv_layer_id=None,
            layer_broadcast_comm=None,
            remote_kv_buffer=object(),
            kv_buffer=[object()],
            start_layer=0,
            _local_layer_idx=lambda layer_id: layer_id,
            _is_layer_owned=lambda _: True,
        )

        def broadcast(tensor, layer_id, *, src_tensor, use_layer_broadcast_comm):
            broadcasts.append((tensor, layer_id, src_tensor, use_layer_broadcast_comm))

        pool._broadcast_tensor_from_owner = broadcast
        # Bind the real method against a lightweight stand-in so the sync
        # (no dedicated NCCL comm) fallback path can be exercised on CPU.
        LayerSplitDSATokenToKVPool.prefetch_kv_buffer(
            pool,
            layer_id=0,
            layer_transfer_counter=counter,
            layer_transfer_idx=3,
        )

        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(pool.remote_kv_layer_id, 0)

    def test_finalize_pending_broadcast_promotes_layer_id(self):
        # After an async prefetch, finalizing must promote pending -> remote so a
        # subsequent read of the same layer reuses the broadcast result.
        pool = SimpleNamespace(
            pending_remote_kv_broadcast=True,
            pending_remote_kv_layer_id=7,
            remote_kv_layer_id=None,
            device_module=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(wait_stream=lambda _stream: None)
            ),
            kv_broadcast_stream=object(),
        )
        LayerSplitDSATokenToKVPool._finalize_pending_kv_broadcast(
            pool, set_remote_layer_id=True
        )
        self.assertFalse(pool.pending_remote_kv_broadcast)
        self.assertEqual(pool.remote_kv_layer_id, 7)
        self.assertIsNone(pool.pending_remote_kv_layer_id)

    def test_get_broadcastable_kv_buffer_returns_owner_contents(self):
        # A non-owner read must return the *owner's* KV bytes, copied into the
        # remote scratch buffer by the broadcast. This checks prefetch_kv_buffer
        # + _get_broadcastable_kv_buffer surface the correct contents.
        layer_num = 4
        shard_size = 2
        owner_kv = {
            layer_id: torch.full((3, 1, 8), float(layer_id + 1))
            for layer_id in range(layer_num)
        }
        remote = torch.zeros((3, 1, 8))

        pool = SimpleNamespace(
            layer_num=layer_num,
            layer_shard_size=shard_size,
            start_layer=0,
            remote_kv_layer_id=None,
            pending_remote_kv_broadcast=False,
            pending_remote_kv_layer_id=None,
            remote_kv_buffer=remote,
        )
        pool._local_layer_idx = lambda layer_id: layer_id - pool.start_layer
        pool._is_layer_owned = lambda layer_id: True
        # kv_buffer holds this rank's owned layers; broadcast copies owner->remote.
        pool.kv_buffer = [owner_kv[i] for i in range(layer_num)]

        def broadcast(tensor, layer_id, *, src_tensor, use_layer_broadcast_comm=False):
            # Simulate the owner writing its layer into the remote scratch buffer.
            tensor.copy_(owner_kv[layer_id])

        pool._broadcast_tensor_from_owner = broadcast

        for layer_id in range(layer_num):
            buf = LayerSplitDSATokenToKVPool._get_broadcastable_kv_buffer(
                pool, layer_id
            )
            self.assertTrue(torch.equal(buf, owner_kv[layer_id]))
            self.assertEqual(pool.remote_kv_layer_id, layer_id)
