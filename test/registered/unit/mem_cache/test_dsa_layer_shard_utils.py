from types import SimpleNamespace

from sglang.srt.mem_cache.memory_pool import (
    MLATokenToKVPool,
    _get_layer_owner,
    _get_layer_shard_range,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSALayerShardUtils(CustomTestCase):
    def test_balanced_layer_ranges_cover_all_layers_once(self):
        ranges = [_get_layer_shard_range(rank, 4, 10) for rank in range(4)]
        self.assertEqual(ranges, [(0, 3), (3, 6), (6, 8), (8, 10)])

        covered = [layer_id for start, end in ranges for layer_id in range(start, end)]
        self.assertEqual(covered, list(range(10)))

    def test_owner_matches_uneven_layer_ranges(self):
        self.assertEqual(
            [_get_layer_owner(i, 4, 10) for i in range(10)],
            [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        )

    def test_empty_tail_shards_have_empty_ranges(self):
        ranges = [_get_layer_shard_range(rank, 4, 2) for rank in range(4)]
        self.assertEqual(ranges, [(0, 1), (1, 2), (2, 2), (2, 2)])

    def test_prefetch_uses_sync_fallback_without_dedicated_communicator(self):
        broadcasts = []
        counter = SimpleNamespace(wait_until=lambda _: self.fail("unexpected wait"))
        pool = SimpleNamespace(
            layer_shard_enabled=True,
            remote_kv_layer_id=None,
            pending_remote_kv_broadcast=False,
            pending_remote_kv_layer_id=None,
            layer_broadcast_comm=None,
            remote_kv_buffer=object(),
            kv_buffer=[object()],
            _local_layer_idx=lambda _: 0,
            _is_layer_owned=lambda _: True,
        )

        def broadcast(tensor, layer_id, *, src_tensor, use_layer_broadcast_comm):
            broadcasts.append((tensor, layer_id, src_tensor, use_layer_broadcast_comm))

        pool._broadcast_tensor_from_owner = broadcast
        MLATokenToKVPool.prefetch_kv_buffer(
            pool,
            layer_id=7,
            layer_transfer_counter=counter,
            layer_transfer_idx=3,
        )

        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(pool.remote_kv_layer_id, 7)
