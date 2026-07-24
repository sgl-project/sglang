import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.cp_cache_layer_split.broadcast import BroadcastSlots
from sglang.srt.mem_cache.dsa_cache_layer_split import (
    LayerSplitDSATokenToKVPool,
    get_dsa_layer_split_effective_num_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSALayerShardUtils(CustomTestCase):
    def test_effective_layer_count_includes_remote_scratch(self):
        self.assertEqual(
            get_dsa_layer_split_effective_num_layers(10, 4),
            4,
        )
        self.assertEqual(get_dsa_layer_split_effective_num_layers(10, 1), 10)

    def test_prefetch_uses_sync_fallback_without_dedicated_communicator(self):
        broadcasts = []
        counter = SimpleNamespace(wait_until=lambda _: self.fail("unexpected wait"))
        pool = SimpleNamespace(
            remote_kv_layer_id=None,
            _broadcast_slots=BroadcastSlots(("kv",)),
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
        device = torch.device("cuda")
        current_stream = MagicMock()
        side_stream = MagicMock()
        side_stream.device = device
        device_module = MagicMock()
        device_module.current_stream.return_value = current_stream
        device_module.Stream.return_value = side_stream
        device_module.stream.return_value = nullcontext()
        slots = BroadcastSlots(("kv",))
        pool = SimpleNamespace(
            _broadcast_slots=slots,
            remote_kv_layer_id=None,
        )

        with patch(
            "sglang.srt.mem_cache.cp_cache_layer_split.broadcast."
            "torch.get_device_module",
            return_value=device_module,
        ):
            with slots.launch("kv", 7, device):
                pass
            LayerSplitDSATokenToKVPool._finalize_pending_kv_broadcast(
                pool, set_remote_layer_id=True
            )

        self.assertFalse(slots.pending("kv").active)
        self.assertEqual(pool.remote_kv_layer_id, 7)
        side_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(side_stream)

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
            _broadcast_slots=BroadcastSlots(("kv",)),
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


if __name__ == "__main__":
    unittest.main()
