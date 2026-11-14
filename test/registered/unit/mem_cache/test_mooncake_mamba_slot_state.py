import contextlib
import ctypes
import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_pool(page_size=1, with_side=True, conv_only=False):
    pool = object.__new__(MambaPoolHost)
    pool.page_size = page_size
    pool.layout = "page_first"
    pool.num_mamba_layers = 2
    pool.temporal_dtype = torch.bfloat16
    pool.conv_dtype = torch.float32
    pool.temporal_state_elem_size = 0 if conv_only else 4
    pool.conv_state_elem_sizes = [3]
    pool.conv_state_shapes = [(3,)]
    pool.temporal_buffer = (
        torch.arange(8 * 2 * pool.temporal_state_elem_size, dtype=torch.float32)
        .to(torch.bfloat16)
        .reshape(8 // page_size, 2, page_size, pool.temporal_state_elem_size)
    )
    pool.conv_buffer = [
        torch.arange(48, dtype=torch.float32).reshape(8 // page_size, 2, page_size, 3)
    ]
    pool.slot_state_buffers = (
        [
            torch.arange(40, dtype=torch.int64).reshape(8, 5),
            torch.empty(8, 0),
            torch.arange(24, dtype=torch.float32).reshape(8, 3),
        ]
        if with_side
        else []
    )
    pool.slot_state_device_tensors = pool.slot_state_buffers
    return pool


def make_backend(pool, objects):
    backend = object.__new__(MooncakeStore)
    backend.registered_pools = {PoolName.MAMBA: pool}
    backend.mha_suffix = "0"
    backend.config_prefix = None
    backend.mem_pool_host = SimpleNamespace(kv_buffer=None)
    backend.request_context = lambda **kwargs: contextlib.nullcontext()
    backend._batch_exist = lambda keys: [int(key in objects) for key in keys]
    return backend


class TestMooncakeMambaSlotState(CustomTestCase):
    def test_component_round_trip_and_missing_side_state(self):
        for page_size, with_side, conv_only in itertools.product(
            (1, 2), (False, True), (False, True)
        ):
            with self.subTest(page_size=page_size, side=with_side, conv_only=conv_only):
                pool = make_pool(page_size, with_side, conv_only)
                objects = {}
                backend = make_backend(pool, objects)
                transfer = PoolTransfer(name=PoolName.MAMBA)
                source = torch.tensor([0, 4] if page_size == 1 else [0, 1, 4, 5])
                target = torch.tensor([2, 6] if page_size == 1 else [2, 3, 6, 7])
                pointers, sizes = pool.get_page_buffer_meta(source)
                keys, multiplier = backend._get_hybrid_page_component_keys(
                    ["a", "b"], transfer
                )
                expected_components = int(not conv_only) + 1 + 2 * int(with_side)
                self.assertEqual(multiplier, expected_components)
                self.assertEqual(len(keys), len(pointers))
                self.assertTrue(all(size > 0 for size in sizes))
                for key, ptr, size in zip(keys, pointers, sizes):
                    objects[key] = ctypes.string_at(ptr, size)
                dest_ptrs, dest_sizes = pool.get_page_buffer_meta(target)
                self.assertEqual(sizes, dest_sizes)
                for key, ptr, size in zip(keys, dest_ptrs, dest_sizes):
                    ctypes.memmove(ptr, objects[key], size)
                # Check the actual tensors, not just the metadata used to copy.
                for src, dst in ((0, 2), (4, 6)):
                    for buf in pool.slot_state_buffers:
                        torch.testing.assert_close(
                            buf[src : src + page_size], buf[dst : dst + page_size]
                        )
                self.assertEqual(
                    backend.batch_exists_v2(["a", "b"], [transfer]).kv_hit_pages, 2
                )
                if with_side:
                    self.assertIn("b_0_slot_state_v1_2", objects)
                    self.assertNotIn("b_0_slot_state_v1_1", objects)
                    for key in ("b_0_slot_state_v1_0", "b_0_slot_state_v1_2"):
                        value = objects.pop(key)
                        self.assertEqual(
                            backend.batch_exists_v2(
                                ["a", "b"], [transfer]
                            ).kv_hit_pages,
                            1,
                        )
                        objects[key] = value

    def test_rejects_noncontiguous_side_state(self):
        pool = make_pool()
        pool.slot_state_buffers = [torch.zeros(8, 6)[:, ::2]]
        with self.assertRaisesRegex(ValueError, "complete and contiguous"):
            pool.get_page_buffer_meta(torch.tensor([0]))

    def test_rejects_incomplete_side_state_page(self):
        pool = make_pool(page_size=2)
        pool.slot_state_buffers = [torch.zeros(7, 3)]
        with self.assertRaisesRegex(ValueError, "complete and contiguous"):
            pool.get_page_buffer_meta(torch.tensor([6, 7]))


if __name__ == "__main__":
    unittest.main()
