import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode_host_cache import DecodeHostCache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.kv_cache_builder import resolve_decode_retraction_backup
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MODULE = "sglang.srt.disaggregation.decode_host_cache"


class TestDecodeHostCache(unittest.TestCase):
    def setUp(self):
        self.device_pool = object.__new__(MHATokenToKVPool)
        self.device_pool.kv_cache_layout = "nhd"
        self.device_pool.head_dim = self.device_pool.v_head_dim = 8
        self.device_pool.device = "cpu"
        self.device_pool.layer_num = 2
        self.host_pool = Mock(
            device_pool=self.device_pool,
            page_size=4,
            layout="layer_first",
            logical_size=24,
        )
        self.host_pool.available_size.return_value = 24
        self.event = Mock()
        self.event.query.return_value = False
        self.engine = Mock(io_backend="kernel")
        self.engine.submit_host_to_device.return_value.finish_event = self.event
        self.manager = DecodeHostCache(
            self.device_pool, 4, self.host_pool, 8, self.engine
        )

    @staticmethod
    def req(index=0):
        return Mock(rid="reused-rid", kv=SimpleNamespace(req_pool_idx=index))

    def test_allocation_is_page_aligned_and_failure_keeps_no_ownership(self):
        first, second = self.req(), self.req()
        indices = torch.arange(8)
        self.host_pool.alloc.side_effect = [indices, None]

        self.assertIs(self.manager.allocate(first, 5), indices)
        self.assertIsNone(self.manager.allocate(second, 5))
        self.assertEqual(self.host_pool.alloc.call_args.args, (8,))
        self.assertTrue(self.manager.contains(first))
        self.assertFalse(self.manager.contains(second))
        with self.assertRaisesRegex(ValueError, "already owns"):
            self.manager.allocate(first, 1)

        self.manager.release(second)
        self.host_pool.free.assert_not_called()
        self.manager.release(first)
        self.manager.release(first)
        self.host_pool.free.assert_called_once_with(indices)

    def test_batched_load_excludes_padding_and_holds_pages_until_completion(self):
        first, second, direct = self.req(), self.req(1), self.req(2)
        first_indices, second_indices = torch.arange(4), torch.arange(4, 12)
        self.host_pool.alloc.side_effect = [first_indices, second_indices]
        self.manager.allocate(first, 3)
        self.manager.allocate(second, 5)
        req_pool = SimpleNamespace(
            req_to_token=torch.tensor(
                [[11, 17, 19, 99, 99], [7, 8, 2, 3, 4], [21, 22, 23, 24, 25]],
                dtype=torch.int32,
            )
        )

        result = self.manager.load([first, second, direct], req_pool)

        self.assertIs(result, self.event)
        transfers = self.engine.submit_host_to_device.call_args.args[0]
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].device_indices.dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                transfers[0].host_indices, torch.tensor([0, 1, 2, 4, 5, 6, 7, 8])
            )
        )
        self.assertTrue(
            torch.equal(
                transfers[0].device_indices, torch.tensor([11, 17, 19, 7, 8, 2, 3, 4])
            )
        )
        self.event.wait.assert_called_once_with()
        self.event.synchronize.assert_not_called()
        self.assertIsNone(self.manager.load([first, second], req_pool))
        self.engine.submit_host_to_device.assert_called_once()
        self.manager.poll()
        self.host_pool.free.assert_not_called()
        self.assertTrue(self.manager.contains(first))

        self.event.query.return_value = True
        self.manager.poll()

        self.assertFalse(self.manager.contains(first))
        self.assertFalse(self.manager.contains(second))
        self.assertEqual(self.host_pool.free.call_count, 2)
        self.assertIs(self.host_pool.free.call_args_list[0].args[0], first_indices)
        self.assertIs(self.host_pool.free.call_args_list[1].args[0], second_indices)
        self.manager.release(first)
        self.assertEqual(self.host_pool.free.call_count, 2)

    def test_abort_waits_for_copy_before_returning_pages(self):
        req = self.req()
        indices = torch.arange(4)
        self.host_pool.alloc.return_value = indices
        self.manager.allocate(req, 3)
        self.manager.load(
            [req], SimpleNamespace(req_to_token=torch.tensor([[7, 3, 2]]))
        )
        order = []
        self.event.synchronize.side_effect = lambda: order.append("copy finished")
        self.host_pool.free.side_effect = lambda _: order.append("pages freed")

        self.manager.release(req)
        self.manager.poll()

        self.assertEqual(order, ["copy finished", "pages freed"])
        self.assertFalse(self.manager.contains(req))
        self.host_pool.free.assert_called_once_with(indices)

    def test_poll_reclaims_only_rank_consensus_and_skips_waiting_requests(self):
        waiting, first, second = self.req(), self.req(1), self.req(2)
        indices = [torch.arange(i * 4, (i + 1) * 4) for i in range(3)]
        self.host_pool.alloc.side_effect = indices
        for req in (waiting, first, second):
            self.manager.allocate(req, 3)
        self.manager.load(
            [first, second],
            SimpleNamespace(req_to_token=torch.arange(9).reshape(3, 3)),
        )
        self.event.query.return_value = True
        group = object()
        with (
            patch(f"{MODULE}.torch.distributed.get_world_size", return_value=2),
            patch(
                f"{MODULE}.torch.distributed.all_reduce",
                side_effect=lambda count, **_: count.fill_(1),
            ) as reduce,
        ):
            self.manager.poll(group)

        self.assertIs(reduce.call_args.kwargs["group"], group)
        self.assertEqual(reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.MIN)
        self.host_pool.free.assert_called_once_with(indices[1])
        self.assertTrue(self.manager.contains(waiting))
        self.assertFalse(self.manager.contains(first))
        self.assertTrue(self.manager.contains(second))

    def test_registration_uses_host_buffers_in_device_k_v_order(self):
        buffers = [torch.zeros((8, 2, 8), dtype=torch.float16) for _ in range(4)]
        self.host_pool.host_kv_data_refs = buffers
        self.host_pool.token_stride_size = 32

        pointers, sizes, item_sizes = self.manager.get_contiguous_buf_infos()

        self.assertEqual(pointers, [buffer.data_ptr() for buffer in buffers])
        self.assertEqual(sizes, [buffer.nbytes for buffer in buffers])
        self.assertEqual(item_sizes, [128] * 4)

    def test_receive_reserves_retraction_capacity_and_clear_keeps_shared_pool(self):
        self.host_pool.available_size.return_value = 11
        waiting = self.req()
        self.assertIsNone(self.manager.allocate(waiting, 1))
        self.assertFalse(self.manager.contains(waiting))
        self.host_pool.alloc.assert_not_called()

        self.host_pool.available_size.return_value = 12
        indices = torch.arange(4)
        self.host_pool.alloc.return_value = indices
        self.assertIs(self.manager.allocate(waiting, 1), indices)
        self.manager.clear()
        self.host_pool.free.assert_called_once_with(indices)
        self.host_pool.destroy.assert_not_called()

        self.host_pool.logical_size = 8
        with self.assertRaisesRegex(ValueError, "increase --hicache-size"):
            DecodeHostCache(self.device_pool, 4, self.host_pool, 8, self.engine)

    def test_unsupported_pool_layouts_fail_before_allocating_host_memory(self):
        self.device_pool.kv_cache_layout = "hnd"
        with self.assertRaisesRegex(ValueError, "symmetric NHD"):
            DecodeHostCache(self.device_pool, 4, self.host_pool, 8, self.engine)
        self.device_pool.kv_cache_layout = "nhd"
        self.device_pool.v_head_dim = 16
        with self.assertRaisesRegex(ValueError, "symmetric NHD"):
            DecodeHostCache(self.device_pool, 4, self.host_pool, 8, self.engine)
        mla = object.__new__(MLATokenToKVPool)
        mla.use_dsa = True
        with self.assertRaisesRegex(ValueError, "plain MLA"):
            DecodeHostCache(mla, 4, self.host_pool, 8, self.engine)

    def test_retraction_shares_dense_mla_pool_but_rejects_indexer_state(self):
        mla = object.__new__(MLATokenToKVPool)
        mla.use_dsa = False
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = Mock()
        cache.host_pool_group = SimpleNamespace(entry_map={PoolName.KV: self.host_pool})
        cache.is_mamba_enabled = False
        cache.token_to_kv_pool_allocator = Mock(get_kvcache=Mock(return_value=mla))
        self.assertTrue(cache.supports_retraction_backup())
        mla.use_dsa = True
        self.assertFalse(cache.supports_retraction_backup())


class TestDecodeHostCacheSizing(unittest.TestCase):
    def test_automatic_size_fits_request_and_receive_reservation(self):
        for enabled, context_len, expected_ratio in (
            (False, 81, 80 / 256),
            (True, 81, 160 / 256),
            (False, 2049, 1.0),
            (True, 2049, 2.0),
            (False, 2, 0.2),
            (True, 2, 0.2),
        ):
            req_pool = SimpleNamespace(max_context_len=context_len)
            kv_pool = SimpleNamespace(size=256, page_size=16)
            worker = Mock(
                get_memory_pool=Mock(
                    return_value=(
                        req_pool,
                        Mock(get_kvcache=Mock(return_value=kv_pool)),
                    )
                )
            )
            with (
                self.subTest(enabled=enabled, context_len=context_len),
                get_context().override_server_args(
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup="host_pool",
                    disaggregation_decode_enable_host_cache=enabled,
                    enable_hierarchical_cache=False,
                    hicache_ratio=None,
                    hicache_size=0,
                ),
            ):
                self.assertEqual(
                    resolve_decode_retraction_backup(tp_worker=worker), "host_pool"
                )
                self.assertEqual(get_memory().hicache_ratio, expected_ratio)

    def test_explicit_host_size_and_ratio_are_preserved(self):
        for ratio, size in ((0.1, 0), (None, 1.25), (0.3, 1.25)):
            with (
                self.subTest(ratio=ratio, size=size),
                get_context().override_server_args(
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup="host_pool",
                    disaggregation_decode_enable_host_cache=True,
                    enable_hierarchical_cache=False,
                    hicache_ratio=ratio,
                    hicache_size=size,
                ),
            ):
                worker = Mock()
                resolve_decode_retraction_backup(tp_worker=worker)
                self.assertEqual(get_memory().hicache_size, size)
                if ratio is not None:
                    self.assertEqual(get_memory().hicache_ratio, ratio)
                worker.get_memory_pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
