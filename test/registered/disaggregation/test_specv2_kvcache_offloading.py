"""
Unit tests for _release_finished_req in DecodeKVCacheOffloadManager.

Verifies that over-allocated KV cache slots (from speculative decoding v2)
are correctly freed when a request finishes, preventing GPU memory leaks.

Requires: torch, sglang (run in an environment with sglang installed)
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="stage-b", runner_config="1-gpu-small")


def _make_mock_req(
    req_pool_idx: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    prefix_indices_len: int = 0,
    rid: int = 0,
):
    """Create a mock Req with the KV cache state needed for testing."""
    req = MagicMock()
    req.rid = rid
    req.req_pool_idx = req_pool_idx
    req.kv_committed_len = kv_committed_len
    req.kv_allocated_len = kv_allocated_len
    req.kv_committed_freed = False
    req.kv_overallocated_freed = False
    req.prefix_indices = list(range(prefix_indices_len))

    def pop_committed():
        assert not req.kv_committed_freed
        req.kv_committed_freed = True
        return req.kv_committed_len

    def pop_overallocated():
        assert not req.kv_overallocated_freed
        req.kv_overallocated_freed = True
        return req.kv_committed_len, req.kv_allocated_len

    req.pop_committed_kv_cache = pop_committed
    req.pop_overallocated_kv_cache = pop_overallocated
    return req


def _make_manager(pool_size: int, page_size: int = 1):
    """Create a DecodeKVCacheOffloadManager with mock pools for testing."""
    # Build a real req_to_token tensor so indexing works
    req_to_token = torch.arange(pool_size, dtype=torch.int64).unsqueeze(0)

    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = req_to_token

    freed_indices = []

    allocator = MagicMock()
    allocator.free = MagicMock(
        side_effect=lambda idx: freed_indices.append(idx.clone())
    )

    tree_cache = MagicMock()
    tree_cache.protected_size_ = 0

    # Bypass __init__ entirely and set attributes directly
    manager = object.__new__(DecodeKVCacheOffloadManager)
    manager.req_to_token_pool = req_to_token_pool
    manager.token_to_kv_pool_allocator = allocator
    manager.page_size = page_size
    manager.tree_cache = tree_cache
    manager.offloaded_state = {}
    manager._extra_pool_specs = ()

    return manager, freed_indices


def _make_server_args(page_size: int):
    return SimpleNamespace(
        page_size=page_size,
        hicache_ratio=2.0,
        hicache_size=0,
        hicache_mem_layout="layer_first",
        hicache_storage_backend="file",
        hicache_storage_backend_extra_config=None,
        hicache_io_backend="kernel",
        hicache_write_policy="write_through_selective",
        served_model_name="test-model",
    )


def _make_mha_like_cache():
    return SimpleNamespace(
        head_num=8,
        head_dim=128,
        layer_num=16,
    )


def _make_mla_like_cache():
    return SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_cache_dim=576,
        layer_num=16,
    )


def _make_nsa_like_cache():
    return SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_cache_dim=640,
        layer_num=16,
        index_k_with_scale_buffer=[MagicMock() for _ in range(16)],
        index_head_dim=128,
        quant_block_size=128,
    )


class TestReleaseFinishedReq(unittest.TestCase):
    """Tests for _release_finished_req overallocation cleanup."""

    def test_no_overallocation(self):
        """Without spec v2, kv_committed == kv_allocated; no extra free."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,  # no overallocation
        )
        prefill_offloaded_len = 8

        manager._release_finished_req(req, prefill_offloaded_len)

        # Only one free call: the committed range [8:20]
        self.assertEqual(len(freed), 1)
        expected = torch.arange(8, 20, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_with_overallocation(self):
        """With spec v2, overallocated slots [committed:allocated] must be freed."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=28,  # 8 over-allocated slots
        )
        prefill_offloaded_len = 8

        manager._release_finished_req(req, prefill_offloaded_len)

        # Two free calls: committed [8:20] and overallocated [20:28]
        self.assertEqual(len(freed), 2)
        expected_committed = torch.arange(8, 20, dtype=torch.int64)
        expected_overalloc = torch.arange(20, 28, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))
        self.assertTrue(torch.equal(freed[1], expected_overalloc))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_overallocation_with_page_alignment(self):
        """With page_size > 1, start of overallocated range is ceil-aligned."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # not page-aligned
            kv_allocated_len=28,
        )
        prefill_offloaded_len = 4

        manager._release_finished_req(req, prefill_offloaded_len)

        # Committed range [4:10]
        # Overallocated: start_p = ceil_align(10, 4) = 12, end_p = 28 => [12:28]
        self.assertEqual(len(freed), 2)
        expected_committed = torch.arange(4, 10, dtype=torch.int64)
        expected_overalloc = torch.arange(12, 28, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))
        self.assertTrue(torch.equal(freed[1], expected_overalloc))

    def test_overallocation_page_aligned_noop(self):
        """When ceil_align(committed, page_size) >= allocated, no overalloc free."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # ceil_align(10, 4) = 12
            kv_allocated_len=12,  # same as aligned start
        )
        prefill_offloaded_len = 4

        manager._release_finished_req(req, prefill_offloaded_len)

        # Only committed [4:10], no overalloc because start_p == end_p
        self.assertEqual(len(freed), 1)
        expected_committed = torch.arange(4, 10, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))

    def test_prefix_indices_decremented(self):
        """protected_size_ is decremented by len(req.prefix_indices)."""
        manager, _ = _make_manager(pool_size=32)
        manager.tree_cache.protected_size_ = 10
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            prefix_indices_len=5,
        )

        manager._release_finished_req(req, start_offset=0)

        self.assertEqual(manager.tree_cache.protected_size_, 5)


class TestDecodeKVCacheOffloadManagerStackAssembly(unittest.TestCase):
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.torch.distributed.get_world_size",
        return_value=1,
    )
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.build_shared_anchor_stack"
    )
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.build_kv_only_stack"
    )
    def test_init_uses_single_pool_hybrid_stack_for_mha_like_pool(
        self,
        mock_build_kv_only_stack,
        mock_build_shared_anchor_stack,
        _mock_world_size,
    ):
        req_to_token_pool = MagicMock()
        token_to_kv_pool_allocator = MagicMock()
        kv_cache = _make_mha_like_cache()
        token_to_kv_pool_allocator.get_kvcache.return_value = kv_cache
        tree_cache = MagicMock()
        tp_group = MagicMock()

        host_pool_group = MagicMock()
        hybrid_controller = MagicMock()
        mock_build_kv_only_stack.return_value = (
            host_pool_group,
            hybrid_controller,
        )

        manager = DecodeKVCacheOffloadManager(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tp_group=tp_group,
            tree_cache=tree_cache,
            server_args=_make_server_args(page_size=64),
        )

        self.assertIs(manager.decode_host_mem_pool, host_pool_group)
        self.assertIs(manager.cache_controller, hybrid_controller)
        self.assertEqual(manager._extra_pool_specs, ())
        called_kwargs = mock_build_kv_only_stack.call_args.kwargs
        self.assertFalse(called_kwargs["use_mla"])
        mock_build_kv_only_stack.assert_called_once()
        mock_build_shared_anchor_stack.assert_not_called()

    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.torch.distributed.get_world_size",
        return_value=1,
    )
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.build_shared_anchor_stack"
    )
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.build_kv_only_stack"
    )
    def test_init_uses_single_pool_hybrid_stack_for_mla_like_pool(
        self,
        mock_build_kv_only_stack,
        mock_build_shared_anchor_stack,
        _mock_world_size,
    ):
        req_to_token_pool = MagicMock()
        token_to_kv_pool_allocator = MagicMock()
        kv_cache = _make_mla_like_cache()
        token_to_kv_pool_allocator.get_kvcache.return_value = kv_cache
        tree_cache = MagicMock()
        tp_group = MagicMock()

        host_pool_group = MagicMock()
        hybrid_controller = MagicMock()
        mock_build_kv_only_stack.return_value = (
            host_pool_group,
            hybrid_controller,
        )

        manager = DecodeKVCacheOffloadManager(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tp_group=tp_group,
            tree_cache=tree_cache,
            server_args=_make_server_args(page_size=64),
        )

        self.assertIs(manager.decode_host_mem_pool, host_pool_group)
        self.assertIs(manager.cache_controller, hybrid_controller)
        self.assertEqual(manager._extra_pool_specs, ())
        called_kwargs = mock_build_kv_only_stack.call_args.kwargs
        self.assertTrue(called_kwargs["use_mla"])
        self.assertIsNone(called_kwargs["override_kv_cache_dim"])
        mock_build_kv_only_stack.assert_called_once()
        mock_build_shared_anchor_stack.assert_not_called()

    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.torch.distributed.get_world_size",
        return_value=1,
    )
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.build_kv_only_stack"
    )
    @patch(
        "sglang.srt.disaggregation.decode_kvcache_offload_manager.build_shared_anchor_stack"
    )
    def test_init_uses_hybrid_stack_for_nsa_sidecar_capability(
        self,
        mock_build_shared_anchor_stack,
        mock_build_kv_only_stack,
        _mock_world_size,
    ):
        req_to_token_pool = MagicMock()
        token_to_kv_pool_allocator = MagicMock()
        kv_cache = _make_nsa_like_cache()
        token_to_kv_pool_allocator.get_kvcache.return_value = kv_cache
        tree_cache = MagicMock()
        tp_group = MagicMock()

        host_pool_group = MagicMock()
        hybrid_controller = MagicMock()
        mock_build_shared_anchor_stack.return_value = (
            host_pool_group,
            hybrid_controller,
        )

        manager = DecodeKVCacheOffloadManager(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tp_group=tp_group,
            tree_cache=tree_cache,
            server_args=_make_server_args(page_size=64),
        )

        self.assertIs(manager.decode_host_mem_pool, host_pool_group)
        self.assertIs(manager.cache_controller, hybrid_controller)
        self.assertEqual(
            manager._extra_pool_specs,
            ((PoolName.INDEXER, PoolHitPolicy.ALL_PAGES),),
        )
        called_kwargs = mock_build_shared_anchor_stack.call_args.kwargs
        self.assertEqual(called_kwargs["shared_pool_name"], PoolName.INDEXER)
        self.assertTrue(called_kwargs["use_mla"])
        self.assertEqual(called_kwargs["override_kv_cache_dim"], kv_cache.kv_cache_dim)
        mock_build_shared_anchor_stack.assert_called_once()
        mock_build_kv_only_stack.assert_not_called()

    def test_init_rejects_cache_without_supported_decode_offload_capability(self):
        req_to_token_pool = MagicMock()
        token_to_kv_pool_allocator = MagicMock()
        token_to_kv_pool_allocator.get_kvcache.return_value = SimpleNamespace(
            layer_num=16
        )
        tree_cache = MagicMock()
        tp_group = MagicMock()

        with self.assertRaisesRegex(
            ValueError,
            "supported host-pool assembly capability",
        ):
            DecodeKVCacheOffloadManager(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                tp_group=tp_group,
                tree_cache=tree_cache,
                server_args=_make_server_args(page_size=64),
            )


class TestDecodeKVCacheOffloadManagerNSA(unittest.TestCase):

    def test_offload_kv_cache_emits_indexer_pool_transfer_for_nsa(self):
        manager = object.__new__(DecodeKVCacheOffloadManager)
        manager.cache_controller = MagicMock()
        manager.decode_host_mem_pool = MagicMock()
        manager.cache_controller.write.return_value = torch.arange(
            0, 4, dtype=torch.int64
        )
        manager.req_to_token_pool = MagicMock()
        manager.req_to_token_pool.req_to_token = torch.arange(
            16, dtype=torch.int64
        ).unsqueeze(0)
        manager.token_to_kv_pool_allocator = MagicMock()
        manager.page_size = 4
        manager.offload_stride = 4
        manager.offloaded_state = {}
        manager.request_counter = 0
        manager._extra_pool_specs = ((PoolName.INDEXER, PoolHitPolicy.ALL_PAGES),)

        req = SimpleNamespace(
            rid="rid-1",
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[5, 6, 7, 8, 9],
        )

        result = manager.offload_kv_cache(req)

        self.assertTrue(result)
        call_kwargs = manager.cache_controller.write.call_args.kwargs
        self.assertIn("extra_pools", call_kwargs)
        self.assertEqual(len(call_kwargs["extra_pools"]), 1)
        self.assertEqual(call_kwargs["extra_pools"][0].name, PoolName.INDEXER)
        manager.token_to_kv_pool_allocator.free.assert_called_once()
        freed_indices = manager.token_to_kv_pool_allocator.free.call_args.args[0]
        self.assertTrue(torch.equal(freed_indices, torch.arange(0, 4)))
        self.assertEqual(manager.offloaded_state["rid-1"].inc_len, 4)

    def test_trigger_backup_emits_indexer_pool_transfer_for_nsa(self):
        manager = object.__new__(DecodeKVCacheOffloadManager)
        manager.cache_controller = MagicMock()
        manager.cache_controller.write_storage.return_value = 7
        manager.cache_controller.get_hash_str.side_effect = (
            lambda tokens, prior: f"{prior}|{','.join(map(str, tokens))}"
        )
        manager._extra_pool_specs = ((PoolName.INDEXER, PoolHitPolicy.ALL_PAGES),)
        manager.page_size = 4
        manager.ongoing_backup = {}

        req = SimpleNamespace(rid="rid-2")
        host_indices = torch.arange(0, 4, dtype=torch.int64)

        last_hash = manager._trigger_backup(
            req=req,
            host_indices=host_indices,
            incremental_tokens=[10, 11, 12, 13, 14, 15, 16, 17],
            start_time=123.0,
            prior_hash="seed",
        )

        call_args = manager.cache_controller.write_storage.call_args
        self.assertTrue(torch.equal(call_args.args[0], host_indices))
        self.assertEqual(call_args.args[1], [10, 11, 12, 13, 14, 15, 16, 17])
        self.assertIn("extra_pools", call_args.kwargs)
        self.assertEqual(len(call_args.kwargs["extra_pools"]), 1)
        self.assertEqual(call_args.kwargs["extra_pools"][0].name, PoolName.INDEXER)
        self.assertEqual(
            call_args.kwargs["hash_value"],
            [
                "seed|10,11,12,13",
                "seed|10,11,12,13|14,15,16,17",
            ],
        )
        self.assertEqual(last_hash, "seed|10,11,12,13|14,15,16,17")
        self.assertEqual(manager.ongoing_backup[7], ("rid-2", host_indices, 123.0))


if __name__ == "__main__":
    unittest.main()
