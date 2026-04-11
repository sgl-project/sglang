import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")


class TestMamba(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_hybrid_linear_kv_pool(self):
        size = 16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=None,
        )
        assert pool._transfer_full_attention_id(global_interval - 1) == 0
        assert pool._transfer_full_attention_id(2 * global_interval - 1) == 1
        with self.assertRaises(ValueError) as context:
            pool._transfer_full_attention_id(1)
        self.assertIn(
            "layer_id=1 not in full attention layers:", str(context.exception)
        )

    def test_mamba_pool(self):
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        global_interval = 4
        num_layers = 48
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4096,
            n_groups=16,
            num_heads=32,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )

        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )

        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        sampling_params = SamplingParams(
            temperature=0,
            max_new_tokens=1,
        )
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sampling_params,
        )

        # alloc req
        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        # free req
        req_to_token_pool.free_mamba_cache(req)
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size

        # alloc req without free mamba cache
        req.mamba_pool_idx = None
        req_to_token_pool.alloc([req])
        req_to_token_pool.free(req)
        assert req_to_token_pool.available_size() == max_num_reqs
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

        # alloc again
        req_to_token_pool.alloc([req])
        assert req_to_token_pool.available_size() == max_num_reqs - 1
        assert req_to_token_pool.mamba_pool.available_size() == mamba_cache_size - 1

    def test_mamba_radix_cache_1(self):
        tree, allocator, req_to_token_pool, make_dummy_req = (
            self._setup_tree_and_allocator()
        )
        mamba_pool = req_to_token_pool.mamba_pool
        # test
        print(
            f"[Start] allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req1 = make_dummy_req()
        req1_token_ids, req1_kv_indices = [1, 2, 3], allocator.alloc(3)
        assert len(req1_token_ids) == len(req1_kv_indices)
        print(
            f"req1: inserting, req1_token_ids: {req1_token_ids}, req1_kv_indices: {req1_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req1_token_ids),
                value=req1_kv_indices,
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req1: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req2 = make_dummy_req()
        req2_token_ids, req2_kv_indices = [1, 2, 3, 4, 5, 6, 7], allocator.alloc(7)
        assert len(req2_token_ids) == len(req2_kv_indices)
        print(
            f"req2: inserting, req2_token_ids: {req2_token_ids}, req2_kv_indices: {req2_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req2_token_ids),
                value=req2_kv_indices,
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req2: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )

        req3 = make_dummy_req()
        req3_token_ids, req3_kv_indices = [10, 11, 12], allocator.alloc(3)
        assert len(req3_token_ids) == len(req3_kv_indices)
        print(
            f"req3: inserting, req3_token_ids: {req3_token_ids}, req3_kv_indices: {req3_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req3_token_ids),
                value=req3_kv_indices,
                mamba_value=req3.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req3: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )
        req4 = make_dummy_req()
        req4_token_ids, req4_kv_indices = [1, 2, 3, 4, 5, 60, 70], allocator.alloc(7)
        assert len(req4_token_ids) == len(req4_kv_indices)
        print(
            f"req4: inserting, req4_token_ids: {req4_token_ids}, req4_kv_indices: {req4_kv_indices}"
        )
        result = tree.insert(
            InsertParams(
                key=RadixKey(req4_token_ids),
                value=req4_kv_indices,
                mamba_value=req4.mamba_pool_idx.unsqueeze(0),
            )
        )
        prefix_len = result.prefix_len
        print(
            f"req4: prefix_len: {prefix_len}, allocator mamba available size: {mamba_pool.available_size()}, full available size: {allocator.available_size()}"
        )

        tree.pretty_print()
        full_num_tokens = 1
        print(f"evicting {full_num_tokens} full token")
        result = tree.evict(EvictParams(num_tokens=full_num_tokens))
        assert (
            result.num_tokens_evicted >= full_num_tokens
        ), f"evicted {result.num_tokens_evicted} full tokens, expected {full_num_tokens}"
        tree.pretty_print()

        mamba_num = 1
        print(f"evicting {mamba_num} mamba")
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
        assert (
            result.mamba_num_evicted >= mamba_num
        ), f"evicted {result.mamba_num_evicted} mamba states, expected {mamba_num}"
        tree.pretty_print()

        req5_token_ids = [1, 2, 3, 4, 5]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req5_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req5: token_ids: {req5_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 0

        req6_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req6_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req6: token_ids: {req6_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 7
        assert len(last_node.key) == 2

        req7_token_ids = [1, 2, 3, 4, 5, 6, 7]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req7_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req7: token_ids: {req7_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 7
        assert len(last_node.key) == 2

        mamba_num = 1
        print(f"evicting {mamba_num} mamba")
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=mamba_num))
        assert (
            result.mamba_num_evicted >= mamba_num
        ), f"evicted {result.mamba_num_evicted} mamba states, expected {mamba_num}"
        tree.pretty_print()

        req8_token_ids = [1, 2, 3, 4, 5, 60, 70]
        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(req8_token_ids)))
        kv_indices, last_node = result.device_indices, result.last_device_node
        print(
            f"req8: token_ids: {req8_token_ids}, matched kv_indices: {kv_indices}, last_node.key: {last_node.key}"
        )
        assert len(kv_indices) == 0
        assert len(last_node.key) == 0

        req9_token_ids = [1, 2, 3, 4, 5, 6, 7]
        req9 = make_dummy_req()
        result = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(req9_token_ids), req=req9, cow_mamba=True)
        )
        kv_indices, last_node = result.device_indices, result.last_device_node
        assert req9.mamba_pool_idx is not None
        assert torch.all(
            mamba_pool.mamba_cache.conv[0][:, req9.mamba_pool_idx]
            == mamba_pool.mamba_cache.conv[0][:, last_node.mamba_value]
        )
        assert torch.all(
            mamba_pool.mamba_cache.temporal[:, req9.mamba_pool_idx]
            == mamba_pool.mamba_cache.temporal[:, last_node.mamba_value]
        )

        print(tree.available_and_evictable_str())
        print(available_and_evictable_str(tree))
        tree.sanity_check()

    def _setup_tree_and_allocator(self):
        """Helper to create a MambaRadixCache with allocator for testing."""
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )
        size = 128
        dtype = torch.bfloat16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        max_num_reqs = 10
        mamba_cache_size = 20
        max_context_len = 128
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
        )
        tree = MambaRadixCache(params=params)

        def make_dummy_req():
            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            req = Req(
                rid=0,
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=sampling_params,
            )
            req_to_token_pool.alloc([req])
            return req

        return tree, allocator, req_to_token_pool, make_dummy_req

    def _setup_tree_and_allocator_extra_buffer(self):
        """Helper to create a MambaRadixCache with enable_mamba_extra_buffer=True."""
        set_global_server_args_for_scheduler(
            ServerArgs(
                model_path="dummy",
                page_size=1,
                mamba_scheduler_strategy="extra_buffer",
            )
        )
        size = 128
        dtype = torch.bfloat16
        head_num = 2
        head_dim = 256
        num_layers = 48
        global_interval = 4
        max_num_reqs = 10
        mamba_cache_size = 30
        max_context_len = 128
        device = get_device()
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=True,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=size,
            dtype=dtype,
            page_size=1,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=size,
            dtype=dtype,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            enable_mamba_extra_buffer=True,
        )
        tree = MambaRadixCache(params=params)

        rid_counter = [0]

        def make_dummy_req():
            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )
            req = Req(
                rid=rid_counter[0],
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=sampling_params,
            )
            rid_counter[0] += 1
            req_to_token_pool.alloc([req])
            req.last_node = tree.root_node
            tree.inc_lock_ref(tree.root_node)
            return req

        return tree, allocator, req_to_token_pool, make_dummy_req

    def test_extra_buffer_pending_radix_slot_lifecycle(self):
        """Test pending_radix_mamba_slot allocation, ownership transfer to radix
        tree via cache_unfinished_req, and subsequent match_prefix with cow_mamba.
        This exercises the core path changed by the ping-pong elimination."""
        tree, allocator, req_to_token_pool, make_dummy_req = (
            self._setup_tree_and_allocator_extra_buffer()
        )
        mamba_pool = req_to_token_pool.mamba_pool
        initial_mamba_avail = mamba_pool.available_size()

        # --- Step 1: Simulate chunked prefill with pending_radix_mamba_slot ---
        req1 = make_dummy_req()
        assert req1.mamba_pool_idx is not None
        # After alloc: 1 working slot consumed
        self.assertEqual(mamba_pool.available_size(), initial_mamba_avail - 1)

        # Simulate the scheduler allocating a pending radix slot (what prepare_for_extend does)
        pending_slot = mamba_pool.alloc(1)
        self.assertIsNotNone(pending_slot)
        req1.pending_radix_mamba_slot = pending_slot
        # Now: 1 working + 1 pending = 2 slots consumed
        self.assertEqual(mamba_pool.available_size(), initial_mamba_avail - 2)

        # Write distinctive data into the pending slot to verify it survives
        # ownership transfer. We write to the conv state of layer 0.
        mamba_pool.mamba_cache.conv[0][:, pending_slot[0]] = 42.0

        # Simulate filling token ids and setting mamba_last_track_seqlen
        req1.fill_ids = [1, 2, 3, 4, 5]
        req1.origin_input_ids = [1, 2, 3, 4, 5]
        req1.mamba_last_track_seqlen = 5
        req1.kv_committed_len = 5
        req1.kv_allocated_len = 5

        # Allocate KV indices for the tokens
        kv_indices = allocator.alloc(5)
        self.assertIsNotNone(kv_indices)
        req_to_token_pool.write(
            (req1.req_pool_idx, slice(0, 5)),
            kv_indices,
        )

        # cache_unfinished_req should transfer pending slot to radix tree
        tree.cache_unfinished_req(req1, chunked=True)

        # After cache_unfinished_req: old pending slot transferred to radix tree,
        # new one pre-allocated for next chunk/decode.
        self.assertIsNotNone(req1.pending_radix_mamba_slot)
        # Working slot still alive
        self.assertIsNotNone(req1.mamba_pool_idx)

        tree.pretty_print()

        # Note: cannot call tree.sanity_check() here because req1 holds a lock on its last_node.
        # We verify the tree state via match_prefix instead.

        # --- Step 2: Match prefix and verify cow_mamba copies the correct state ---
        # First, release req1's lock so we start clean for req2
        tree.dec_lock_ref(req1.last_node)
        req_to_token_pool.free_mamba_cache(req1)
        req_to_token_pool.free(req1)

        tree.sanity_check()

        req2 = make_dummy_req()
        # dec_lock_ref the initial root_node lock from make_dummy_req
        tree.dec_lock_ref(req2.last_node)
        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7]),
                req=req2,
                cow_mamba=True,
            )
        )
        kv_indices_matched = result.device_indices
        last_node = result.last_device_node
        # Simulate scheduler locking the matched node
        tree.inc_lock_ref(last_node)
        req2.last_node = last_node
        req2.cache_protected_len = len(kv_indices_matched)
        self.assertEqual(len(kv_indices_matched), 5)
        self.assertIsNotNone(req2.mamba_pool_idx)

        # Verify the copied mamba state matches what we wrote (42.0)
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][:, req2.mamba_pool_idx]
                == mamba_pool.mamba_cache.conv[0][:, last_node.mamba_value]
            )
        )
        expected_val = torch.tensor(42.0, dtype=mamba_pool.mamba_cache.conv[0].dtype)
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][0, req2.mamba_pool_idx] == expected_val
            )
        )

        # --- Step 3: cache_finished_req with pending_radix_mamba_slot ---
        # Simulate a decode phase: req2 matched prefix [1,2,3,4,5] and decoded [6,7]
        req2.origin_input_ids = [1, 2, 3, 4, 5]
        req2.output_ids = [6, 7]
        new_pending = mamba_pool.alloc(1)
        self.assertIsNotNone(new_pending)
        req2.pending_radix_mamba_slot = new_pending
        mamba_pool.mamba_cache.conv[0][:, new_pending[0]] = 99.0
        req2.mamba_last_track_seqlen = 7
        req2.kv_committed_len = 7
        req2.kv_allocated_len = 7

        # Write the matched prefix indices + new decode KV into req2's token pool
        req_to_token_pool.write(
            (req2.req_pool_idx, slice(0, 5)),
            kv_indices_matched,
        )
        kv_new = allocator.alloc(2)
        self.assertIsNotNone(kv_new)
        req_to_token_pool.write(
            (req2.req_pool_idx, slice(5, 7)),
            kv_new,
        )

        tree.cache_finished_req(req2)
        req_to_token_pool.free(req2)

        # After cache_finished_req: working + pending slots should be freed from req
        self.assertIsNone(req2.mamba_pool_idx)
        self.assertIsNone(req2.pending_radix_mamba_slot)

        tree.pretty_print()
        tree.sanity_check()

        # Verify the new node has the correct mamba value (99.0)
        result3 = tree.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5, 6, 7]))
        )
        self.assertEqual(len(result3.device_indices), 7)
        last_node3 = result3.last_device_node
        self.assertIsNotNone(last_node3.mamba_value)
        self.assertTrue(
            torch.all(mamba_pool.mamba_cache.conv[0][0, last_node3.mamba_value] == 99.0)
        )

        # --- Step 4: cache_finished_req without pending slot (short decode) ---
        req3 = make_dummy_req()
        tree.dec_lock_ref(req3.last_node)
        result4 = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7]),
                req=req3,
                cow_mamba=True,
            )
        )
        self.assertEqual(len(result4.device_indices), 7)
        tree.inc_lock_ref(result4.last_device_node)
        req3.last_node = result4.last_device_node
        req3.origin_input_ids = [1, 2, 3, 4, 5, 6, 7]
        req3.output_ids = [8]  # short output, no tracking occurred
        req3.mamba_last_track_seqlen = None
        req3.pending_radix_mamba_slot = None
        req3.cache_protected_len = 7
        req3.kv_committed_len = 8
        req3.kv_allocated_len = 8
        req3.kv_committed_freed = False

        # Write matched prefix + decode output KV into req3's token pool
        req_to_token_pool.write(
            (req3.req_pool_idx, slice(0, 7)),
            result4.device_indices,
        )
        kv_extra = allocator.alloc(1)
        self.assertIsNotNone(kv_extra)
        req_to_token_pool.write((req3.req_pool_idx, slice(7, 8)), kv_extra)

        avail_before = mamba_pool.available_size()
        tree.cache_finished_req(req3)
        req_to_token_pool.free(req3)
        # Working slot freed, no pending slot to worry about
        self.assertIsNone(req3.mamba_pool_idx)
        self.assertIsNone(req3.pending_radix_mamba_slot)
        # Should have freed exactly 1 mamba slot (working)
        self.assertEqual(mamba_pool.available_size(), avail_before + 1)

        tree.sanity_check()

        # --- Step 5: free_mamba_cache with both working + pending ---
        req4 = make_dummy_req()
        tree.dec_lock_ref(req4.last_node)
        pending_slot4 = mamba_pool.alloc(1)
        self.assertIsNotNone(pending_slot4)
        req4.pending_radix_mamba_slot = pending_slot4
        avail_before = mamba_pool.available_size()

        req_to_token_pool.free_mamba_cache(req4)
        # Should free both working and pending slots
        self.assertIsNone(req4.mamba_pool_idx)
        self.assertIsNone(req4.pending_radix_mamba_slot)
        self.assertEqual(mamba_pool.available_size(), avail_before + 2)

        req_to_token_pool.free(req4)
        tree.sanity_check()
        print(tree.available_and_evictable_str())
        print(available_and_evictable_str(tree))

    def test_extra_buffer_eviction_frees_mamba_slots(self):
        """Test that evicting radix tree nodes correctly frees mamba slots
        that were transferred via pending_radix_mamba_slot."""
        tree, allocator, req_to_token_pool, make_dummy_req = (
            self._setup_tree_and_allocator_extra_buffer()
        )
        mamba_pool = req_to_token_pool.mamba_pool

        # Insert two separate prefixes via the pending slot path
        for token_ids in [[1, 2, 3], [10, 11, 12]]:
            req = make_dummy_req()
            pending = mamba_pool.alloc(1)
            self.assertIsNotNone(pending)
            req.pending_radix_mamba_slot = pending
            req.fill_ids = token_ids
            req.origin_input_ids = token_ids
            req.mamba_last_track_seqlen = len(token_ids)
            req.kv_committed_len = len(token_ids)
            req.kv_allocated_len = len(token_ids)

            kv = allocator.alloc(len(token_ids))
            req_to_token_pool.write((req.req_pool_idx, slice(0, len(token_ids))), kv)
            tree.cache_unfinished_req(req, chunked=True)

            # After cache_unfinished_req, req.last_node points to the new cached node.
            # Simulate request finishing by freeing working slot and req pool,
            # and unlocking the tree node.
            tree.dec_lock_ref(req.last_node)
            req_to_token_pool.free_mamba_cache(req)
            req_to_token_pool.free(req)

        tree.sanity_check()

        avail_before_evict = mamba_pool.available_size()

        # Evict 1 mamba state
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        self.assertGreater(mamba_pool.available_size(), avail_before_evict)

        tree.sanity_check()

        # Evict the other one
        avail_before_evict2 = mamba_pool.available_size()
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        self.assertGreater(mamba_pool.available_size(), avail_before_evict2)

        tree.sanity_check()

    def test_extra_buffer_multiple_cache_unfinished(self):
        """Test multiple cache_unfinished_req calls (simulating chunked prefill
        with multiple chunks), each with a fresh pending_radix_mamba_slot."""
        tree, allocator, req_to_token_pool, make_dummy_req = (
            self._setup_tree_and_allocator_extra_buffer()
        )
        mamba_pool = req_to_token_pool.mamba_pool

        req = make_dummy_req()

        # Chunk 1: tokens [1,2,3]
        pending1 = mamba_pool.alloc(1)
        self.assertIsNotNone(pending1)
        req.pending_radix_mamba_slot = pending1
        mamba_pool.mamba_cache.conv[0][:, pending1[0]] = 10.0

        req.fill_ids = [1, 2, 3]
        req.origin_input_ids = [1, 2, 3, 4, 5, 6]
        req.mamba_last_track_seqlen = 3
        req.kv_committed_len = 3
        req.kv_allocated_len = 3

        kv1 = allocator.alloc(3)
        req_to_token_pool.write((req.req_pool_idx, slice(0, 3)), kv1)
        tree.cache_unfinished_req(req, chunked=True)

        # cache_unfinished_req transfers the old pending slot to the radix tree
        # and pre-allocates a new one for the next chunk.
        self.assertIsNotNone(req.pending_radix_mamba_slot)
        self.assertFalse(torch.equal(req.pending_radix_mamba_slot, pending1))

        # Verify chunk 1 mamba state is in radix tree
        result1 = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        self.assertEqual(len(result1.device_indices), 3)
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][0, result1.last_device_node.mamba_value]
                == 10.0
            )
        )

        # Chunk 2: tokens [1,2,3,4,5,6]
        # Use the pre-allocated pending slot from cache_unfinished_req
        pending2 = req.pending_radix_mamba_slot
        mamba_pool.mamba_cache.conv[0][:, pending2[0]] = 20.0

        req.fill_ids = [1, 2, 3, 4, 5, 6]
        req.mamba_last_track_seqlen = 6
        req.kv_committed_len = 6
        req.kv_allocated_len = 6

        kv2 = allocator.alloc(3)
        req_to_token_pool.write((req.req_pool_idx, slice(3, 6)), kv2)
        tree.cache_unfinished_req(req, chunked=True)

        # Again, old pending transferred to tree, new one pre-allocated
        self.assertIsNotNone(req.pending_radix_mamba_slot)
        self.assertFalse(torch.equal(req.pending_radix_mamba_slot, pending2))

        # Verify chunk 2 mamba state is in radix tree (should supersede chunk 1's)
        result2 = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5, 6])))
        self.assertEqual(len(result2.device_indices), 6)
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][0, result2.last_device_node.mamba_value]
                == 20.0
            )
        )

        tree.pretty_print()

        # Clean up: unlock, free mamba cache and req pool, then sanity check
        tree.dec_lock_ref(req.last_node)
        req_to_token_pool.free_mamba_cache(req)
        req_to_token_pool.free(req)

        tree.sanity_check()

    def test_insert_prev_prefix_len(self):
        """Test that prev_prefix_len correctly controls which KV indices are freed
        during insert, covering: full free, partial free across multi-node, and no free.
        """
        tree, allocator, req_to_token_pool, make_dummy_req = (
            self._setup_tree_and_allocator()
        )

        initial_avail = allocator.available_size()

        # Step 1: Insert [1,2,3] to create first node
        req1 = make_dummy_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=allocator.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        assert allocator.available_size() == initial_avail - 3

        # Step 2: Insert [1,2,3,4,5,6,7] with prev_prefix_len=0 (free all matched)
        # Creates tree: [1,2,3] -> [4,5,6,7]
        req2 = make_dummy_req()
        result = tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7]),
                value=allocator.alloc(7),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
                prev_prefix_len=0,
            )
        )
        assert result.prefix_len == 3
        # alloc 7, freed 3 (dup prefix [0..2]), stored 4 in new node => net -4
        assert allocator.available_size() == initial_avail - 3 - 4
        avail_after_step2 = allocator.available_size()

        # Step 3: Insert [1,2,3,4,5,6,7,8] with prev_prefix_len=2
        # Matched prefix = 7 (across two nodes: [1,2,3] len=3, [4,5,6,7] len=4)
        # Protected [0..1], freed [2..6] = 5 slots, new [7] = 1 slot stored
        req3 = make_dummy_req()
        result = tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7, 8]),
                value=allocator.alloc(8),
                mamba_value=req3.mamba_pool_idx.unsqueeze(0),
                prev_prefix_len=2,
            )
        )
        assert result.prefix_len == 7
        # alloc 8, freed 5, stored 1 => net -3
        assert allocator.available_size() == avail_after_step2 - 3
        avail_after_step3 = allocator.available_size()

        # Step 4: Insert [1,2,3,4,5,6,7,8,9] with prev_prefix_len=8 (covers all matched)
        # Matched prefix = 8, prev_prefix_len=8 => nothing freed
        req4 = make_dummy_req()
        result = tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                value=allocator.alloc(9),
                mamba_value=req4.mamba_pool_idx.unsqueeze(0),
                prev_prefix_len=8,
            )
        )
        assert result.prefix_len == 8
        # alloc 9, freed 0, stored 1 => net -9
        assert allocator.available_size() == avail_after_step3 - 9

        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
