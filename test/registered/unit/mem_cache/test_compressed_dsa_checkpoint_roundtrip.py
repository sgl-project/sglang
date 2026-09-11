"""A cached recurrent state must describe exactly the prefix that owns it."""

import unittest
from array import array
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_exec, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCompressedDsaCheckpointRoundtrip(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def _fixture(self, *, lazy=False, swa=False, mamba_first=False, no_buffer=False):
        page_size = 1 if swa else 64
        args = ServerArgs(
            model_path="dummy",
            page_size=page_size,
            mamba_radix_cache_strategy=(
                "no_buffer"
                if no_buffer
                else "extra_buffer_lazy"
                if lazy
                else "extra_buffer"
            ),
        )
        args._mamba_cache_chunk_size = 64
        publish(args, role="scheduler")
        self.assertEqual(get_exec().mamba.enable_mamba_extra_buffer_lazy, lazy)
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=16,
            n_groups=1,
            num_heads=1,
            head_dim=16,
            state_size=2,
            conv_kernel=4,
        )
        req_pool = HybridReqToTokenPool(
            size=4,
            mamba_size=32,
            mamba_spec_state_size=4,
            max_context_len=2048,
            device="cpu",
            enable_memory_saver=False,
            # These are exact symbolic states, not a kernel precision test.
            # FP32 preserves prefix, layer, and convolution-position markers.
            cache_params=Mamba2CacheParams(
                shape=shape,
                layers=[0, 1],
                dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
            ),
            mamba_layer_ids=[0, 1],
            enable_mamba_extra_buffer=not no_buffer,
            enable_mamba_extra_buffer_lazy=lazy,
        )
        # No attention kernel runs here; the real DSA pool type supplies the
        # ownership geometry while real CPU pools own KV indices and SSM slots.
        dsa = object.__new__(DSATokenToKVPool)
        dsa.page_size = 64
        dsa.index_kpool = 4
        dsa.kpool_use_compress = True
        kv_pool = object.__new__(HybridLinearKVPool)
        kv_pool.full_kv_pool = dsa
        allocator = TokenToKVPoolAllocator(
            size=4096,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        components = (ComponentType.FULL, ComponentType.MAMBA)
        if swa:
            kv_pool = SWAKVPool(
                size=4096,
                size_swa=4096,
                page_size=1,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=16,
                swa_attention_layer_ids=[1],
                full_attention_layer_ids=[0],
                device="cpu",
            )
            allocator = SWATokenToKVPoolAllocator(
                size=4096,
                size_swa=4096,
                page_size=1,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=kv_pool,
                need_sort=False,
            )
            components = (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
            if mamba_first:
                components = (
                    ComponentType.FULL,
                    ComponentType.MAMBA,
                    ComponentType.SWA,
                )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                tree_components=components,
                sliding_window_size=64 if swa else None,
                enable_mamba_extra_buffer=not no_buffer,
                enable_mamba_extra_buffer_lazy=lazy,
            )
        )
        return cache, allocator, req_pool

    def _request(self, cache, allocator, req_pool, *, length=1000, token=1):
        sampling = SamplingParams(max_new_tokens=1)
        sampling.normalize(None)
        req = Req(
            rid="chunked",
            origin_input_text="",
            origin_input_ids=array("q", [token] * length),
            sampling_params=sampling,
            vocab_size=128,
        )
        req_pool.alloc([req])
        req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
        values = allocator.alloc(length)
        req_pool.write((req.kv.req_pool_idx, slice(0, length)), values)
        req.kv.kv_allocated_len = length
        req.last_node = cache.root_node_handle()
        req.prefix_indices = torch.empty(0, dtype=torch.int64)
        return req

    @staticmethod
    def _marker(tokens, depth, layer):
        # Equal prefixes have equal states. The unrelated prefixes and layers
        # used here have distinct markers, all exact FP32 integers.
        return depth + 2048 * sum(tokens[:depth]) + layer * (1 << 22)

    def _fill_temporal(self, req_pool, slot, tokens, depth):
        states = req_pool.mamba_pool.mamba_cache.temporal
        for layer in range(states.shape[0]):
            states[layer, slot].fill_(self._marker(tokens, depth, layer))

    def _assert_slot_matches_prefix(self, req_pool, slot, tokens, expected):
        states = req_pool.mamba_pool.mamba_cache.temporal
        for layer in range(states.shape[0]):
            marker = self._marker(tokens, expected, layer)
            self.assertTrue(torch.all(states[layer, slot] == marker).item())
            for conv in req_pool.mamba_pool.mamba_cache.conv:
                window = (
                    torch.arange(conv.shape[-1], dtype=conv.dtype)
                    + marker
                    - conv.shape[-1]
                    + 1
                )
                self.assertTrue(
                    torch.equal(conv[layer, slot], window.expand_as(conv[layer, slot]))
                )

    def _run_chunk(self, cache, req_pool, req, end, *, publish_checkpoint=True):
        prefix = len(req.prefix_indices)
        req.set_extend_range(prefix, end)
        req.kv.kv_committed_len = end
        if not cache.enable_mamba_extra_buffer:
            # This strategy keeps the live state instead of selecting a saved
            # intermediate snapshot. Give that state its actual prefix depth.
            self._fill_temporal(
                req_pool, req.kv.mamba_pool_idx, req.origin_input_ids, end
            )
            for conv in req_pool.mamba_pool.mamba_cache.conv:
                for layer in range(conv.shape[0]):
                    conv[layer, req.kv.mamba_pool_idx] = (
                        torch.arange(conv.shape[-1])
                        + self._marker(req.origin_input_ids, end, layer)
                        - conv.shape[-1]
                        + 1
                    )
            if publish_checkpoint:
                cache.cache_unfinished_req(req, chunked=True)
            return end, None
        batch = ScheduleBatch(reqs=[req])
        batch.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=64)
        )
        batch.tree_cache = cache
        batch.req_to_token_pool = req_pool
        entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)

        # Let the real backend select symbolic intermediate/final states.
        # A wrong snapshot index must not acquire the expected prefix marker.
        forward = object.__new__(ForwardBatch)
        forward.extend_seq_lens = torch.tensor([end - prefix])
        forward.extend_prefix_lens = torch.tensor([prefix])
        forward.mamba_track_seqlens = torch.tensor([entry.track_seqlen])
        forward.mamba_track_mask = torch.tensor([entry.track_mask])
        forward.mamba_track_indices = torch.tensor([entry.track_index])
        indices = MambaAttnBackendBase._init_track_ssm_indices(
            SimpleNamespace(device="cpu", mamba_chunk_size=64),
            req.kv.mamba_pool_idx.unsqueeze(0),
            forward,
        )
        _, h_src, h_dst, _, final_src, final_dst, *_ = indices
        states = req_pool.mamba_pool.mamba_cache.temporal
        self._fill_temporal(req_pool, req.kv.mamba_pool_idx, req.origin_input_ids, end)
        for src, dst in zip(h_src.tolist(), h_dst.tolist()):
            self._fill_temporal(req_pool, dst, req.origin_input_ids, prefix + src * 64)
        for src, dst in zip(final_src.tolist(), final_dst.tolist()):
            states[:, dst].copy_(states[:, src])
        for conv in req_pool.mamba_pool.mamba_cache.conv:
            positions = MambaAttnBackendBase._init_track_conv_indices(
                SimpleNamespace(device="cpu", conv_states_shape=conv.shape[2:]),
                torch.tensor([0, end - prefix]),
                forward,
            )
            if entry.track_mask:
                positions = prefix + positions[0] + 1
                depth = positions[-1].item()
                for layer in range(conv.shape[0]):
                    conv[layer, entry.track_index] = (
                        positions
                        + self._marker(req.origin_input_ids, depth, layer)
                        - depth
                    )
        checkpoint = req.kv.mamba_last_track_seqlen
        if publish_checkpoint:
            cache.cache_unfinished_req(req, chunked=True)
        return checkpoint, entry

    def _assert_cached_state_matches_prefix(self, cache, req_pool, tokens, expected):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(len(match.device_indices), expected)
        if expected:
            slot = cache.tree_core.get_component_device_value(
                match.best_match_node, ComponentType.MAMBA
            )
            self.assertIsNotNone(slot)
            self._assert_slot_matches_prefix(req_pool, slot, tokens, expected)

    def test_successive_320_token_chunks_publish_the_matching_state(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                cache, allocator, req_pool = self._fixture(lazy=lazy)
                req = self._request(cache, allocator, req_pool)
                for end, expected in [(320, 256), (640, 512), (960, 768), (1000, 768)]:
                    with self.subTest(end=end):
                        self._run_chunk(cache, req_pool, req, end)
                        self.assertEqual(len(req.prefix_indices), end)
                        self._assert_cached_state_matches_prefix(
                            cache, req_pool, req.origin_input_ids[:end], expected
                        )

    def test_short_chunks_track_when_they_cross_an_absolute_boundary(self):
        cache, allocator, req_pool = self._fixture()
        req = self._request(cache, allocator, req_pool)
        # Alternate short chunks that stop before and cross the next boundary.
        for end, expected in ((192, 0), (256, 256), (448, 256), (512, 512)):
            with self.subTest(end=end):
                self._run_chunk(cache, req_pool, req, end)
                self._assert_cached_state_matches_prefix(
                    cache, req_pool, req.origin_input_ids[:end], expected
                )

    def test_second_request_copies_the_state_at_its_matched_prefix(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                cache, allocator, req_pool = self._fixture(lazy=lazy)
                req = self._request(cache, allocator, req_pool)
                self._run_chunk(cache, req_pool, req, 320)
                self._run_chunk(cache, req_pool, req, 640)
                # Another branch has the same checkpoint depth but different
                # state, so selecting it cannot pass a depth-only comparison.
                unrelated = self._request(
                    cache, allocator, req_pool, length=640, token=3
                )
                self._run_chunk(cache, req_pool, unrelated, 640)
                other = self._request(cache, allocator, req_pool, length=700)
                other.origin_input_ids[512:] = array("q", [2] * 188)
                match = cache.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(other.origin_input_ids), req=other, cow_mamba=True
                    )
                )
                self.assertEqual(len(match.device_indices), 512)
                batch = ScheduleBatch(reqs=[other])
                batch._collect_deferred_mamba_cow_and_clear([other])
                self.assertIsNotNone(batch.mamba_cow_src_indices)
                req_pool.mamba_pool.copy_from(
                    batch.mamba_cow_src_indices, batch.mamba_cow_dst_indices
                )
                self._assert_slot_matches_prefix(
                    req_pool, other.kv.mamba_pool_idx, other.origin_input_ids, 512
                )
                # Copy-on-write must clone both state kinds and leave the
                # tree's checkpoint independent of writes to the destination.
                req_pool.mamba_pool.mamba_cache.temporal[
                    :, other.kv.mamba_pool_idx
                ] = -1
                for conv in req_pool.mamba_pool.mamba_cache.conv:
                    conv[:, other.kv.mamba_pool_idx] = -1
                self._assert_cached_state_matches_prefix(
                    cache, req_pool, other.origin_input_ids, 512
                )

    def test_branch_checkpoint_must_be_on_the_absolute_tree_grid(self):
        for branch, expected in ((448, 768), (512, 512)):
            with self.subTest(branch=branch):
                cache, allocator, req_pool = self._fixture()
                req = self._request(cache, allocator, req_pool)
                self._run_chunk(cache, req_pool, req, 320)
                req.mamba_branching_seqlen = branch
                checkpoint, _ = self._run_chunk(cache, req_pool, req, 960)
                self.assertEqual(checkpoint, expected)
                self._assert_cached_state_matches_prefix(
                    cache, req_pool, req.origin_input_ids[:960], expected
                )

    def test_unrepresentable_snapshot_is_not_published(self):
        cache, allocator, req_pool = self._fixture()
        req = self._request(cache, allocator, req_pool)
        self._run_chunk(cache, req_pool, req, 324)
        checkpoint, entry = self._run_chunk(cache, req_pool, req, 644)
        self.assertFalse(entry.track_mask)
        self.assertIsNone(checkpoint)
        self._assert_cached_state_matches_prefix(
            cache, req_pool, req.origin_input_ids[:644], 256
        )

    def test_finished_request_donates_exact_checkpoint_and_frees_other_slots(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                cache, allocator, req_pool = self._fixture(lazy=lazy)
                available = req_pool.mamba_allocator.available_size()
                req = self._request(cache, allocator, req_pool)
                self._run_chunk(cache, req_pool, req, 320)
                self._run_chunk(cache, req_pool, req, 640, publish_checkpoint=False)
                cache.cache_finished_req(req, kv_len_to_handle=640)
                self._assert_cached_state_matches_prefix(
                    cache, req_pool, req.origin_input_ids[:640], 512
                )
                self.assertEqual(
                    req_pool.mamba_allocator.available_size(), available - 2
                )
                # A freed checkpoint still has correct bytes until its slot
                # is reused. Recycle all free slots to check ownership, not
                # just the count or the state immediately after cleanup.
                cached_slots = set()
                for depth in (256, 512):
                    match = cache.match_prefix(
                        MatchPrefixParams(key=RadixKey(req.origin_input_ids[:depth]))
                    )
                    self.assertEqual(len(match.device_indices), depth)
                    slot = cache.tree_core.get_component_device_value(
                        match.best_match_node, ComponentType.MAMBA
                    )
                    self.assertIsNotNone(slot)
                    cached_slots.add(slot.item())
                self.assertEqual(len(cached_slots), 2)
                recycled = req_pool.mamba_allocator.alloc(available - 2)
                self.assertIsNotNone(recycled)
                self.assertEqual(len(recycled.unique()), available - 2)
                self.assertTrue(cached_slots.isdisjoint(recycled.tolist()))
                self.assertIsNone(req_pool.mamba_allocator.alloc(1))
                req_pool.mamba_pool.mamba_cache.temporal[:, recycled] = -1
                for conv in req_pool.mamba_pool.mamba_cache.conv:
                    conv[:, recycled] = -1
                for depth in (256, 512):
                    self._assert_cached_state_matches_prefix(
                        cache, req_pool, req.origin_input_ids[:depth], depth
                    )

    def test_invalid_checkpoint_is_not_rounded_to_a_different_prefix(self):
        for finished in (False, True):
            for checkpoint, token_count in ((576, 640), (768, 640), (None, 640)):
                with self.subTest(finished=finished, checkpoint=checkpoint):
                    cache, allocator, req_pool = self._fixture()
                    available = req_pool.mamba_allocator.available_size()
                    req = self._request(cache, allocator, req_pool)
                    req.set_extend_range(0, token_count)
                    req.kv.kv_committed_len = token_count
                    req.kv.mamba_last_track_seqlen = checkpoint
                    before = req_pool.mamba_allocator.available_size()
                    if finished:
                        cache.cache_finished_req(req, kv_len_to_handle=token_count)
                        self.assertEqual(
                            req_pool.mamba_allocator.available_size(), available
                        )
                    else:
                        cache.cache_unfinished_req(req, chunked=True)
                        self.assertEqual(
                            req_pool.mamba_allocator.available_size(), before
                        )
                    self._assert_cached_state_matches_prefix(
                        cache, req_pool, req.origin_input_ids[:token_count], 0
                    )

    def test_swa_branch_cannot_shorten_a_recurrent_checkpoint(self):
        for strategy, finished, mamba_first in product(
            ("extra_buffer", "extra_buffer_lazy", "no_buffer"),
            (False, True),
            (False, True),
        ):
            with self.subTest(
                strategy=strategy, finished=finished, mamba_first=mamba_first
            ):
                cache, allocator, req_pool = self._fixture(
                    lazy=strategy == "extra_buffer_lazy",
                    swa=True,
                    mamba_first=mamba_first,
                    no_buffer=strategy == "no_buffer",
                )
                first = self._request(cache, allocator, req_pool, length=128)
                self._run_chunk(cache, req_pool, first, 64)
                available = req_pool.mamba_allocator.available_size()
                other = self._request(cache, allocator, req_pool, length=96)
                other.origin_input_ids[32:] = array("q", [2] * 64)
                other.full_untruncated_fill_ids = array("q", other.origin_input_ids)
                # Real matching splits the old 64-token node at 32. SWA
                # can branch there, but no recurrent state exists at 32.
                match_prefix_for_req(cache, other)
                self.assertEqual(other.swa_branching_seqlen, 32)
                self.assertIsNone(other.mamba_branching_seqlen)
                self.assertEqual(len(other.prefix_indices), 0)
                before = req_pool.mamba_allocator.available_size()
                self._run_chunk(
                    cache, req_pool, other, 64, publish_checkpoint=not finished
                )
                if finished:
                    cache.cache_finished_req(other, kv_len_to_handle=64)
                # The newly saved state is for 64 tokens. It cannot be
                # attached to the SWA-limited 32-token key or leaked.
                match = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(other.origin_input_ids))
                )
                self._assert_cached_state_matches_prefix(
                    cache, req_pool, other.origin_input_ids, len(match.device_indices)
                )
                self.assertEqual(len(match.device_indices), 0)
                self.assertEqual(
                    req_pool.mamba_allocator.available_size(),
                    available if finished else before,
                )
                self._assert_cached_state_matches_prefix(
                    cache, req_pool, first.origin_input_ids, 64
                )


if __name__ == "__main__":
    unittest.main()
