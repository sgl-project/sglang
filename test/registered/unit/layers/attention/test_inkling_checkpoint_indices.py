"""Inkling checkpoint writes use live physical slots, including graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
    InklingShortConvAttnBackend,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedHybridReqToTokenPool,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMambaSlotAllocator,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.inkling_common.sconv import ShortConvolution
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_BACKEND = "sglang.srt.layers.attention.linear.inkling_sconv_backend"


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestInklingCheckpointIndices(CustomTestCase):
    def make_backend(self, *, lazy=False, static=False):
        mamba = MambaSubPoolSpec(
            name="mamba",
            layer_num=2,
            grow_direction="up",
            conv_state_shapes=((3, 64),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(0,),
            temporal_dtype=torch.float32,
        )
        full = MHASubPoolSpec(
            name="full",
            layer_num=1,
            head_num=1,
            head_dim=8,
            store_dtype=torch.bfloat16,
            grow_direction="down",
        )
        shared = UnifiedKVPool(
            total_bytes=64 * mamba.entry_bytes(),
            sub_pool_specs=[mamba, full],
            device="cuda",
            enable_memory_saver=False,
        )
        shared._raw.zero_()
        pool = UnifiedMambaPool(
            unified_buffer=shared,
            sub_pool_name="mamba",
            spec_state_size=0,
            mamba_layer_ids=[0, 1],
        )
        allocator = MultiEndedAllocator(
            kvcache=pool,
            unified_buffer=shared,
            sub_pool_name="mamba",
            device="cuda",
            is_id_owner=True,
            lazy_compaction=lazy,
        )
        slots = allocator.alloc(6)
        self.assertIsNotNone(slots)
        req_type = HybridReqToTokenPool if static else UnifiedHybridReqToTokenPool
        req_pool = req_type.__new__(req_type)
        req_pool.size = 4
        req_pool.mamba_pool = pool
        req_pool.mamba_allocator = UnifiedMambaSlotAllocator(
            allocator, max_size=pool.size, device="cuda"
        )
        req_pool.req_index_to_mamba_index_mapping = slots[1:2].to(torch.int32).repeat(4)
        backend = InklingShortConvAttnBackend.__new__(InklingShortConvAttnBackend)
        backend.req_to_token_pool = req_pool
        backend.device = "cuda"
        backend.conv_state_len = 3
        backend._cache_indices_buf = None
        backend._slot_gather_recordable = static
        config = SimpleNamespace(
            decode=SimpleNamespace(bs=[1, 2, 4], max_bs=4),
            prefill=SimpleNamespace(bs=[8]),
        )
        with (
            patch(
                _BACKEND + ".get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(cuda_graph_config=config)
                ),
            ),
            patch(
                _BACKEND + ".get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=None),
            ),
        ):
            backend._alloc_graph_buffers()
        return backend, allocator, pool, slots

    def batch(self, ids, mode=ForwardMode.EXTEND, mask=None):
        n = len(ids)
        return SimpleNamespace(
            forward_mode=mode,
            batch_size=n,
            req_pool_indices=torch.arange(n, device="cuda"),
            mamba_track_indices=ids,
            mamba_track_mask=torch.ones(n, dtype=torch.bool, device="cuda")
            if mask is None
            else mask,
        )

    def scatter(self, batch, cache):
        hidden = (
            torch.arange(3 * 64, device="cuda", dtype=torch.float32)
            .reshape(3, 64)
            .to(torch.bfloat16)
        )
        rows = torch.arange(3, device="cuda").repeat(batch.batch_size, 1)
        ShortConvolution._prepare_extend_sconv_cache(None, batch, cache, hidden, rows)
        return hidden

    def test_prefill_checkpoint_after_real_compaction(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                backend, allocator, pool, slots = self.make_backend(lazy=lazy)
                virtual = slots[-1:].clone()
                allocator.free(slots[:1].clone())
                if lazy:
                    allocator._flush(urgent=True)
                physical = backend._translate_mamba_indices(virtual)
                self.assertFalse(torch.equal(virtual, physical.to(virtual.dtype)))
                ids = torch.cat([virtual, virtual.new_tensor([-1])])
                original = ids.clone()
                batch = self.batch(ids, mask=torch.tensor([True, False], device="cuda"))
                backend._prepare_slot_indices(batch)
                cache = pool.mamba_cache.conv[0][0]
                before = cache.clone()
                expected = self.scatter(batch, cache)
                torch.testing.assert_close(cache[physical[0]], expected, rtol=0, atol=0)
                before[physical[0]] = expected
                torch.testing.assert_close(cache, before, rtol=0, atol=0)
                torch.testing.assert_close(ids, original, rtol=0, atol=0)
                self.assertEqual(batch.mamba_track_indices.dtype, torch.int64)

    def test_replay_refreshes_captured_destination_after_compaction(self):
        backend, allocator, pool, slots = self.make_backend(lazy=True)
        virtual = slots[-1:].clone()
        batch = self.batch(virtual, mode=ForwardMode.DECODE)
        backend.init_forward_metadata_out_graph(batch)
        pointer = batch.mamba_track_indices.data_ptr()
        cache = pool.mamba_cache.conv[0][0]
        self.scatter(batch, cache)  # compile before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.scatter(batch, cache)
        allocator.free(slots[:1].clone())
        allocator._flush(urgent=True)
        physical = backend._translate_mamba_indices(virtual)
        cache.zero_()
        fresh = self.batch(virtual, mode=ForwardMode.DECODE)
        backend.init_forward_metadata_out_graph(fresh)
        self.assertEqual(fresh.mamba_track_indices.data_ptr(), pointer)
        graph.replay()
        expected = (
            torch.arange(3 * 64, device="cuda", dtype=torch.float32)
            .reshape(3, 64)
            .to(torch.bfloat16)
        )
        torch.testing.assert_close(cache[physical[0]], expected, rtol=0, atol=0)
        self.assertEqual(
            torch.count_nonzero(cache).item(), torch.count_nonzero(expected).item()
        )
        torch.testing.assert_close(virtual, slots[-1:], rtol=0, atol=0)

    def test_decode_tracks_the_physical_checkpoint(self):
        backend, allocator, pool, slots = self.make_backend()
        allocator.free(slots[:1].clone())
        batch = self.batch(slots[-1:].clone(), mode=ForwardMode.DECODE)
        backend._prepare_slot_indices(batch)
        cache = pool.mamba_cache.conv[0][0]
        conv = SimpleNamespace(
            activation=None,
            use_residual=True,
            _weight_2d=lambda: torch.ones(64, 4, dtype=torch.bfloat16, device="cuda"),
        )
        ShortConvolution._apply_decode_sconv_kernel(
            conv,
            torch.ones(1, 64, dtype=torch.bfloat16, device="cuda"),
            cache,
            backend._cache_indices,
            {"cache_mask": torch.ones(1, 1, 1, dtype=torch.bool, device="cuda")},
            batch,
        )
        physical = backend._translate_mamba_indices(slots[-1:])
        torch.testing.assert_close(
            cache[physical[0]], cache[backend._cache_indices[0]], rtol=0, atol=0
        )
        self.assertGreater(torch.count_nonzero(cache[physical[0]]).item(), 0)

    def test_static_identity_path_and_disabled_tracking(self):
        backend, _, _, slots = self.make_backend(static=True)
        ids = slots[-1:].clone()
        batch = self.batch(ids)
        backend._prepare_slot_indices(batch)
        self.assertIs(batch.mamba_track_indices, ids)
        batch.mamba_track_indices = None
        backend._prepare_slot_indices(batch)
        self.assertIsNone(batch.mamba_track_indices)
        backend._slot_gather_recordable = False
        backend._prepare_slot_indices(batch)
        self.assertIsNone(batch.mamba_track_indices)

    def test_graph_hooks_translate_once_for_each_mode(self):
        backend, allocator, _, slots = self.make_backend()
        allocator.free(slots[:1].clone())
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode):
                ids = slots[-1:].clone()
                batch = self.batch(ids, mode=mode)
                with (
                    patch.object(backend, "_refresh_sconv_metadata"),
                    patch.object(
                        backend,
                        "_translate_mamba_indices",
                        wraps=backend._translate_mamba_indices,
                    ) as translate,
                ):
                    backend.init_forward_metadata_out_graph(batch)
                    calls = translate.call_count
                    self.assertEqual(calls, 2)  # active and checkpoint slots
                    backend.init_forward_metadata_in_graph(batch)
                    self.assertEqual(translate.call_count, calls)
                torch.testing.assert_close(
                    batch.mamba_track_indices,
                    backend._translate_mamba_indices(ids).to(torch.int64),
                )
                torch.testing.assert_close(ids, slots[-1:])

    def test_strided_batched_ids_and_padding_keep_static_storage(self):
        backend, allocator, _, slots = self.make_backend()
        allocator.free(slots[:1].clone())
        storage = slots[-1:].repeat(8)
        ids = storage[::2]
        self.assertFalse(ids.is_contiguous())
        batch = self.batch(ids)
        backend._prepare_slot_indices(batch)
        pointer = batch.mamba_track_indices.data_ptr()
        torch.testing.assert_close(
            batch.mamba_track_indices,
            backend._translate_mamba_indices(ids).to(torch.int64),
        )
        smaller = self.batch(ids[:2])
        backend._prepare_slot_indices(smaller)
        self.assertEqual(smaller.mamba_track_indices.data_ptr(), pointer)
        torch.testing.assert_close(storage, slots[-1:].repeat(8))

    def test_verify_commit_translates_both_destinations(self):
        backend, allocator, _, slots = self.make_backend()
        allocator.free(slots[:1].clone())
        ids = slots[-1:].clone()
        indices = torch.tensor([0], device="cuda")
        with (
            patch.object(
                backend.req_to_token_pool,
                "get_speculative_mamba2_params_all_layers",
                return_value=object(),
            ),
            patch(_BACKEND + ".scatter_mamba_states_after_mtp_verify") as scatter,
        ):
            backend.commit_conv_state_after_mtp_verify(
                req_pool_indices=indices,
                last_correct_step_indices=indices,
                mamba_track_indices=ids,
                mamba_steps_to_track=indices,
            )
        torch.testing.assert_close(
            scatter.call_args.args[3], backend._translate_mamba_indices(ids)
        )
        torch.testing.assert_close(
            scatter.call_args.args[1],
            backend._translate_mamba_indices(slots[1:2].to(torch.int32)),
        )


if __name__ == "__main__":
    unittest.main()
