"""Inkling checkpoint writes use live physical slots, including graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BACKEND = "sglang.srt.layers.attention.linear.inkling_sconv_backend"


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestInklingCheckpointIndices(CustomTestCase):
    def make_backend(
        self, *, lazy=False, static=False, slot_count=6, draft_token_num=None
    ):
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
        slots = allocator.alloc(slot_count)
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
                return_value=SimpleNamespace(
                    speculative_num_draft_tokens=draft_token_num
                ),
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

    def test_distinct_strided_destinations_keep_static_storage(self):
        backend, allocator, pool, slots = self.make_backend()
        allocator.free(slots[:1].clone())
        selected = slots[2:].flip(0)
        storage = torch.stack(
            [selected, slots[1:2].expand_as(selected)], dim=1
        ).flatten()
        original = storage.clone()
        ids = storage[::2]
        self.assertFalse(ids.is_contiguous())
        batch = self.batch(ids)
        backend._prepare_slot_indices(batch)
        pointer = batch.mamba_track_indices.data_ptr()
        physical = backend._translate_mamba_indices(selected).to(torch.int64)
        torch.testing.assert_close(batch.mamba_track_indices, physical, rtol=0, atol=0)
        cache = pool.mamba_cache.conv[0][0]
        expected = cache.clone()
        hidden = torch.arange(12 * 64, device="cuda").reshape(12, 64).to(torch.bfloat16)
        rows = torch.arange(12, device="cuda").reshape(4, 3)
        ShortConvolution._prepare_extend_sconv_cache(None, batch, cache, hidden, rows)
        expected[physical] = hidden.reshape(4, 3, 64)
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)
        smaller = self.batch(ids[:2])
        backend._prepare_slot_indices(smaller)
        self.assertEqual(smaller.mamba_track_indices.data_ptr(), pointer)
        torch.testing.assert_close(storage, original, rtol=0, atol=0)

    def test_decode_graph_refreshes_distinct_destinations_and_masks_padding(self):
        backend, allocator, pool, slots = self.make_backend(lazy=True, slot_count=12)
        req_pool = backend.req_to_token_pool
        req_pool.req_index_to_mamba_index_mapping = torch.cat(
            [slots.new_zeros(1), slots[1:5]]
        ).to(torch.int32)
        ids = slots[-4:].clone()
        track_mask = torch.ones(4, dtype=torch.bool, device="cuda")
        batch = self.batch(ids, mode=ForwardMode.DECODE, mask=track_mask)
        batch.req_pool_indices = torch.arange(1, 5, device="cuda")
        backend.init_forward_metadata_out_graph(batch)
        pointer = batch.mamba_track_indices.data_ptr()
        cache = pool.mamba_cache.conv[0][0]
        hidden = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
        weight = torch.ones(64, 4, dtype=torch.bfloat16, device="cuda")
        conv = SimpleNamespace(
            activation=None, use_residual=True, _weight_2d=lambda: weight
        )

        def decode():
            backend.init_forward_metadata_in_graph(batch)
            ShortConvolution._apply_decode_sconv_kernel(
                conv,
                hidden,
                cache,
                backend._cache_indices,
                backend.sconv_metadata.precomputed,
                batch,
            )

        decode()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode()
        for step, live in enumerate((3, 1, 4)):
            with self.subTest(live=live):
                if step < 2:
                    hole = slots[:1] if step == 0 else slots[5:6]
                    allocator.free(hole.clone())
                    allocator._flush(urgent=True)
                ids.copy_(slots[-4:].roll(step + 1))
                ids[live:].zero_()
                original = ids.clone()
                track_mask.zero_()
                track_mask[:live] = True
                if live > 1:
                    track_mask[1] = False
                fresh = self.batch(ids, mode=ForwardMode.DECODE, mask=track_mask)
                fresh.req_pool_indices = torch.arange(1, 5, device="cuda")
                fresh.req_pool_indices[live:] = 0
                backend.init_forward_metadata_out_graph(fresh)
                self.assertEqual(fresh.mamba_track_indices.data_ptr(), pointer)
                physical = backend._translate_mamba_indices(ids).to(torch.int64)
                active = backend._cache_indices.clone().to(torch.int64)
                cache.fill_(-7)
                cache[0].zero_()
                hidden.copy_(torch.arange(4, device="cuda")[:, None] + step + 1)
                hidden[live:].zero_()
                expected = cache.clone()
                for row in range(live):
                    window = torch.cat([cache[active[row], 1:], hidden[row : row + 1]])
                    expected[active[row]] = window
                    if row != 1:
                        expected[physical[row]] = window
                graph.replay()
                torch.testing.assert_close(cache, expected, rtol=0, atol=0)
                torch.testing.assert_close(ids, original, rtol=0, atol=0)

    def test_oversized_checkpoint_batch_fails_without_reallocating(self):
        backend, _, _, slots = self.make_backend()
        pointer = backend._graph_track_indices.data_ptr()
        batch = self.batch(slots[-1:].repeat(5))
        with self.assertRaisesRegex(
            AssertionError, "checkpoint-index buffer too small"
        ):
            backend._prepare_slot_indices(batch)
        self.assertEqual(backend._graph_track_indices.data_ptr(), pointer)

    def test_verify_commit_translates_both_destinations(self):
        backend, allocator, _, slots = self.make_backend()
        allocator.free(slots[:1].clone())
        indices = torch.tensor([0], device="cuda")
        for tracking in (False, True):
            with self.subTest(tracking=tracking):
                ids = slots[-1:].clone() if tracking else None
                with (
                    patch.object(
                        backend.req_to_token_pool,
                        "get_speculative_mamba2_params_all_layers",
                        return_value=object(),
                    ),
                    patch(
                        _BACKEND + ".scatter_mamba_states_after_mtp_verify"
                    ) as scatter,
                ):
                    backend.commit_conv_state_after_mtp_verify(
                        req_pool_indices=indices,
                        last_correct_step_indices=indices,
                        mamba_track_indices=ids,
                        mamba_steps_to_track=indices if tracking else None,
                    )
                if tracking:
                    torch.testing.assert_close(
                        scatter.call_args.args[3], backend._translate_mamba_indices(ids)
                    )
                    torch.testing.assert_close(ids, slots[-1:], rtol=0, atol=0)
                else:
                    self.assertIsNone(scatter.call_args.args[3])
                    self.assertIsNone(scatter.call_args.args[4])
                torch.testing.assert_close(
                    scatter.call_args.args[1],
                    backend._translate_mamba_indices(slots[1:2].to(torch.int32)),
                )

    def test_missing_tracking_field_still_refreshes_active_slots(self):
        backend, allocator, _, slots = self.make_backend()
        allocator.free(slots[:1].clone())
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            req_pool_indices=torch.tensor([0], device="cuda"),
        )
        backend.init_forward_metadata_out_graph(batch)
        torch.testing.assert_close(
            backend._cache_indices,
            backend._translate_mamba_indices(slots[1:2]).to(torch.int32),
        )

    def test_draft_runners_capture_and_replay_checkpoint_destinations(self):
        for multilayer in (False, True):
            for static in (False, True):
                for tracking in (False, True):
                    with self.subTest(
                        multilayer=multilayer, static=static, tracking=tracking
                    ):
                        self._check_draft_checkpoint_replay(
                            multilayer, static, tracking
                        )

    def _make_draft_checkpoint_runner(self, multilayer, static, tracking):
        from sglang.srt.speculative import eagle_draft_extend_cuda_graph_runner as eagle
        from sglang.srt.speculative import (
            multi_layer_eagle_draft_extend_cuda_graph_runner as multi,
        )

        backend, allocator, pool, slots = self.make_backend(
            lazy=True, static=static, slot_count=12, draft_token_num=4
        )
        backend.mamba_cache_chunk_size = 4
        req_pool = backend.req_to_token_pool
        req_pool.req_index_to_mamba_index_mapping = torch.cat(
            [slots.new_zeros(1), slots[1:5]]
        ).to(torch.int32)
        buffers = SimpleNamespace(
            input_ids=torch.zeros(16, dtype=torch.int64, device="cuda"),
            req_pool_indices=torch.arange(1, 5, device="cuda"),
            mamba_track_indices=slots[-4:].clone() if tracking else None,
            out_cache_loc=torch.zeros(16, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(16, dtype=torch.int64, device="cuda"),
            mrope_positions=torch.zeros(3, 16, dtype=torch.int64, device="cuda"),
            hidden_states=torch.zeros(16, 64, dtype=torch.bfloat16, device="cuda"),
            seq_lens=torch.full((4,), 6, dtype=torch.int64, device="cuda"),
            seq_lens_cpu=torch.full((4,), 6, dtype=torch.int64),
            extend_seq_lens=torch.full((4,), 4, dtype=torch.int32, device="cuda"),
            extend_start_loc=torch.arange(0, 16, 4, device="cuda"),
            num_correct_drafts=torch.ones(4, dtype=torch.int32, device="cuda"),
            num_accept_tokens=torch.full((4,), 3, dtype=torch.int32, device="cuda"),
            select_index=torch.arange(0, 16, 4, device="cuda"),
            next_token_logits_buffer=torch.zeros(16, 8, device="cuda"),
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_gpu=None,
            dsa_seed_topk_capture=None,
            temperatures=None,
            draft_probs=None,
        )
        cls = (
            multi.MultiLayerEagleDraftExtendCudaGraphRunner
            if multilayer
            else eagle.EAGLEDraftExtendCudaGraphRunner
        )
        runner = cls.__new__(cls)
        runner.buffers = buffers
        runner.captured_req_width = 4
        runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        runner.extend_seq_lens_cpu = [4] * 4
        runner.require_mlp_tp_gather = False
        runner.require_attn_tp_gather = False
        runner.require_gathered_buffer = False
        runner.capture_bs = [4]
        runner.seq_len_fill_value = 4
        runner.num_front_tokens = 0
        runner.prune_draft_extend_logits = False
        runner.metadata_captured_in_graph = False
        runner.step = 0
        runner.deepep_adapter = Mock()
        runner.device_module = torch.cuda
        runner.attn_backend = SimpleNamespace(
            supports_draft_extend_metadata_staging=False
        )
        runner.draft_extend_attn_backend = backend
        runner.eagle_worker = SimpleNamespace(draft_extend_attn_backend_list=[backend])
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
            device_timer=None,
            canary_manager=None,
        )
        return runner, backend, allocator, pool, slots, buffers

    def _check_draft_checkpoint_replay(self, multilayer, static, tracking):
        from sglang.srt.speculative import eagle_draft_extend_cuda_graph_runner as eagle
        from sglang.srt.speculative import (
            multi_layer_eagle_draft_extend_cuda_graph_runner as multi,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        runner, backend, allocator, pool, slots, buffers = (
            self._make_draft_checkpoint_runner(multilayer, static, tracking)
        )
        cache = pool.mamba_cache.conv[0][0]
        captured = []
        graph = torch.cuda.CUDAGraph()
        output = SimpleNamespace(
            next_token_logits=buffers.next_token_logits_buffer,
            hidden_states=buffers.hidden_states,
        )

        def model_forward(input_ids, positions, batch):
            if not captured:
                captured.append(batch)
            ShortConvolution._update_sconv_cache_for_draft_extend(
                None, batch, cache, backend._cache_indices, buffers.hidden_states
            )
            return output

        def capture_graph(key, run_once, **kwargs):
            run_once()
            with torch.cuda.graph(graph):
                run_once()

        runner.model_runner.model = SimpleNamespace(forward=model_forward)
        runner.backend = SimpleNamespace(capture_one=capture_graph)
        runner._replay_graph = lambda *args: (graph.replay(), output)[1]
        sconv_module = "sglang.srt.models.inkling_common.sconv"
        with (
            patch(
                sconv_module + ".get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(
                        mamba_track_interval=4, enable_mamba_extra_buffer=True
                    )
                ),
            ),
            patch.object(eagle, "maybe_flashinfer_autotune_speculative_draft"),
            patch.object(eagle, "set_dp_buffer_len"),
            patch.object(eagle, "set_is_extend_in_batch"),
        ):
            if multilayer:
                batch = runner.get_forward_batch(4)
                backend.init_forward_metadata_out_graph(batch, in_capture=True)

                def run_once():
                    backend.init_forward_metadata_in_graph(batch)
                    return model_forward(batch.input_ids, batch.positions, batch)

                capture_graph(None, run_once)
            else:
                runner.capture_one_shape(4, None)
            pointer = captured[0].mamba_track_indices.data_ptr() if tracking else None
            self.assertIsNone(backend.sconv_metadata.track_conv_indices)
            for live in (4, 2, 1, 3, 4):
                if live == 2 and not static:
                    allocator.free(slots[:1].clone())
                    allocator._flush(urgent=True)
                ids = slots[-live:].flip(0).clone()
                spec = EagleDraftExtendInput(
                    hidden_states=buffers.hidden_states[: live * 4],
                    num_correct_drafts=torch.ones(
                        live, dtype=torch.int32, device="cuda"
                    ),
                    num_accept_tokens=torch.full(
                        (live,), 3, dtype=torch.int32, device="cuda"
                    ),
                )
                fresh = SimpleNamespace(
                    batch_size=live,
                    input_ids=buffers.input_ids[: live * 4],
                    positions=buffers.positions[: live * 4],
                    out_cache_loc=buffers.out_cache_loc[: live * 4],
                    req_pool_indices=torch.arange(1, live + 1, device="cuda"),
                    seq_lens=torch.full((live,), 6, dtype=torch.int64, device="cuda"),
                    seq_lens_sum=6 * live,
                    seq_lens_cpu=None,
                    extend_seq_lens=torch.full(
                        (live,), 4, dtype=torch.int32, device="cuda"
                    ),
                    extend_seq_lens_cpu=None,
                    mamba_track_indices=ids if tracking and live != 3 else None,
                    spec_info=spec,
                )
                cache.fill_(-7)
                cache[0].zero_()
                buffers.hidden_states.copy_(
                    torch.arange(16, device="cuda")[:, None] + 1
                )
                original = ids.clone()
                active = backend._translate_mamba_indices(slots[1 : live + 1]).long()
                physical = backend._translate_mamba_indices(ids).long()
                expected = cache.clone()
                for row in range(live):
                    joined = torch.cat(
                        [
                            cache[active[row]],
                            buffers.hidden_states[row * 4 : (row + 1) * 4],
                        ]
                    )
                    if tracking and live != 3:
                        expected[physical[row]] = joined[1:4]
                    expected[active[row]] = joined[3:6]
                if multilayer:
                    composite = multi.MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.__new__(
                        multi.MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
                    )
                    composite.buffers = buffers
                    composite.captured_req_width = 4
                    composite.require_mlp_tp_gather = False
                    composite.require_gathered_buffer = False
                    composite.capture_bs = [4]
                    composite.num_front_tokens = 0
                    composite.seq_len_fill_value = 4
                    composite.runners = [runner]
                    composite._stage_metadata = Mock()
                    composite.prepare(fresh)
                    runner.replay(
                        4, composite.seq_lens_sum, composite._replay_spec_info, None
                    )
                else:
                    runner.execute(fresh, torch.arange(live, device="cuda") * 4 + 1)
                torch.testing.assert_close(cache[1:], expected[1:], rtol=0, atol=0)
                torch.testing.assert_close(ids, original, rtol=0, atol=0)
                if tracking:
                    expected_ids = torch.zeros_like(buffers.mamba_track_indices)
                    if live != 3:
                        expected_ids[:live] = original
                    torch.testing.assert_close(
                        buffers.mamba_track_indices, expected_ids, rtol=0, atol=0
                    )
                    self.assertEqual(
                        captured[0].mamba_track_indices.data_ptr(), pointer
                    )
                else:
                    self.assertIsNone(captured[0].mamba_track_indices)


if __name__ == "__main__":
    unittest.main()
