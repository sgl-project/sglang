"""Window drafts keep their attention mask when placement falls back to FULL.

The old multi-step path only built window indices for SWA pools, so accepted
FULL placements reached decode with no window metadata. Exercise placement,
binding and the real index reader through eager and captured metadata gathers.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend as ab
from sglang.srt.layers.attention import flashinfer_backend as fb
from sglang.srt.layers.attention import triton_backend as tb
from sglang.srt.layers.attention import trtllm_mha_backend as tm
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.routed_cache import RoutedKVCache
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_FULL,
    LAYER_WINDOW,
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    PlacementContext,
    place_fused_draft,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_draft_pool import bind_fused_draft
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import arena_of, plan_stand_in

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


class TestFusedDraftWindowMetadata(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.args = ServerArgs(model_path="dummy", page_size=1)
        publish(self.args, role="tokenizer")

    def _fixture(self, window, target_window, backend_kind="triton"):
        decision = place_fused_draft(
            profile=DraftKVProfile(
                num_layers=1,
                num_depths=1,
                kinds={
                    LAYER_WINDOW: DraftLayerSet(
                        layer_ids=(0,),
                        geometry=DraftKVGeometry(head_num=1, head_dim=8, v_head_dim=8),
                        window=window,
                    )
                },
            ),
            num_runners=1,
            ctx=PlacementContext(
                host_names=("full", "swa") if target_window else ("full", "mamba"),
                target_window=target_window,
            ),
            store_dtype=torch.bfloat16,
        )
        self.assertIsNotNone(decision.placement, decision.declined)
        placement = decision.placement
        host = "swa" if target_window and window <= target_window else "full"
        self.assertEqual(placement.hosts(), (host,))
        specs = [
            MHASubPoolSpec(
                name=name,
                layer_num=1,
                head_num=1,
                head_dim=8,
                store_dtype=torch.bfloat16,
                grow_direction=direction,
                draft_region=placement.region(name),
            )
            for name, direction in (("full", "up"), ("swa", "down"))
        ]
        buffer = arena_of(
            total_bytes=1024 * sum(spec.entry_bytes() for spec in specs),
            sub_pool_specs=specs,
            device="cuda",
            enable_memory_saver=False,
            page_size=1,
            fused_draft=placement,
        )
        views = {
            name: MHAView(
                arena=buffer,
                spec=buffer.spec(name),
                page_size=1,
                model_layer_ids=(layer_id,),
            )
            for name, layer_id in (("full", 0), ("swa", 1))
        }
        target_pool = RoutedKVCache(
            arena=buffer,
            owner="full",
            layer_maps=[("full", views["full"], {0: 0}), ("swa", views["swa"], {1: 0})],
            page_size=1,
            start_layer=0,
            end_layer=None,
            size=buffer.num_entries("full") - 1,
            head_num=views["full"].head_num,
            head_dim=views["full"].head_dim,
        )
        allocator = ChainAllocator(
            roles=ChainRoles(owner="full", window="swa"),
            arena=buffer,
            kvcache=target_pool,
            views=views,
            device="cuda",
            full_max_total_num_tokens=1024,
            swa_max_total_num_tokens=1024,
        )
        # Different FULL/SWA addresses make a wrong storage route observable.
        old = allocator.alloc(7)
        allocator.free_swa_segment(old, start_pos=0)
        virtual = allocator.alloc(512)
        req = SimpleNamespace(size=1, req_to_token=virtual.reshape(1, -1))
        layer = RadixAttention(
            num_heads=1,
            head_dim=8,
            scaling=0.5,
            num_kv_heads=1,
            layer_id=0,
            sliding_window_size=window,
        )
        bound = bind_fused_draft(
            placement=placement,
            runner=0,
            arena=buffer,
            plan=plan_stand_in(buffer, placement),
            owner="full",
            target_state=None,
            model=SimpleNamespace(modules=lambda: [layer]),
            model_config=SimpleNamespace(),
            req_to_token_pool=req,
            page_size=1,
        )
        reader = RunnerIds(
            req_to_token=req.req_to_token,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=bound.token_to_kv_pool,
            page_size=1,
            device="cuda",
            is_draft_worker=True,
        )
        runner = SimpleNamespace(
            req_to_token_pool=req,
            device="cuda",
            model_config=SimpleNamespace(get_max_num_attention_heads=lambda: 1),
            server_args=self.args,
            sliding_window_size=window,
            runner_ids=reader,
            token_to_kv_pool=bound.token_to_kv_pool,
            page_size=1,
        )
        # The model's per-step attention is outside this metadata test. All
        # placement, allocation, translation and multi-step metadata are real.
        if backend_kind == "triton":
            inner = SimpleNamespace(max_context_len=512)
            with patch.object(tb, "TritonAttnBackend", return_value=inner):
                backend = tb.TritonMultiStepDraftBackend(runner, 1, 3)
        else:
            # Whether the step backend writes a window sub-pool, as
            # FlashInferAttnBackend resolves it for this runner.
            inner = SimpleNamespace(
                max_context_len=512, use_sliding_window_kv_pool=host == "swa"
            )
            with patch.object(fb, "FlashInferAttnBackend", return_value=inner):
                if backend_kind == "flashinfer":
                    backend = fb.FlashInferMultiStepDraftBackend(runner, 1, 3)
                else:
                    # The step backends the trtllm_mha draft builds in place
                    # of its parent's, with the arguments they get.
                    def step(*args, **kwargs):
                        return SimpleNamespace(kwargs=kwargs, max_context_len=512)

                    with patch.object(tm, "TRTLLMHAAttnBackend", side_effect=step):
                        backend = tm.TRTLLMHAAttnMultiStepDraftBackend(runner, 1, 3)
        self.bound_pool = bound.token_to_kv_pool
        self.runner, self.layer = runner, layer
        self.allocator, self.virtual, self.reader = allocator, virtual, reader
        rows = torch.zeros(1, dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([300], dtype=torch.int32, device="cuda")
        plan = reader.plan(
            req_pool_indices=rows,
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            write_virtual=None,
            read_extent=3,
        )
        batch = SimpleNamespace(
            batch_size=1,
            seq_lens_sum=None if backend_kind == "triton" else 300,
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            req_pool_indices=rows,
            positions=seq_lens.to(torch.int64),
            iteration_ids=plan,
            spec_info=SimpleNamespace(
                window_kv_indices=None,
                window_kv_indptr=None,
                is_draft_input=lambda: True,
            ),
        )
        full = allocator.translate_kv_loc(virtual)
        swa = allocator.id_spaces()[IdSpaceKind.SLIDING_WINDOW].ids(virtual)
        self.assertFalse(torch.equal(full, swa))
        return backend, batch, full if host == "full" else swa, host

    def test_window_reads_follow_placement_in_eager_and_graph(self):
        for window, target_window in ((256, 128), (64, None), (64, 128)):
            with self.subTest(window=window, target_window=target_window):
                backend, batch, physical, _ = self._fixture(window, target_window)
                full_ids = torch.empty((3, 512), dtype=torch.int64, device="cuda")
                window_ids = torch.zeros((3, window), dtype=torch.int64, device="cuda")
                observed = []

                def collect(step, fb):
                    self.assertIsNotNone(fb.spec_info.window_kv_indices)
                    observed.append(
                        (
                            step,
                            fb.spec_info.window_kv_indptr.clone(),
                            fb.spec_info.window_kv_indices.clone(),
                        )
                    )

                def gather():
                    backend.common_template(batch, full_ids, collect, window_ids)

                def check(length):
                    for step, indptr, indices in observed:
                        end = length + step + 1
                        torch.testing.assert_close(
                            indptr,
                            torch.tensor([0, window], dtype=torch.int32, device="cuda"),
                        )
                        torch.testing.assert_close(
                            indices, physical[end - window : end]
                        )

                gather()
                check(300)
                observed.clear()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    gather()
                torch.cuda.current_stream().wait_stream(stream)
                batch.seq_lens.fill_(301)
                batch.positions.fill_(301)
                graph.replay()
                check(301)

    def _flashinfer_draft_extend_plans(self, window, seq_len, accept, *, graph):
        """How each FlashInfer prefill wrapper plans a window draft's draft
        extend, eagerly as a plain extend or as a graph replay plans it from
        the draft-extend input: the ids it reads, and its ``window_left`` (-1
        when planned without the per-query window mask)."""
        reader = self.reader
        rows = torch.zeros(1, dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        plan = reader.plan(
            req_pool_indices=rows,
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            write_virtual=None,
            read_extent=0,
        )
        updater = object.__new__(fb.FlashInferIndicesUpdaterPrefill)
        updater.num_qo_heads = updater.num_kv_heads = 1
        updater.head_dim = 8
        updater.data_type = updater.q_data_type = torch.bfloat16
        updater.sliding_window_size = window
        updater.attn_backend = SimpleNamespace(
            runner_ids=reader, dq_paged_kernel_lens=None
        )
        updater.kv_indptr = [
            torch.zeros(2, dtype=torch.int32, device="cuda") for _ in range(2)
        ]
        updater.qo_indptr = [
            torch.zeros(2, dtype=torch.int32, device="cuda") for _ in range(2)
        ]
        updater.kv_last_page_len = torch.ones(1, dtype=torch.int32, device="cuda")
        updater._swa_kv_pool = self.bound_pool
        updater._swa_space = reader.space(IdSpaceKind.SLIDING_WINDOW)
        updater.prefill_wrapper_ragged = None
        planned = []

        def begin_forward(qo_indptr, kv_indptr, kv_indices, *args, **kwargs):
            planned.append(
                SimpleNamespace(
                    ids=kv_indices[: int(kv_indptr[-1])].to(torch.int64).clone(),
                    window_left=kwargs.get("window_left", -1),
                )
            )

        wrappers = [
            SimpleNamespace(
                begin_forward=begin_forward,
                is_cuda_graph_enabled=graph,
                _custom_mask_buf=None,
            )
            for _ in range(2)
        ]
        accepted = torch.tensor([accept], dtype=torch.int32, device="cuda")
        updater.update_sliding_window(
            rows,
            seq_lens,
            None,
            seq_len,
            prefix_lens=None if graph else seq_lens - accept,
            prefill_wrappers=wrappers,
            use_ragged=False,
            encoder_lens=None,
            spec_info=(
                EagleDraftExtendInput(
                    num_correct_drafts=accepted - 1,
                    num_accept_tokens=accepted,
                    num_tokens_per_req=accept,
                )
                if graph
                else None
            ),
            plan=plan,
        )
        return planned

    def test_flashinfer_window_draft_extend_plans(self):
        """BUG REGRESSION. A draft extend replayed from a CUDA graph plans
        FlashInfer from the draft-extend input, whose gather dropped the
        window's start: the window wrapper read the sequence's first
        ``window + accepted`` tokens, here evicted window pages, where the
        eager extend reads its last ones. Both now read the window.

        BUG REGRESSION. The graph replay planned its window wrapper without
        the layer's window, so its kernel had no per-query window mask; both
        now plan with it."""
        window, seq_len, accept = 64, 300, 2
        self._fixture(window, target_window=128)
        # The window side no longer holds the prefix before the window.
        self.allocator.free_swa_segment(self.virtual[:200], start_pos=0)
        swa = self.allocator.id_spaces()[IdSpaceKind.SLIDING_WINDOW].ids(
            self.virtual[:seq_len]
        )
        span = window + accept
        live = swa[seq_len - span :]
        self.assertFalse(torch.equal(live, swa[:span]))
        self.assertFalse(
            torch.equal(
                live,
                self.allocator.translate_kv_loc(self.virtual[:seq_len])[-span:],
            )
        )
        eager = self._flashinfer_draft_extend_plans(
            window, seq_len, accept, graph=False
        )
        graph = self._flashinfer_draft_extend_plans(window, seq_len, accept, graph=True)
        torch.testing.assert_close(eager[0].ids, live.to(torch.int64))
        for eager_plan, graph_plan in zip(eager, graph):
            torch.testing.assert_close(graph_plan.ids, eager_plan.ids)
        # Window wrapper, then full wrapper.
        self.assertEqual([p.window_left for p in eager], [window, -1])
        self.assertEqual([p.window_left for p in graph], [window, -1])

    def test_flashinfer_window_reads_take_their_own_rail(self):
        """BUG REGRESSION. A flashinfer multi-step draft of a window model
        could not start (its step backends took one wrapper), and its window
        wrapper read the shared draft indices, FULL ids, with no translation
        when reads are translated. Each step now reads a rail of the last
        window + 1 tokens (the decode window wrapper's span) in the ids of
        the storage its window layers live in."""
        for window, target_window in ((256, 128), (64, None), (64, 128)):
            with self.subTest(window=window, target_window=target_window):
                backend, batch, physical, _ = self._fixture(
                    window, target_window, backend_kind="flashinfer"
                )
                self.assertTrue(backend.window_rail)
                full_ids = torch.empty((3, 512), dtype=torch.int32, device="cuda")
                window_ids = torch.zeros(
                    (3, window + 1), dtype=torch.int32, device="cuda"
                )
                observed = []

                def collect(step, fb_):
                    observed.append(
                        (
                            step,
                            fb_.spec_info.window_kv_indptr,
                            fb_.spec_info.window_kv_indices,
                            fb.global_override_window_indptr_cpu,
                        )
                    )

                for length in (300, 301):
                    batch.seq_lens.fill_(length)
                    batch.positions.fill_(length)
                    observed.clear()
                    backend.common_template(batch, full_ids, collect, window_ids)
                    self.assertEqual(len(observed), 2)
                    for step, indptr, indices, host_indptr in observed:
                        end = length + step + 1
                        span = window + 1
                        expected_indptr = torch.tensor(
                            [0, span], dtype=torch.int32, device="cuda"
                        )
                        torch.testing.assert_close(indptr, expected_indptr)
                        torch.testing.assert_close(
                            host_indptr, expected_indptr.cpu(), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            indices.to(torch.int64), physical[end - span : end]
                        )
                    self.assertIsNone(fb.global_override_window_indptr_cpu)

    def test_trtllm_steps_take_the_window_rail(self):
        """BUG REGRESSION. The trtllm_mha multi-step draft replaced its
        parent's step backends without their window indptr, so a window draft's
        step backend planned one wrapper for a two-wrapper model and failed at
        construction. Each step backend now gets its step's window rail."""
        for window, target_window in ((64, 128), (256, 128)):
            with self.subTest(window=window, target_window=target_window):
                backend, *_ = self._fixture(
                    window, target_window, backend_kind="trtllm"
                )
                self.assertTrue(backend.window_rail)
                self.assertEqual(len(backend.attn_backends), 2)
                for i, step in enumerate(backend.attn_backends):
                    # The step's row of the rail buffer, the storage it plans on.
                    rail = step.kwargs["window_kv_indptr_buf"]
                    self.assertEqual(
                        rail.data_ptr(), backend.window_kv_indptr[i].data_ptr()
                    )
                    self.assertEqual(rail.shape, backend.window_kv_indptr[i].shape)
                    self.assertEqual(
                        step.kwargs["kv_indptr_buf"].data_ptr(),
                        backend.kv_indptr[i].data_ptr(),
                    )

    def test_trtllm_window_draft_extend_reads_the_window_ids(self):
        """A trtllm_mha draft extend on translated storage reads a window layer
        through the plan's window page table, refreshed on every replay, as its
        target verify does, and its spec-decode kernel masks each query to the
        layer's window. Checked on the replay prep: the window table's live
        window holds the window sub-pool's ids of the request's tokens, not the
        owner's."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        window, seq_len = 64, 300
        self._fixture(window, 128)
        reader, virtual = self.reader, self.virtual
        rows = torch.zeros(1, dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        plan = reader.plan(
            req_pool_indices=rows,
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            write_virtual=None,
            read_extent=0,
        )
        backend = object.__new__(tm.TRTLLMHAAttnBackend)
        backend.runner_ids = reader
        backend.use_sliding_window_kv_pool = False
        metadata = SimpleNamespace(
            page_table=torch.full((1, 512), -1, dtype=torch.int32, device="cuda"),
            swa_page_table=torch.full((1, 512), -1, dtype=torch.int32, device="cuda"),
        )
        backend.draft_extend_metadata = {1: metadata}
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=None,
            iteration_ids=plan,
            out_cache_loc=None,
        )
        backend.init_forward_metadata_out_graph(batch, in_capture=False)
        self.assertIs(backend.forward_metadata, metadata)
        window_ids = reader.space(IdSpaceKind.SLIDING_WINDOW).ids(virtual)
        owner_ids = self.allocator.translate_kv_loc(virtual)
        live = slice(seq_len - window, seq_len)
        self.assertTrue(
            torch.equal(
                metadata.swa_page_table[0, live].to(torch.int64),
                window_ids[live].to(torch.int64),
            )
        )
        self.assertFalse(torch.equal(window_ids[live], owner_ids[live]))
        self.assertTrue(
            torch.equal(
                metadata.page_table[0, :seq_len].to(torch.int64),
                owner_ids[:seq_len].to(torch.int64),
            )
        )

    def test_a_trtllm_step_hands_its_window_rail_to_its_wrappers(self):
        """The trtllm_mha step backend passes the window indptr to the
        FlashInfer base, which plans the window wrapper on it."""
        seen = {}

        class Stop(Exception):
            pass

        def base_init(self, *args, **kwargs):
            seen.update(args=args, kwargs=kwargs)
            raise Stop

        rail = torch.zeros(5, dtype=torch.int32)
        with patch.object(fb.FlashInferAttnBackend, "__init__", base_init):
            with self.assertRaises(Stop):
                tm.TRTLLMHAAttnBackend(
                    SimpleNamespace(),
                    skip_prefill=True,
                    kv_indptr_buf=torch.zeros(5, dtype=torch.int32),
                    kv_last_page_len_buf=torch.ones(4, dtype=torch.int32),
                    speculative_step_id=1,
                    window_kv_indptr_buf=rail,
                )
        self.assertIs(seen["kwargs"]["window_kv_indptr_buf"], rail)

    def test_aiter_refuses_a_routed_window_view_at_init(self):
        """BUG REGRESSION. With the draft backend lists gone, an aiter draft
        fused into the window sub-pool failed at init asking its pool for a
        full view, and a mixed draft later read a static full -> window slot
        table its pool does not have. aiter now reads the layout of the views
        the pool routes to, and refuses at init a pool that takes physical
        write ids, which its window path cannot read."""
        self._fixture(64, 128)
        pool = self.bound_pool
        self.assertIsNone(pool.layer_router.sub_pool(IdSpaceKind.FULL))
        self.assertIsNotNone(pool.layer_router.sub_pool(IdSpaceKind.SLIDING_WINDOW))
        self.assertFalse(ab._kv_cache_is_vectorized_5d(pool))
        with self.assertRaisesRegex(ValueError, "physical write ids"):
            ab._require_untranslated_storage(pool, None)

    def test_aiter_refuses_a_full_only_fused_draft(self):
        """BUG REGRESSION. An aiter draft fused into the full sub-pool has no
        window pool, so a guard on the window pool alone let it through; its
        normal prefill then gathered the cached prefix's ids straight from the
        request table, which hold virtual ids, for KV stored at physical ids.
        With a moved prefix the two differ and the draft read other rows. aiter
        now refuses at init any pool that takes physical write ids."""
        placement = place_fused_draft(
            profile=DraftKVProfile(
                num_layers=1,
                num_depths=1,
                kinds={
                    LAYER_FULL: DraftLayerSet(
                        layer_ids=(0,),
                        geometry=DraftKVGeometry(head_num=1, head_dim=8, v_head_dim=8),
                    )
                },
            ),
            num_runners=1,
            ctx=PlacementContext(host_names=("full", "swa"), target_window=128),
            store_dtype=torch.bfloat16,
        ).placement
        self.assertEqual(placement.hosts(), ("full",))
        specs = [
            MHASubPoolSpec(
                name=name,
                layer_num=1,
                head_num=1,
                head_dim=8,
                store_dtype=torch.bfloat16,
                grow_direction=direction,
                draft_region=placement.region(name),
            )
            for name, direction in (("full", "up"), ("swa", "down"))
        ]
        buffer = arena_of(
            total_bytes=1024 * sum(spec.entry_bytes() for spec in specs),
            sub_pool_specs=specs,
            device="cuda",
            enable_memory_saver=False,
            page_size=1,
            fused_draft=placement,
        )
        views = {
            name: MHAView(
                arena=buffer,
                spec=buffer.spec(name),
                page_size=1,
                model_layer_ids=(layer_id,),
            )
            for name, layer_id in (("full", 0), ("swa", 1))
        }
        target_pool = RoutedKVCache(
            arena=buffer,
            owner="full",
            layer_maps=[("full", views["full"], {0: 0}), ("swa", views["swa"], {1: 0})],
            page_size=1,
            start_layer=0,
            end_layer=None,
            size=buffer.num_entries("full") - 1,
            head_num=views["full"].head_num,
            head_dim=views["full"].head_dim,
        )
        allocator = ChainAllocator(
            roles=ChainRoles(owner="full", window="swa"),
            arena=buffer,
            kvcache=target_pool,
            views=views,
            device="cuda",
            full_max_total_num_tokens=1024,
            swa_max_total_num_tokens=1024,
        )
        prefix = allocator.alloc(4)
        req = SimpleNamespace(size=1, req_to_token=prefix.reshape(1, -1))
        layer = RadixAttention(
            num_heads=1, head_dim=8, scaling=0.5, num_kv_heads=1, layer_id=0
        )
        bound = bind_fused_draft(
            placement=placement,
            runner=0,
            arena=buffer,
            plan=plan_stand_in(buffer, placement),
            owner="full",
            target_state=None,
            model=SimpleNamespace(modules=lambda: [layer]),
            model_config=SimpleNamespace(),
            req_to_token_pool=req,
            page_size=1,
        )
        # Move the cached prefix: rebind its virtual ids to other physical
        # units, as compaction does.
        ids = allocator.full_attn_allocator.ids
        physical = ids.translate(prefix)
        ids.bind(prefix, physical.flip(0))
        moved = ids.translate(prefix)
        self.assertFalse(torch.equal(moved, req.req_to_token[0]))

        draft_pool = bound.token_to_kv_pool
        self.assertIsNone(draft_pool.layer_router.sub_pool(IdSpaceKind.SLIDING_WINDOW))
        with self.assertRaisesRegex(ValueError, "physical write ids"):
            ab._require_untranslated_storage(draft_pool, None)

    def test_aiter_keeps_reading_a_static_window_pool(self):
        static = SWAKVPool(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cuda",
        )
        ab._require_untranslated_storage(static, None)
        ab._require_untranslated_storage(None, None)
        self.assertFalse(ab._kv_cache_is_vectorized_5d(static))


if __name__ == "__main__":
    unittest.main()
