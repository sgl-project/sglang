import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpace, IdSpaceKind, KVLocPlan
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


class TestFlashAttentionGraphMetadata(CustomTestCase):
    def test_full_prefill_swa_buffer_uses_capture_token_capacity(self):
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.full_cg_prefill_metadata = None
        backend.use_sliding_window_kv_pool = True
        backend.max_num_pages = 4
        backend.max_context_len = 4
        backend.page_size = 1
        backend.req_to_token = torch.zeros((1, 4), dtype=torch.int32)
        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda locations: locations
        )
        # The metadata builder reads `is_translating` to choose between the
        # translated block table and the strided one this test covers, so the
        # source has to be real; the stub pools disable translation, which is
        # the static-pool view the assertions below are written against.
        backend.kv_index_translator = KVIndexTranslator(
            req_to_token=backend.req_to_token,
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=backend.page_size,
            device="cpu",
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.zeros(1, dtype=torch.int64),
            extend_seq_lens=torch.zeros(1, dtype=torch.int64),
            seq_lens_cpu=torch.zeros(1, dtype=torch.int64),
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            out_cache_loc=torch.arange(8, dtype=torch.int64),
            positions=torch.arange(8, dtype=torch.int64),
        )

        backend._init_full_cg_prefill_metadata(forward_batch, in_capture=True)

        self.assertEqual(backend.full_cg_prefill_swa_out_cache_loc.shape, (8,))
        forward_batch.out_cache_loc = torch.arange(9, dtype=torch.int64)
        with self.assertRaisesRegex(AssertionError, "used 9 > capacity 8"):
            backend._init_full_cg_prefill_metadata(forward_batch, in_capture=False)

    def test_empty_cpu_sequence_lengths_use_static_bound(self):
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.max_context_len = 16
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 16), dtype=torch.int32)
        )
        # A real source over the stub pool: the probe disables it, giving the
        # backend the strict passthrough view it now reads its tables from.
        backend.kv_index_translator = KVIndexTranslator(
            req_to_token=backend.req_to_token_pool.req_to_token,
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device="cpu",
        )
        backend.is_prefill_aware_swa = False
        backend.has_swa = False
        backend.use_sliding_window_kv_pool = False
        backend.page_size = 1
        backend._compute_scheduler_metadata = lambda *_: None
        backend._maybe_init_local_attn_metadata = lambda *_: None
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.empty(0, dtype=torch.int64),
            batch_size=0,
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            spec_info=None,
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            encoder_lens=None,
        )

        backend.init_forward_metadata(forward_batch)

        self.assertEqual(backend.forward_metadata.max_seq_len_k, 16)


class TestSpecReadSeqLenDelta(CustomTestCase):
    """How far past seq_lens each spec mode's translated read table reaches;
    too short silently leaves the verify or draft tail stale."""

    def _backend(self, *, topk=1, num_steps=3, step_id=1, num_draft=4):
        b = FlashAttentionBackend.__new__(FlashAttentionBackend)
        b.topk = topk
        b.speculative_num_steps = num_steps
        b.speculative_step_id = step_id
        b.speculative_num_draft_tokens = num_draft
        return b

    def test_mode_to_delta_map(self):
        b = self._backend()
        spec = object()
        # Verify reads [prefix + drafts]; draft decode step i reads
        # [prefix + i + 1] (both write the drafts into the pool first).
        self.assertEqual(b._spec_read_seq_len_delta(ForwardMode.TARGET_VERIFY, spec), 4)
        self.assertEqual(b._spec_read_seq_len_delta(ForwardMode.DECODE, spec), 2)
        # Prefix-only shapes stay un-widened.
        self.assertEqual(
            b._spec_read_seq_len_delta(ForwardMode.DRAFT_EXTEND_V2, spec), 0
        )
        self.assertEqual(b._spec_read_seq_len_delta(ForwardMode.DECODE, None), 0)
        self.assertEqual(b._spec_read_seq_len_delta(ForwardMode.EXTEND, spec), 0)
        # Draft-extend's idle batch carries no live draft chain.
        idle = self._backend(num_steps=0)
        self.assertEqual(idle._spec_read_seq_len_delta(ForwardMode.IDLE, spec), 0)
        # topk>1 reads the drafts via the expand metadata - prefix only.
        tree = self._backend(topk=2)
        self.assertEqual(
            tree._spec_read_seq_len_delta(ForwardMode.TARGET_VERIFY, spec), 0
        )
        self.assertEqual(tree._spec_read_seq_len_delta(ForwardMode.DECODE, spec), 0)


class TestDraftExtendInGraph(CustomTestCase):
    """A translating fa3 must not record the in-graph draft-extend gather: it
    reads raw (virtual) req_to_token ids over the tables rebuilt out of graph."""

    def _run(self, *, translating):
        b = FlashAttentionBackend.__new__(FlashAttentionBackend)
        b.kv_index_translator = SimpleNamespace(is_translating=translating)
        b.draft_extend_metadata = {}  # any metadata access raises KeyError
        fb = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2, batch_size=2)
        b.init_forward_metadata_in_graph(fb)

    def test_only_a_static_backend_builds_in_graph(self):
        self._run(translating=True)
        with self.assertRaises(KeyError):
            self._run(translating=False)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph replay requires CUDA")
class TestUnifiedVerifyGraphMetadata(CustomTestCase):
    def test_replay_uses_live_mapping_and_request_rows(self):
        """Captured tables must follow compaction, page reuse and shorter rows.

        A raw req_to_token gather silently uses virtual pages; an eager-only
        builder leaves stale physical pages after replay. Both violate the
        expected physical addresses checked after each mapping change.
        """
        for page_size in (1, 128):
            for use_swa in (False, True):
                with self.subTest(page_size=page_size, use_swa=use_swa):
                    self._check_replay(page_size, use_swa)

    def _check_replay(self, ps, use_swa):
        device = "cuda"
        bs, pages, delta = 3, 6, 3
        b = FlashAttentionBackend.__new__(FlashAttentionBackend)
        b.fa_impl_ver = 4
        b.topk = 1
        b.page_size = ps
        b.max_num_pages = pages
        b.speculative_num_draft_tokens = delta
        b.use_sliding_window_kv_pool = use_swa
        # Request zero is the padded sink. Live virtual pages differ from
        # physical pages, including an unbound SWA page (-1).
        b.req_to_token = torch.zeros((4, pages * ps), dtype=torch.int64, device=device)
        for row in range(1, 4):
            b.req_to_token[row] = torch.arange(
                row * pages * ps, (row + 1) * pages * ps, device=device
            )
        full = torch.arange(24, device=device, dtype=torch.int32).flip(0).contiguous()
        full[0] = 0
        swa = torch.arange(24, device=device, dtype=torch.int32)
        swa[7] = -1
        translator = KVIndexTranslator.__new__(KVIndexTranslator)
        translator._spaces = {
            IdSpaceKind.FULL: IdSpace(key="full", read_v2p=full),
        }
        if use_swa:
            translator._spaces[IdSpaceKind.SLIDING_WINDOW] = IdSpace(
                key="swa", read_v2p=swa
            )
        b.kv_index_translator = translator
        b.target_verify_metadata = {
            "cache_seqlens": torch.zeros(bs, dtype=torch.int32, device=device),
            "cu_seqlens_k": torch.zeros(bs + 1, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.arange(
                0, (bs + 1) * delta, delta, dtype=torch.int32, device=device
            ),
            "page_table": torch.full(
                (bs, pages), -99, dtype=torch.int32, device=device
            ),
        }
        if use_swa:
            b.target_verify_metadata["swa_page_table"] = torch.full(
                (bs, pages), -99, dtype=torch.int32, device=device
            )
            b.cuda_graph_swa_out_cache_loc = torch.zeros(
                bs * delta, dtype=torch.int64, device=device
            )
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            req_pool_indices=torch.tensor([2, 1, 0], device=device),
            seq_lens=torch.tensor([ps + 1, 2 * ps - 1, 1], device=device),
            encoder_lens=None,
            spec_info=SimpleNamespace(ragged_verify_layout=None),
            positions=torch.arange(bs * delta, device=device),
            out_cache_loc=None,
        )
        b.init_forward_metadata_out_graph(fb, in_capture=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            b.init_forward_metadata_in_graph(fb)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            b.init_forward_metadata_in_graph(fb)
        metadata = b.target_verify_metadata[bs]
        pointer = metadata.page_table.data_ptr()
        for iteration in range(3):
            if iteration == 1:
                # Compaction and request reorder, plus a shorter live prefix.
                full[6:24] = torch.arange(1, 19, device=device, dtype=torch.int32)
                swa[7] = 21
                swa[13] = -1
                fb.req_pool_indices.copy_(torch.tensor([1, 3, 0], device=device))
                fb.seq_lens.copy_(torch.tensor([0, 3 * ps, 1], device=device))
            elif iteration == 2:
                # Recycle a request row, grow its extent and reuse its pages.
                b.req_to_token[1].copy_(b.req_to_token[2])
                full[12:18] = 4
                fb.seq_lens.copy_(torch.tensor([2 * ps, 0, 1], device=device))
            b.init_forward_metadata_out_graph(fb)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(metadata.page_table.data_ptr(), pointer)
            lengths = fb.seq_lens.cpu() + delta
            torch.testing.assert_close(
                metadata.cache_seqlens_int32.cpu(), lengths.int()
            )
            torch.testing.assert_close(
                metadata.cu_seqlens_k.cpu(),
                torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int())),
            )
            rows = fb.req_pool_indices.cpu().tolist()
            for table, mapping in (
                (metadata.page_table, full),
                (metadata.swa_page_table, swa),
            ):
                if table is None:
                    continue
                expected = torch.zeros((bs, pages), dtype=torch.int32)
                tokens = b.req_to_token.cpu()
                physical = mapping.cpu()
                for lane, row in enumerate(rows):
                    for page in range(min(pages, (int(lengths[lane]) + ps - 1) // ps)):
                        virtual = int(tokens[row, page * ps]) // ps
                        expected[lane, page] = max(0, int(physical[virtual]))
                torch.testing.assert_close(table.cpu(), expected)


class TestDraftExtendStagingCapability(CustomTestCase):
    """Which backends stage their draft-extend tables once per draft cycle: a
    static backend whose graph records the tables keeps doing so, a
    translating one stages them from the cycle's plan, and DCP, whose reads
    stay virtual, keeps the per-step rebuild."""

    def _backend(self, *, topk=1, translating, reads_translated=True, in_graph=True):
        b = FlashAttentionBackend.__new__(FlashAttentionBackend)
        b.topk = topk
        b.kv_index_translator = SimpleNamespace(
            is_translating=translating, reads_are_translated=reads_translated
        )
        b.draft_extend_metadata_captured_in_graph = lambda: in_graph
        return b

    def test_staging_capability(self):
        cases = (
            (dict(translating=False), True),
            (dict(translating=False, in_graph=False), False),
            (dict(translating=True, in_graph=False), True),
            (dict(translating=True, in_graph=False, reads_translated=False), False),
            (dict(translating=True, in_graph=False, topk=2), False),
            (dict(translating=False, topk=2), False),
        )
        for kwargs, stages in cases:
            with self.subTest(**kwargs):
                backend = self._backend(**kwargs)
                self.assertEqual(backend.supports_draft_extend_metadata_staging, stages)


def _page_translator(req_to_token, page_size, full_v2p, swa_v2p):
    """A unified translator over page tables: a virtual token id maps to its
    page's physical page, and an unbound page (-1) to the sink."""

    def through(v2p):
        def write(ids):
            pages = v2p[ids // page_size].clamp(min=0).to(ids.dtype)
            return pages * page_size + ids % page_size

        return write

    translator = KVIndexTranslator.__new__(KVIndexTranslator)
    translator.req_to_token = req_to_token
    translator.page_size = page_size
    translator.device = req_to_token.device
    translator.is_translating = True
    translator.defer_read_translate = False
    translator._rows = torch.arange(req_to_token.shape[0], device=req_to_token.device)
    translator._spaces = {
        IdSpaceKind.FULL: IdSpace(
            key="full", write=through(full_v2p), read_v2p=full_v2p
        )
    }
    if swa_v2p is not None:
        translator._spaces[IdSpaceKind.SLIDING_WINDOW] = IdSpace(
            key="swa", write=through(swa_v2p), read_v2p=swa_v2p
        )
    return translator


@unittest.skipUnless(torch.cuda.is_available(), "the table kernels run on CUDA")
class TestTranslatedDraftExtendStaging(CustomTestCase):
    """A translating backend stages its draft-extend tables once per draft
    cycle, from the cycle's plan, instead of rebuilding them before every
    step's replay. The staged tables must be exactly what the per-step
    rebuild produced: lengths, prefix sums, physical page tables and the
    sliding-window write ids."""

    BS, RAW_BS, WIDTH, PAGES = 4, 3, 3, 6

    def _setup(self, ps, use_swa):
        device = "cuda"
        rows, pages = 5, self.PAGES
        req_to_token = torch.zeros((rows, pages * ps), dtype=torch.int64, device=device)
        for row in range(1, rows):
            req_to_token[row] = torch.arange(
                row * pages * ps, (row + 1) * pages * ps, device=device
            )
        num_virtual_pages = rows * pages
        full = torch.arange(num_virtual_pages, device=device, dtype=torch.int32)
        full = full.flip(0).contiguous()
        full[0] = 0
        swa = None
        if use_swa:
            swa = torch.arange(num_virtual_pages, device=device, dtype=torch.int32)
            swa[pages + 1] = -1  # an unbound sliding-window page reads the sink
        translator = _page_translator(req_to_token, ps, full, swa)
        # Request 0 is the padded lane's sink row.
        req_pool_indices = torch.tensor([2, 4, 1, 0], device=device)
        # Every live lane holds its whole window; lane 3 is the padded one.
        seq_lens = torch.tensor([ps + 3, 2 * ps + 3, 4, 1], device=device)
        extend = torch.full((self.BS,), self.WIDTH, dtype=torch.int32, device=device)
        window = torch.cat(
            [
                req_to_token[int(r), int(n) - self.WIDTH : int(n)]
                for r, n in zip(
                    req_pool_indices[: self.RAW_BS], seq_lens[: self.RAW_BS]
                )
            ]
        )
        return translator, req_pool_indices, seq_lens, extend, window, full, swa

    def _plan(self, translator, req_pool_indices, seq_lens, window):
        return KVLocPlan(
            source=translator,
            req_pool_indices=req_pool_indices[: self.RAW_BS],
            seq_lens=seq_lens[: self.RAW_BS],
            seq_lens_cpu=None,
            write_virtual=window,
        )

    def _backend(self, translator, ps, use_swa):
        device = "cuda"
        b = FlashAttentionBackend.__new__(FlashAttentionBackend)
        b.topk = 1
        b.page_size = ps
        b.max_context_len = self.PAGES * ps
        b.needs_cpu_seq_lens = False
        b.use_sliding_window_kv_pool = use_swa
        b.req_to_token = translator.req_to_token
        b.kv_index_translator = translator
        b.draft_extend_metadata = {
            "cache_seqlens": torch.full(
                (self.BS,), -9, dtype=torch.int32, device=device
            ),
            # The prefix sums' leading zero is never rewritten, as allocated.
            "cu_seqlens_q": torch.zeros(self.BS + 1, dtype=torch.int32, device=device),
            "cu_seqlens_k": torch.zeros(self.BS + 1, dtype=torch.int32, device=device),
            "page_table": torch.full(
                (self.BS, self.PAGES), -99, dtype=torch.int32, device=device
            ),
            "strided_indices": torch.arange(0, self.PAGES * ps, ps, device=device),
        }
        if use_swa:
            b.draft_extend_metadata["swa_page_table"] = torch.full(
                (self.BS, self.PAGES), -99, dtype=torch.int32, device=device
            )
            b.cuda_graph_swa_out_cache_loc = torch.full(
                (self.BS * self.WIDTH + 2,), -99, dtype=torch.int64, device=device
            )
        b._bind_metadata_buffers(
            self.BS,
            self.BS * self.WIDTH,
            None,
            ForwardMode.DRAFT_EXTEND_V2,
            SimpleNamespace(num_tokens_per_req=self.WIDTH),
            torch.device(device),
        )
        return b

    def _staging_batch(self, plan, req_pool_indices, seq_lens, extend, window):
        """What the runner's `_stage_metadata` hands a staging backend."""
        out_cache_loc = torch.zeros(
            self.BS * self.WIDTH, dtype=torch.int64, device="cuda"
        )
        out_cache_loc[: window.numel()] = window
        return SimpleNamespace(
            batch_size=self.BS,
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend,
            out_cache_loc=out_cache_loc,
            mamba_track_indices=None,
            out_cache_loc_virtual=window,
            kv_loc_plan=plan,
            kv_loc_cols=None,
        )

    def _expected(self, translator, req_pool_indices, seq_lens, window, full, swa, ps):
        lengths = seq_lens.cpu().int()
        expected = {
            "cache_seqlens": lengths,
            "cu_seqlens_k": torch.cat(
                (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int())
            ),
            "cu_seqlens_q": torch.arange(
                0, (self.BS + 1) * self.WIDTH, self.WIDTH, dtype=torch.int32
            ),
        }
        tokens = translator.req_to_token.cpu()
        for name, mapping in (("page_table", full), ("swa_page_table", swa)):
            if mapping is None:
                continue
            table = torch.zeros((self.BS, self.PAGES), dtype=torch.int32)
            for lane in range(self.RAW_BS):
                row = int(req_pool_indices[lane])
                live = (int(lengths[lane]) + ps - 1) // ps
                for page in range(min(self.PAGES, live)):
                    virtual = int(tokens[row, page * ps]) // ps
                    table[lane, page] = max(0, int(mapping.cpu()[virtual]))
            expected[name] = table
        if swa is not None:
            writes = torch.zeros(self.BS * self.WIDTH, dtype=torch.int64)
            writes[: window.numel()] = (
                translator.space(IdSpaceKind.SLIDING_WINDOW).write(window).cpu()
            )
            expected["swa_out_cache_loc"] = writes
        return expected

    def _tables(self, backend):
        metadata = backend.draft_extend_metadata[self.BS]
        tables = {
            "cache_seqlens": metadata.cache_seqlens_int32,
            "cu_seqlens_k": metadata.cu_seqlens_k,
            "cu_seqlens_q": metadata.cu_seqlens_q,
            "page_table": metadata.page_table,
        }
        if backend.use_sliding_window_kv_pool:
            tables["swa_page_table"] = metadata.swa_page_table
            tables["swa_out_cache_loc"] = metadata.swa_out_cache_loc
        return {name: t.cpu() for name, t in tables.items()}

    def _assert_tables(self, actual, expected):
        for name, want in expected.items():
            got = actual[name]
            if name.endswith("page_table"):
                # Columns past a lane's live pages are never read.
                torch.testing.assert_close(
                    got[: self.RAW_BS], want[: self.RAW_BS], rtol=0, atol=0, msg=name
                )
            else:
                torch.testing.assert_close(got, want, rtol=0, atol=0, msg=name)

    def test_staging_matches_the_per_step_rebuild(self):
        for ps in (1, 128):
            for use_swa in (False, True):
                with self.subTest(page_size=ps, use_swa=use_swa):
                    translator, rpi, lens, extend, window, full, swa = self._setup(
                        ps, use_swa
                    )
                    expected = self._expected(
                        translator, rpi, lens, window, full, swa, ps
                    )

                    staged = self._backend(translator, ps, use_swa)
                    self.assertTrue(staged.supports_draft_extend_metadata_staging)
                    self.assertFalse(staged.draft_extend_metadata_captured_in_graph())
                    plan = self._plan(translator, rpi, lens, window)
                    staged.stage_draft_extend_metadata(
                        self._staging_batch(plan, rpi, lens, extend, window)
                    )
                    torch.cuda.synchronize()
                    self._assert_tables(self._tables(staged), expected)

                    # The rebuild each step's replay used to run, on its own plan.
                    rebuilt = self._backend(translator, ps, use_swa)
                    replay_view = self._staging_batch(
                        self._plan(translator, rpi, lens, window),
                        rpi,
                        lens,
                        extend,
                        window,
                    )
                    replay_view.seq_lens_cpu = None
                    replay_view.encoder_lens = None
                    replay_view.spec_info = SimpleNamespace(
                        extend_seq_lens_tensor=extend,
                        extend_seq_lens_cpu=None,
                        num_tokens_per_req=self.WIDTH,
                    )
                    rebuilt._init_full_cg_decode_metadata(replay_view, in_capture=False)
                    torch.cuda.synchronize()
                    self._assert_tables(self._tables(rebuilt), expected)

    def test_one_plan_serves_every_step_backend(self):
        """A draft chain's step backends stage from one plan: the first builds
        its page tables, the others copy them, and all hold the same tables."""
        ps, use_swa = 128, True
        translator, rpi, lens, extend, window, full, swa = self._setup(ps, use_swa)
        expected = self._expected(translator, rpi, lens, window, full, swa, ps)
        builds = []
        build = translator.build_iteration_table

        def counted(*args, **kwargs):
            builds.append(kwargs.get("into") is not None)
            return build(*args, **kwargs)

        translator.build_iteration_table = counted
        plan = self._plan(translator, rpi, lens, window)
        batch = self._staging_batch(plan, rpi, lens, extend, window)
        backends = [self._backend(translator, ps, use_swa) for _ in range(3)]
        for backend in backends:
            backend.stage_draft_extend_metadata(batch)
        torch.cuda.synchronize()
        # One build per table kind (full, sliding window), into the first
        # backend's own buffers.
        self.assertEqual(builds, [True, True])
        for backend in backends:
            self._assert_tables(self._tables(backend), expected)

    def test_captured_steps_read_each_cycles_staging(self):
        """A captured step reads the staged buffers in place; its in-graph hook
        records nothing, so a replay sees whatever the cycle staged, including
        after compaction moves pages."""
        ps, use_swa = 128, True
        translator, rpi, lens, extend, window, full, swa = self._setup(ps, use_swa)
        backend = self._backend(translator, ps, use_swa)
        metadata = backend.draft_extend_metadata[self.BS]
        reads = {
            name: torch.zeros_like(t)
            for name, t in (
                ("page_table", metadata.page_table),
                ("swa_page_table", metadata.swa_page_table),
                ("cache_seqlens", metadata.cache_seqlens_int32),
                ("swa_out_cache_loc", metadata.swa_out_cache_loc),
            )
        }
        sources = {
            "page_table": metadata.page_table,
            "swa_page_table": metadata.swa_page_table,
            "cache_seqlens": metadata.cache_seqlens_int32,
            "swa_out_cache_loc": metadata.swa_out_cache_loc,
        }
        capture_view = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2, batch_size=self.BS
        )

        def step():
            backend.init_forward_metadata_in_graph(capture_view)
            for name, out in reads.items():
                out.copy_(sources[name])

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
        for cycle in range(3):
            if cycle == 1:
                # Compaction moves pages; the window's sliding-window page binds.
                full[: self.PAGES * 5] = torch.roll(full[: self.PAGES * 5], 3)
                swa[self.PAGES + 1] = 17
            elif cycle == 2:
                # A shorter batch: the padded lane grows, a live one shrinks.
                lens[1] = ps + 1
                window = torch.cat(
                    [
                        translator.req_to_token[int(r), int(n) - self.WIDTH : int(n)]
                        for r, n in zip(rpi[: self.RAW_BS], lens[: self.RAW_BS])
                    ]
                )
            plan = self._plan(translator, rpi, lens, window)
            backend.stage_draft_extend_metadata(
                self._staging_batch(plan, rpi, lens, extend, window)
            )
            graph.replay()
            torch.cuda.synchronize()
            expected = self._expected(translator, rpi, lens, window, full, swa, ps)
            got = {name: out.cpu() for name, out in reads.items()}
            for name in ("page_table", "swa_page_table"):
                torch.testing.assert_close(
                    got[name][: self.RAW_BS],
                    expected[name][: self.RAW_BS],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                got["cache_seqlens"], expected["cache_seqlens"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                got["swa_out_cache_loc"], expected["swa_out_cache_loc"], rtol=0, atol=0
            )


class _WindowReader:
    """A translator whose sliding-window write ids are the virtual ids times
    ten, answering `write_ids` as the production translator does."""

    write_ids = KVIndexTranslator.write_ids

    def __init__(self):
        self._spaces = {
            IdSpaceKind.FULL: IdSpace(key="full"),
            IdSpaceKind.SLIDING_WINDOW: IdSpace(key="swa", write=lambda ids: ids * 10),
        }

    def space(self, kind):
        return self._spaces.get(kind)


class TestDraftWindowWriteIds(CustomTestCase):
    """A fused draft's window layers write where `_get_swa_write_locations`
    says, derived from the whole draft window. A multi-step draft step must
    keep its own columns (the window is laid out step-minor per row), and a
    multi-layer draft's extend writes the whole window. The fused window-draft
    gate admits fa3 and fa4 on this."""

    def _write_ids(self, *, bs, topk, num_steps, window_cols, step, mode):
        reader = _WindowReader()
        plan = KVLocPlan(
            source=reader,
            req_pool_indices=torch.arange(bs),
            seq_lens=torch.ones(bs, dtype=torch.int64),
            seq_lens_cpu=None,
            write_virtual=torch.arange(1, bs * topk * window_cols + 1),
            read_extent=0,
        )
        # A chain draft (topk 1) inside a wider verify window writes its
        # leading columns; otherwise the draft owns the whole window.
        cols = None if window_cols == num_steps else slice(0, num_steps)
        batch = SimpleNamespace(
            batch_size=bs,
            kv_loc_plan=plan,
            kv_loc_cols=cols,
            out_cache_loc=plan.write_ids(reader, cols=cols),
            forward_mode=mode,
            spec_info=object(),
        )
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.kv_index_translator = reader
        backend.topk = topk
        backend.speculative_num_steps = num_steps
        backend.speculative_step_id = step
        window = plan.write_ids(reader, kind=IdSpaceKind.SLIDING_WINDOW, cols=cols)
        return backend._get_swa_write_locations(batch), window

    def test_each_draft_step_writes_its_own_columns(self):
        for bs, topk, num_steps, window_cols in (
            (3, 1, 3, 3),
            (2, 2, 3, 3),
            (3, 1, 3, 5),
        ):
            per_step = None
            for step in range(num_steps):
                ids, window = self._write_ids(
                    bs=bs,
                    topk=topk,
                    num_steps=num_steps,
                    window_cols=window_cols,
                    step=step,
                    mode=ForwardMode.DECODE,
                )
                per_step = per_step_draft_out_cache_loc(window, bs, topk, num_steps)
                torch.testing.assert_close(ids, per_step[step], rtol=0, atol=0)
            # Step 1's slots are not the window's leading ids.
            self.assertFalse(torch.equal(window[: bs * topk], per_step[1]))

    def test_a_multi_layer_draft_extend_writes_the_whole_window(self):
        ids, window = self._write_ids(
            bs=3,
            topk=1,
            num_steps=3,
            window_cols=3,
            step=1,
            mode=ForwardMode.DRAFT_EXTEND_V2,
        )
        torch.testing.assert_close(ids, window, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
