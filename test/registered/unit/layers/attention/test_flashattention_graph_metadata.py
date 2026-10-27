import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpace, IdSpaceKind
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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


if __name__ == "__main__":
    unittest.main()
