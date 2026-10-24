import unittest
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from sglang.kernels.ops.attention.swa_window_align import align_swa_cache_seqlens
from sglang.srt.configs.step5 import Step5Config, uses_step5_swa_window_align
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


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
        backend.swa_fa4_window_align = False
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


class TestSwaWindowAlignGraphBuffers(CustomTestCase):
    """A captured decode graph bakes the addresses of the SWA window-align
    tensors, and replay prep refills them in place. An eager decode batch wider
    than the captured max used to replace those tensors, so every later replay
    read a stale window offset: sliding-window attention silently went wrong."""

    WINDOW = 127
    MAX_BS = 4

    def _backend(self):
        b = FlashAttentionBackend.__new__(FlashAttentionBackend)
        b.sliding_window_size = self.WINDOW
        b.topk = 0
        b.max_context_len = 16384
        b.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((4 * self.MAX_BS, 16384), dtype=torch.int32)
        )
        b.kv_index_translator = KVIndexTranslator(
            req_to_token=b.req_to_token_pool.req_to_token,
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device="cpu",
        )
        b.is_prefill_aware_swa = False
        b.use_sliding_window_kv_pool = False
        b.swa_fa4_window_align = True
        b.page_size = 1
        b._compute_scheduler_metadata = lambda *_: None
        b._maybe_init_local_attn_metadata = lambda *_: None
        # As init_cuda_graph_state allocates them for MAX_BS.
        b._swa_align_seqlens = torch.zeros(self.MAX_BS, dtype=torch.int32)
        b._swa_align_leftpad = torch.zeros(self.MAX_BS, dtype=torch.int32)
        return b

    def _expected(self, seqlens):
        return align_swa_cache_seqlens(seqlens, window_left=self.WINDOW, query_len=1)

    def test_replay_after_wider_eager_decode_reads_fresh_align(self):
        b = self._backend()
        graph_md = FlashAttentionMetadata()
        graph_md.cache_seqlens_int32 = torch.ones(self.MAX_BS, dtype=torch.int32)
        b._fill_swa_window_align(graph_md)
        captured = (graph_md.swa_aligned_seqlens, graph_md.swa_cache_leftpad)

        eager_bs = 3 * self.MAX_BS
        seq_lens = torch.arange(1, eager_bs + 1, dtype=torch.int64) * 500
        b.init_forward_metadata(
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                seq_lens=seq_lens,
                batch_size=eager_bs,
                seq_lens_cpu=seq_lens,
                spec_info=None,
                req_pool_indices=torch.arange(eager_bs, dtype=torch.int64),
                encoder_lens=None,
            )
        )
        eager_md = b.forward_metadata
        for got, want in zip(
            (eager_md.swa_aligned_seqlens, eager_md.swa_cache_leftpad),
            self._expected(seq_lens.to(torch.int32)),
        ):
            self.assertTrue(torch.equal(got, want))

        replay_seqlens = torch.tensor([100, 300, 1000, 5000], dtype=torch.int32)
        graph_md.cache_seqlens_int32.copy_(replay_seqlens)
        b._fill_swa_window_align(graph_md)
        for got, want in zip(captured, self._expected(replay_seqlens)):
            self.assertTrue(torch.equal(got, want), f"{got.tolist()=}")

    def test_graph_batch_wider_than_capture_raises(self):
        b = self._backend()
        graph_md = FlashAttentionMetadata()
        graph_md.cache_seqlens_int32 = torch.ones(self.MAX_BS + 1, dtype=torch.int32)
        with self.assertRaisesRegex(AssertionError, "used 5 > capacity 4"):
            b._fill_swa_window_align(graph_md)


class TestSwaWindowAlignScope(CustomTestCase):
    """FA4 leftpad window alignment is Step5's decode layout. Other SWA models must
    keep the stock KV split: FA4's dedicated head_dim-256 kernel rejects leftpad."""

    def test_only_step5_aligns_its_swa_decode(self):
        other = PretrainedConfig(sliding_window=127, head_dim=256)
        self.assertTrue(uses_step5_swa_window_align(Step5Config()))
        self.assertFalse(uses_step5_swa_window_align(other))

        b = FlashAttentionBackend.__new__(FlashAttentionBackend)
        b.fa_impl_ver = 4
        b.forward_metadata = None
        b.swa_fa4_window_align = uses_step5_swa_window_align(other)
        seqlens = torch.tensor([1000, 64], dtype=torch.int32)
        aligned, leftpad = b._swa_window_aligned_kvcache(
            True, seqlens, window_left=127, max_seqlen_q=1, extra_kwargs={}
        )
        self.assertIsNone(leftpad)
        self.assertIs(aligned, seqlens)


if __name__ == "__main__":
    unittest.main()
