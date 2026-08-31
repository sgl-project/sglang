import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")


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


if __name__ == "__main__":
    unittest.main()
