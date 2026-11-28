"""Regression: FA3 page_table width must reserve draft_extend headroom.

Speculative ``draft_extend`` / ``target_verify`` paths set ``cache_seqlens`` to
``seq_lens + speculative_num_draft_tokens``. ``FlashAttentionBackend`` sizes CUDA
graph page tables from ``max_context_len``; without widening by
``num_draft_tokens``, FA indexes past the page table near the context wall
(CUDA illegal memory access).

"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

CONTEXT_LEN = 64
NUM_DRAFT = 4


def _make_runner(*, context_len: int = CONTEXT_LEN, page_size: int = 16):
    device = "cuda"
    req_to_token_pool = SimpleNamespace(
        size=4,
        req_to_token=torch.zeros(
            5,
            context_len + NUM_DRAFT,
            dtype=torch.int32,
            device=device,
        ),
    )
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        context_len=context_len,
        attention_arch=AttentionArch.MHA,
        is_local_attention_model=False,
        head_dim=8,
        hf_text_config=SimpleNamespace(
            num_attention_heads=2, attn_logit_softcapping=None
        ),
        get_num_kv_heads=lambda tp_size: 2,
    )
    return SimpleNamespace(
        sliding_window_size=None,
        model_config=model_config,
        device=device,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=object(),
        token_to_kv_pool_allocator=object(),
        kv_cache_dtype=torch.float16,
        kv_cache_dtype_str="auto",
        page_size=page_size,
        ps=SimpleNamespace(attn_cp_size=1, tp_size=1),
        is_draft_worker=False,
        server_args=None,
        attention_chunk_size=None,
        prefill_aware_swa=False,
    )


def _expected_num_pages(context_len: int, page_size: int) -> int:
    return (context_len + page_size - 1) // page_size


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlashAttentionSpecContextHeadroom(CustomTestCase):
    def _publish_runner(self, runner, **fields):
        override = get_context().override_server_args(**fields)
        server_args = override.install()
        runner.server_args = server_args
        runner.kv_cache_dtype_str = server_args.kv_cache_dtype
        self.addCleanup(override.restore)
        return runner

    def _backend(self, *, page_size: int = 16, **server_args_fields):
        runner = self._publish_runner(
            _make_runner(page_size=page_size), **server_args_fields
        )
        backend = FlashAttentionBackend(runner)
        return runner, backend

    def test_no_spec_uses_model_context_len(self):
        runner, backend = self._backend(page_size=16)

        self.assertIsNone(runner.server_args.speculative_num_draft_tokens)
        self.assertEqual(backend.max_context_len, CONTEXT_LEN)
        self.assertEqual(
            backend.max_num_pages,
            _expected_num_pages(CONTEXT_LEN, runner.page_size),
        )

    def test_spec_adds_draft_headroom(self):
        runner, backend = self._backend(
            page_size=16,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=NUM_DRAFT,
            speculative_num_steps=NUM_DRAFT - 1,
        )

        self.assertEqual(runner.server_args.speculative_num_draft_tokens, NUM_DRAFT)
        expected_len = CONTEXT_LEN + NUM_DRAFT
        self.assertEqual(backend.max_context_len, expected_len)
        self.assertEqual(
            backend.max_num_pages,
            _expected_num_pages(expected_len, runner.page_size),
        )

    def test_cuda_graph_buffers_match_widened_bound(self):
        runner, backend = self._backend(
            page_size=16,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=NUM_DRAFT,
            speculative_num_steps=NUM_DRAFT - 1,
        )
        max_bs = 4
        backend.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=16)

        expected_pages = _expected_num_pages(CONTEXT_LEN + NUM_DRAFT, runner.page_size)
        decode_meta = backend.decode_cuda_graph_metadata
        self.assertEqual(decode_meta["page_table"].shape, (max_bs, expected_pages))
        self.assertEqual(
            decode_meta["strided_indices"].numel(),
            expected_pages,
        )
        self.assertIn("draft_extend_metadata", backend.__dict__)
        self.assertEqual(
            backend.draft_extend_metadata["page_table"].shape,
            (max_bs, expected_pages),
        )
        self.assertEqual(
            backend.draft_extend_metadata["strided_indices"].numel(),
            expected_pages,
        )
        self.assertEqual(
            backend.target_verify_metadata["page_table"].shape,
            (max_bs, expected_pages),
        )


if __name__ == "__main__":
    unittest.main()
