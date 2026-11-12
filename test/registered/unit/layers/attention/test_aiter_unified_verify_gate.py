"""When AITER EAGLE target_verify and draft_extend take the unified_attention path."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE = aiter_backend._UNIFIED_VERIFY_MIN_PAGE_SIZE


def _enabled(
    use_mla=False, eagle_topk=1, page_size=_PAGE, gfx942=True, opt_out=None, **kw
):
    environ = {
        k: v for k, v in os.environ.items() if k != "SGLANG_AITER_UNIFIED_VERIFY"
    }
    if opt_out is not None:
        environ["SGLANG_AITER_UNIFIED_VERIFY"] = opt_out
    with (
        patch.dict(os.environ, environ, clear=True),
        patch.object(aiter_backend, "is_gfx942_supported", return_value=gfx942),
    ):
        return aiter_backend._unified_verify_enabled(
            use_mla, eagle_topk, page_size, **kw
        )


class TestAiterUnifiedVerifyGate(CustomTestCase):
    def test_gfx942_linear_chain_uses_unified_verify_by_default(self):
        self.assertTrue(_enabled())
        self.assertTrue(_enabled(page_size=2 * _PAGE))

    def test_unset_topk_counts_as_linear_chain(self):
        self.assertTrue(_enabled(eagle_topk=None))

    def test_small_pages_keep_legacy_verify_by_default(self):
        self.assertFalse(_enabled(page_size=1))
        self.assertFalse(_enabled(page_size=_PAGE // 2))

    def test_other_archs_keep_legacy_verify_by_default(self):
        self.assertFalse(_enabled(gfx942=False))

    def test_unified_attn_env_keeps_unified_verify_everywhere(self):
        for gfx942 in (True, False):
            for page_size in (1, _PAGE):
                self.assertTrue(
                    _enabled(
                        page_size=page_size,
                        gfx942=gfx942,
                        use_triton_unified_attention=True,
                    )
                )

    def test_no_speculative_decoding_needs_no_verify_path(self):
        self.assertFalse(_enabled(speculative=False))
        self.assertFalse(_enabled(speculative=False, use_triton_unified_attention=True))

    def test_mla_keeps_its_own_verify_path(self):
        self.assertFalse(_enabled(use_mla=True))

    def test_tree_draft_keeps_legacy_verify(self):
        self.assertFalse(_enabled(eagle_topk=2))
        self.assertFalse(_enabled(eagle_topk=2, use_triton_unified_attention=True))

    def test_vectorized_5d_kv_keeps_legacy_verify(self):
        self.assertFalse(_enabled(kv_cache_is_vectorized_5d=True))
        self.assertFalse(
            _enabled(kv_cache_is_vectorized_5d=True, use_triton_unified_attention=True)
        )

    def test_env_opt_out_keeps_legacy_verify(self):
        self.assertFalse(_enabled(opt_out="0"))
        self.assertFalse(_enabled(opt_out="0", use_triton_unified_attention=True))


class TestAiterUnifiedDraftExtendGate(CustomTestCase):
    def test_draft_extend_moves_only_with_unified_attn(self):
        enabled = aiter_backend._unified_draft_extend_enabled
        self.assertFalse(enabled(True, False))
        self.assertTrue(enabled(True, True))
        self.assertFalse(enabled(False, True))
        self.assertFalse(enabled(False, False))


class TestAiterUnifiedPageTableWidth(CustomTestCase):
    """Verify writes ceil((seq_len + draft tokens) / page) page ids per row."""

    def _blocks(self, context_len, page_size, req_to_token_cols):
        backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
        backend.max_context_len = context_len
        backend.page_size = page_size
        backend.req_to_token = (
            None
            if req_to_token_cols is None
            else torch.zeros((2, req_to_token_cols), dtype=torch.int32)
        )
        return backend._unified_page_table_blocks()

    def test_covers_speculative_headroom_past_context_len(self):
        # context 4096, 8 draft tokens: the longest admitted prompt (4091) plus
        # its verify tokens needs 257 pages of 16; req_to_token rows hold
        # context_len + 4 + draft tokens.
        blocks = self._blocks(4096, 16, 4096 + 4 + 8)
        self.assertGreaterEqual(blocks * 16, 4091 + 8)
        self.assertEqual(blocks, 257)

    def test_never_narrower_than_context_len(self):
        self.assertEqual(self._blocks(4096, 16, 4096), 256)
        self.assertEqual(self._blocks(4096, 16, None), 256)


if __name__ == "__main__":
    unittest.main()
