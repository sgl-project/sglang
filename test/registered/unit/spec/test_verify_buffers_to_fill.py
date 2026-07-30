"""Unit tests for VerifyBuffersToFill: base-class defaults, and
HybridAttnBackend delegating both the buffer and the read-flag from the
same selected child in one call."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.base_attn_backend import VerifyBuffersToFill
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeAttnBackend:
    def __init__(self, verify_buffers: VerifyBuffersToFill):
        self.needs_cpu_seq_lens = False
        self._verify_buffers = verify_buffers

    def get_verify_buffers_to_fill_after_draft(self) -> VerifyBuffersToFill:
        return self._verify_buffers


def _make_hybrid_backend(speculative_attention_mode, prefill_buffers, decode_buffers):
    model_runner = SimpleNamespace(
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        server_args=SimpleNamespace(
            speculative_attention_mode=speculative_attention_mode
        ),
    )
    return HybridAttnBackend(
        model_runner,
        prefill_backend=_FakeAttnBackend(prefill_buffers),
        decode_backend=_FakeAttnBackend(decode_buffers),
    )


class TestVerifyBuffersToFillDefaults(unittest.TestCase):
    def test_base_defaults(self):
        buffers = VerifyBuffersToFill()
        self.assertIsNone(buffers.tree_mask)
        self.assertIsNone(buffers.positions)
        self.assertTrue(buffers.tree_mask_is_read)


class TestHybridAttnBackendVerifyBufferDelegation(CustomTestCase):
    """HybridAttnBackend must answer buffer + read-flag from the SAME
    selected child -- a single get_verify_buffers_to_fill_after_draft() call,
    not two independently-overridable hooks that could disagree."""

    def test_speculative_attention_mode_decode_selects_decode_backend(self):
        prefill_buffers = VerifyBuffersToFill(tree_mask=torch.zeros(4))
        decode_buffers = VerifyBuffersToFill(
            tree_mask=torch.ones(4), tree_mask_is_read=False
        )
        backend = _make_hybrid_backend("decode", prefill_buffers, decode_buffers)

        result = backend.get_verify_buffers_to_fill_after_draft()

        self.assertIs(result, decode_buffers)
        self.assertFalse(result.tree_mask_is_read)

    def test_speculative_attention_mode_prefill_selects_prefill_backend(self):
        prefill_buffers = VerifyBuffersToFill(
            tree_mask=torch.zeros(4), tree_mask_is_read=False
        )
        decode_buffers = VerifyBuffersToFill(tree_mask=torch.ones(4))
        backend = _make_hybrid_backend("prefill", prefill_buffers, decode_buffers)

        result = backend.get_verify_buffers_to_fill_after_draft()

        self.assertIs(result, prefill_buffers)
        self.assertFalse(result.tree_mask_is_read)


if __name__ == "__main__":
    unittest.main()
