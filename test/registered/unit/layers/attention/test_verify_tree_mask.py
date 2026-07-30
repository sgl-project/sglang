"""Unit tests for the VerifyTreeMask scratch: worst-case sizing, the
allocate-only-for-a-verifying-target gate, and HybridAttnBackend handing out
the scratch of whichever child actually runs target verify."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.verify_tree_mask import (
    VerifyTreeMask,
    maybe_create_verify_tree_mask,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _create(**overrides):
    kwargs = dict(
        is_draft_runner=False,
        skip_prefill=False,
        max_num_tokens=12,
        max_context_len=128,
        num_draft_tokens=3,
        device="cpu",
        is_read=True,
    )
    kwargs.update(overrides)
    return maybe_create_verify_tree_mask(**kwargs)


class TestVerifyTreeMaskSizing(CustomTestCase):
    def test_covers_what_the_tree_kernel_writes(self):
        """Must hold num_draft_tokens * (seq_len + num_draft_tokens) cells per
        request at max context -- the bound the tree kernel writes up to."""
        max_bs, num_draft_tokens, max_context_len = 4, 3, 128

        mask = _create(
            max_num_tokens=max_bs * num_draft_tokens,
            max_context_len=max_context_len,
            num_draft_tokens=num_draft_tokens,
        )

        worst_case_cells = (
            max_bs * num_draft_tokens * (max_context_len + num_draft_tokens)
        )
        self.assertGreaterEqual(mask.buffer.numel(), worst_case_cells)
        self.assertEqual(mask.buffer.dtype, torch.bool)
        self.assertTrue(mask.is_read)

    def test_honors_dtype_and_is_read(self):
        mask = _create(dtype=torch.uint8, is_read=False)
        self.assertEqual(mask.buffer.dtype, torch.uint8)
        self.assertFalse(mask.is_read)


class TestVerifyTreeMaskGate(CustomTestCase):
    def test_allocated_for_a_verifying_target(self):
        self.assertIsNotNone(_create())

    def test_skipped_when_nothing_verifies(self):
        for label, overrides in (
            ("draft runner never verifies", {"is_draft_runner": True}),
            ("decode-only target never verifies", {"skip_prefill": True}),
            ("no spec -> no tree", {"num_draft_tokens": None}),
            ("zero draft tokens", {"num_draft_tokens": 0}),
        ):
            with self.subTest(label):
                self.assertIsNone(_create(**overrides))


class _FakeAttnBackend:
    def __init__(self, verify_tree_mask):
        self.needs_cpu_seq_lens = False
        self.verify_tree_mask = verify_tree_mask


def _make_hybrid_backend(speculative_attention_mode, prefill_mask, decode_mask):
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
        prefill_backend=_FakeAttnBackend(prefill_mask),
        decode_backend=_FakeAttnBackend(decode_mask),
    )


class TestHybridAttnBackendHandsOutSelectedChildScratch(CustomTestCase):
    """The child that runs target verify is the one whose scratch is used --
    a wrapper that forwarded only part of this would silently fall back to
    allocating a fresh mask every verify step."""

    def test_decode_mode_uses_decode_child(self):
        prefill_mask = VerifyTreeMask(buffer=torch.zeros(4, dtype=torch.bool))
        decode_mask = VerifyTreeMask(
            buffer=torch.ones(4, dtype=torch.bool), is_read=False
        )

        backend = _make_hybrid_backend("decode", prefill_mask, decode_mask)

        self.assertIs(backend.verify_tree_mask, decode_mask)

    def test_prefill_mode_uses_prefill_child(self):
        prefill_mask = VerifyTreeMask(
            buffer=torch.zeros(4, dtype=torch.bool), is_read=False
        )
        decode_mask = VerifyTreeMask(buffer=torch.ones(4, dtype=torch.bool))

        backend = _make_hybrid_backend("prefill", prefill_mask, decode_mask)

        self.assertIs(backend.verify_tree_mask, prefill_mask)


if __name__ == "__main__":
    unittest.main()
