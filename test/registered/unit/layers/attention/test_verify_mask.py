import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.verify_mask import (
    VerifyMask,
    maybe_create_verify_mask,
    tree_mask_numel,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, default_tree_mask_mode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_MAX_BS = 4
_DRAFT = 3
_MAX_CONTEXT_LEN = 128


def _create(**overrides):
    kwargs = dict(
        is_draft_runner=False,
        skip_prefill=False,
        max_bs=_MAX_BS,
        max_context_len=_MAX_CONTEXT_LEN,
        num_draft_tokens=_DRAFT,
        device="cpu",
        is_read=True,
    )
    kwargs.update(overrides)
    return maybe_create_verify_mask(**kwargs)


class TestVerifyMaskSizing(CustomTestCase):
    def test_read_mask_covers_its_layouts_write_bound(self):
        """Whichever layout a reader gets must cover what the kernel writes:
        FULL_MASK spans the context, QLEN_ONLY is bs * draft**2."""
        mask = _create()

        if mask.mode == TreeMaskMode.FULL_MASK:
            bound = _MAX_BS * _DRAFT * (_MAX_CONTEXT_LEN + _DRAFT)
        else:
            bound = _MAX_BS * _DRAFT * _DRAFT
        self.assertEqual(mask.mode, default_tree_mask_mode())
        self.assertGreaterEqual(mask.buffer.numel(), bound)

    def test_unread_mask_drops_the_context_dimension(self):
        """Nothing interprets an unread layout, so it takes the compact one --
        paying for the context dimension would be pure waste."""
        mask = _create(is_read=False)

        self.assertEqual(mask.mode, TreeMaskMode.QLEN_ONLY)
        self.assertGreaterEqual(mask.buffer.numel(), _MAX_BS * _DRAFT * _DRAFT)
        self.assertLess(mask.buffer.numel(), _MAX_CONTEXT_LEN)

    def test_honors_dtype_override(self):
        self.assertEqual(_create(dtype=torch.uint8).buffer.dtype, torch.uint8)


class TestVerifyMaskCapacity(CustomTestCase):
    """A batch past the captured max_bs must not silently reuse the buffer."""

    def test_compact_layout_fits_up_to_max_bs(self):
        # is_read=False pins QLEN_ONLY; the read layout is build-dependent.
        mask = _create(is_read=False)
        self.assertTrue(mask.fits(_MAX_BS))
        self.assertTrue(mask.fits(1))

    def test_compact_layout_does_not_fit_beyond_max_bs(self):
        mask = _create(is_read=False)
        self.assertFalse(mask.fits(_MAX_BS + 1))

    def test_full_mask_does_not_fit_beyond_max_bs(self):
        """FULL_MASK's context dimension is per-request slack, not spare room
        for extra requests -- it must not be exempt from the check. Built
        explicitly because default_tree_mask_mode() is host-dependent."""
        mask = VerifyMask(
            buffer=torch.zeros(
                tree_mask_numel(
                    TreeMaskMode.FULL_MASK, _MAX_BS, _DRAFT, _MAX_CONTEXT_LEN
                ),
                dtype=torch.bool,
            ),
            mode=TreeMaskMode.FULL_MASK,
            max_bs=_MAX_BS,
        )

        self.assertTrue(mask.fits(_MAX_BS))
        self.assertFalse(mask.fits(_MAX_BS + 1))


class TestVerifyMaskGate(CustomTestCase):
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
    def __init__(self, verify_mask):
        self.needs_cpu_seq_lens = False
        self.verify_mask = verify_mask


def _mask(numel, **kwargs):
    return VerifyMask(
        buffer=torch.zeros(numel, dtype=torch.bool),
        mode=TreeMaskMode.QLEN_ONLY,
        max_bs=_MAX_BS,
        **kwargs,
    )


def _make_hybrid_backend(speculative_attention_mode, prefill_mask, decode_mask):
    model_runner = SimpleNamespace(
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        server_args=SimpleNamespace(
            speculative_attention_mode=speculative_attention_mode
        ),
        model_config=SimpleNamespace(context_len=_MAX_CONTEXT_LEN),
    )
    return HybridAttnBackend(
        model_runner,
        prefill_backend=_FakeAttnBackend(prefill_mask),
        decode_backend=_FakeAttnBackend(decode_mask),
    )


class TestHybridAttnBackendHandsOutSelectedChildMask(CustomTestCase):
    """Forwarding the wrong child silently falls back to a fresh mask per step."""

    def test_decode_mode_uses_decode_child(self):
        prefill_mask, decode_mask = _mask(4), _mask(8, is_read=False)

        backend = _make_hybrid_backend("decode", prefill_mask, decode_mask)

        self.assertIs(backend.verify_mask, decode_mask)

    def test_prefill_mode_uses_prefill_child(self):
        prefill_mask, decode_mask = _mask(4, is_read=False), _mask(8)

        backend = _make_hybrid_backend("prefill", prefill_mask, decode_mask)

        self.assertIs(backend.verify_mask, prefill_mask)

    def test_capacity_check_needs_nothing_from_the_backend(self):
        backend = _make_hybrid_backend("prefill", _mask(64, is_read=False), None)

        self.assertTrue(backend.verify_mask.fits(_MAX_BS))


class TestTreeMaskNumel(CustomTestCase):
    def test_rejects_layouts_it_cannot_size(self):
        """A packed layout must raise, not silently take FULL_MASK's size."""
        with self.assertRaises(NotImplementedError):
            tree_mask_numel(
                TreeMaskMode.QLEN_ONLY_BITPACKING, 1, _DRAFT, _MAX_CONTEXT_LEN
            )


if __name__ == "__main__":
    unittest.main()
