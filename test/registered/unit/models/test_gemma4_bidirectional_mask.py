import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_request(extend: int, prefix: int, image_spans=None):
    """One request spec. `image_spans` are inclusive (begin, end) token offsets."""
    if image_spans is None:
        return SimpleNamespace(extend=extend, prefix=prefix, mm_inputs=None)
    mm_items = [
        SimpleNamespace(is_image=lambda: True, offsets=list(image_spans)),
    ]
    return SimpleNamespace(
        extend=extend, prefix=prefix, mm_inputs=SimpleNamespace(mm_items=mm_items)
    )


def make_forward_batch(requests):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(requests),
        extend_seq_lens=torch.tensor([r.extend for r in requests]),
        extend_prefix_lens=torch.tensor([r.prefix for r in requests]),
        mm_inputs=[r.mm_inputs for r in requests],
    )


def run_prepare(requests):
    """Drive prepare_attn_masks against a stub batch; return the attention backend.

    `prepare_attn_masks` does not touch `self`, so it is invoked unbound.
    """
    backend = object.__new__(TritonAttnBackend)
    backend.forward_metadata = SimpleNamespace(custom_mask=None, mask_indptr=None)
    forward_batch = make_forward_batch(requests)
    input_ids = torch.zeros(sum(r.extend for r in requests), dtype=torch.long)
    with patch("sglang.srt.models.gemma4_mm.get_attn_backend", return_value=backend):
        Gemma4ForConditionalGeneration.prepare_attn_masks(
            None, forward_batch, input_ids, torch.bool
        )
    return backend


def slot_of(backend, requests, index):
    """The submatrix the flat custom mask reserves for `requests[index]`."""
    indptr = backend.forward_metadata.mask_indptr
    flat = backend.forward_metadata.custom_mask
    req = requests[index]
    return flat[indptr[index] : indptr[index + 1]].reshape(
        req.extend, req.extend + req.prefix
    )


def causal_slot(extend: int, prefix: int):
    return torch.ones(extend, extend + prefix, dtype=torch.bool).tril(diagonal=prefix)


# A genuinely contained multi-token image span; used to force mask installation
# so that the degenerate requests sharing the batch can be inspected.
CONTAINED = make_request(extend=8, prefix=4, image_spans=[(6, 9)])


class TestGemma4MaskSkippedWhenCausal(unittest.TestCase):
    """Batches in which no image span widens the mask must not install one."""

    def assert_no_mask(self, requests):
        backend = run_prepare(requests)
        self.assertIsNone(backend.forward_metadata.custom_mask)
        self.assertIsNone(backend.forward_metadata.mask_indptr)

    def test_text_only_batch(self):
        self.assert_no_mask([make_request(extend=6, prefix=0)])

    def test_image_span_wholly_in_cached_prefix(self):
        # Multi-turn follow-up: the image tokens were prefilled in an earlier
        # turn, so the span sits entirely below prefix_len and neither branch
        # of the offset dispatch fires.
        self.assert_no_mask([make_request(extend=6, prefix=12, image_spans=[(2, 5)])])

    def test_chunked_prefill_split_span(self):
        # Span crosses the chunk boundary, so it only lands in `split_images`.
        self.assert_no_mask([make_request(extend=8, prefix=8, image_spans=[(6, 10)])])

    def test_chunked_prefill_span_running_past_chunk_end(self):
        self.assert_no_mask([make_request(extend=8, prefix=8, image_spans=[(12, 20)])])

    def test_single_token_image_span(self):
        # im_begin == im_end rewrites only the diagonal element the causal fill
        # already set.
        self.assert_no_mask([make_request(extend=8, prefix=4, image_spans=[(6, 6)])])

    def test_several_degenerate_requests_together(self):
        self.assert_no_mask(
            [
                make_request(extend=6, prefix=12, image_spans=[(2, 5)]),
                make_request(extend=8, prefix=8, image_spans=[(6, 10)]),
                make_request(extend=8, prefix=4, image_spans=[(6, 6)]),
                make_request(extend=5, prefix=3),
            ]
        )


class TestGemma4MaskInstalledWhenBidirectional(unittest.TestCase):
    def test_contained_span_installs_non_causal_mask(self):
        requests = [make_request(extend=8, prefix=4, image_spans=[(6, 9)])]
        backend = run_prepare(requests)
        self.assertIsNotNone(backend.forward_metadata.custom_mask)
        mask = slot_of(backend, requests, 0)
        # Rows 2..5 are the span's query rows, columns 6..9 its key columns.
        self.assertTrue(mask[2:6, 6:10].all())
        # The upper-triangular part of that block is what the causal fill left
        # at zero, so the installed mask genuinely differs from tril.
        self.assertFalse(torch.equal(mask, causal_slot(8, 4)))
        self.assertTrue(mask[2, 9])

    def test_two_spans_one_contained_one_split(self):
        requests = [make_request(extend=8, prefix=4, image_spans=[(6, 9), (0, 5)])]
        backend = run_prepare(requests)
        self.assertIsNotNone(backend.forward_metadata.custom_mask)

    def test_mixed_batch_reserves_full_slot_for_text_only_request(self):
        text_only = make_request(extend=5, prefix=3)
        requests = [text_only, CONTAINED]
        backend = run_prepare(requests)
        self.assertIsNotNone(backend.forward_metadata.custom_mask)

        expected = [0]
        for req in requests:
            expected.append(expected[-1] + req.extend * (req.extend + req.prefix))
        self.assertEqual(
            backend.forward_metadata.mask_indptr.tolist(),
            expected,
        )
        self.assertEqual(
            backend.forward_metadata.custom_mask.numel(),
            expected[-1],
        )
        # The text-only request keeps a full, correctly strided slot.
        self.assertTrue(torch.equal(slot_of(backend, requests, 0), causal_slot(5, 3)))


class TestGemma4DegenerateSlotsAreExactlyCausal(unittest.TestCase):
    """Pin the degeneracy claim on the bytes the loop actually produces.

    The per-request loop body does not depend on whether some other request in
    the batch widened its mask, so pairing each degenerate request with a
    genuinely contained span exposes the exact buffer that would otherwise have
    been installed on its own.
    """

    def assert_slot_is_causal(self, degenerate):
        requests = [degenerate, CONTAINED]
        backend = run_prepare(requests)
        self.assertIsNotNone(backend.forward_metadata.custom_mask)
        self.assertTrue(
            torch.equal(
                slot_of(backend, requests, 0),
                causal_slot(degenerate.extend, degenerate.prefix),
            )
        )

    def test_span_in_cached_prefix_slot_is_causal(self):
        self.assert_slot_is_causal(
            make_request(extend=6, prefix=12, image_spans=[(2, 5)])
        )

    def test_split_span_slot_is_causal(self):
        self.assert_slot_is_causal(
            make_request(extend=8, prefix=8, image_spans=[(6, 10)])
        )

    def test_single_token_span_slot_is_causal(self):
        self.assert_slot_is_causal(
            make_request(extend=8, prefix=4, image_spans=[(6, 6)])
        )


if __name__ == "__main__":
    unittest.main()
