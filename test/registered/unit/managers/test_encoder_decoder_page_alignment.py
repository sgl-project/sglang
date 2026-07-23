"""Unit tests for encoder-decoder KV page-alignment in prepare_encoder_info_extend.

The decoder KV region must start on a page boundary, so pad_input_ids reserves a
page-aligned number of encoder slots (``ceil_align(encoder_len, page_size)``). The
encoder still writes only its true ``encoder_len`` KV vectors; the decoder region
is offset/stripped by the padded reserve. ``ceil_align(x, 1) == x`` so page_size==1
(CUDA and every other backend today) is a byte-identical no-op.

The encoder is cached all-or-nothing, so ``len(prefix_indices)`` is only ever 0
(fresh) or >= encoder_len (cached); the branch predicate is
``len(prefix_indices) < encoder_len``.
"""

import types
import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ForwardMode, ScheduleBatch  # noqa: E402
from sglang.srt.utils.common import ceil_align  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (page_size, encoder_len) pairs spanning: page_size==1 no-op, already page-aligned
# (reserve == encoder_len, zero padding), and rounded-up (reserve > encoder_len).
ALIGNMENT_CASES = [
    (1, 1),
    (1, 1500),
    (16, 1500),
    (64, 1500),
    (128, 128),  # already aligned
    (128, 1500),  # rounds up 1500 -> 1536
    (128, 2048),  # already aligned
]


def _make_req(prefix_len: int, extend_len: int, encoder_len: int):
    """Minimal Req stand-in. encoder_len == 0 models a request with no encoder
    (multimodal_inputs is None)."""
    if encoder_len == 0:
        multimodal_inputs = None
    else:
        multimodal_inputs = types.SimpleNamespace(num_image_tokens=encoder_len)
    return types.SimpleNamespace(
        multimodal_inputs=multimodal_inputs,
        prefix_indices=list(range(prefix_len)),
        extend_range=types.SimpleNamespace(
            length=extend_len,
            start=0,
            _replace=lambda **kw: types.SimpleNamespace(
                length=extend_len, start=kw.get("start", 0), _replace=lambda **k: None
            ),
        ),
        logprob_start_len=0,
    )


def _make_batch(reqs, page_size, total_extend_tokens):
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cpu"
    batch.model_config = types.SimpleNamespace(is_encoder_decoder=True)
    batch.forward_mode = ForwardMode.EXTEND
    batch.extend_lens = [r.extend_range.length for r in reqs]
    batch.prefix_lens = [len(r.prefix_indices) for r in reqs]
    batch.extend_num_tokens = sum(batch.extend_lens)
    batch.extend_input_logprob_token_ids = None
    # Contiguous physical slots for the whole extend region.
    batch.out_cache_loc = torch.arange(total_extend_tokens, dtype=torch.int64)
    # page_size is read via get_server_args().page_size inside the method.
    batch._test_page_size = page_size
    return batch


class TestEncoderDecoderPageAlignment(CustomTestCase):
    def _run(self, batch, input_ids, seq_lens):
        server_args = types.SimpleNamespace(page_size=batch._test_page_size)
        with patch(
            "sglang.srt.managers.schedule_batch.get_server_args",
            return_value=server_args,
        ):
            batch.prepare_encoder_info_extend(input_ids, seq_lens)

    def test_fresh_encoder_is_page_aligned(self):
        """Fresh encoder: encoder KV = true encoder_len, decoder region starts at
        the page-aligned reserve, and seq_lens / extend_range.start are offset by
        the reserve — across page sizes and both aligned and rounded-up lengths."""
        decode_len = 4
        for page_size, encoder_len in ALIGNMENT_CASES:
            with self.subTest(page_size=page_size, encoder_len=encoder_len):
                reserve = ceil_align(encoder_len, page_size)
                extend_len = reserve + decode_len
                req = _make_req(
                    prefix_len=0, extend_len=extend_len, encoder_len=encoder_len
                )
                batch = _make_batch([req], page_size, total_extend_tokens=extend_len)

                self._run(batch, [array("q", [0] * extend_len)], [extend_len])

                self.assertEqual(len(batch.encoder_out_cache_loc), encoder_len)
                self.assertTrue(
                    torch.equal(
                        batch.out_cache_loc,
                        torch.arange(reserve, reserve + decode_len, dtype=torch.int64),
                    )
                )
                self.assertEqual(len(batch.out_cache_loc), batch.extend_num_tokens)
                self.assertEqual(batch.seq_lens_cpu.tolist(), [decode_len])
                self.assertEqual(req.extend_range.start, reserve)
                self.assertGreaterEqual(req.logprob_start_len, reserve)

    def test_fully_cached_encoder_takes_cached_branch(self):
        """A cached encoder (prefix == encoder_len, and the padded prefix ==
        reserve seen in production) must take the else branch: no new encoder KV,
        no ``assert len(prefix_indices) == 0``."""
        decode_len = 4
        for page_size, encoder_len in ALIGNMENT_CASES:
            reserve = ceil_align(encoder_len, page_size)
            # Both values a backend may report as "encoder fully cached".
            for cached_prefix in {encoder_len, reserve}:
                with self.subTest(
                    page_size=page_size,
                    encoder_len=encoder_len,
                    cached_prefix=cached_prefix,
                ):
                    req = _make_req(
                        prefix_len=cached_prefix,
                        extend_len=decode_len,
                        encoder_len=encoder_len,
                    )
                    batch = _make_batch(
                        [req], page_size, total_extend_tokens=decode_len
                    )

                    self._run(batch, [array("q", [1] * decode_len)], [decode_len])

                    self.assertEqual(len(batch.encoder_out_cache_loc), 0)
                    self.assertEqual(len(batch.out_cache_loc), decode_len)

    def test_partial_encoder_cache_raises(self):
        """The encoder is cached all-or-nothing: 0 < prefix_indices < encoder_len
        is invalid and must trip ``assert len(prefix_indices) == 0``."""
        page_size, encoder_len = 128, 1500
        reserve = ceil_align(encoder_len, page_size)
        extend_len = reserve + 4
        req = _make_req(
            prefix_len=encoder_len // 2, extend_len=extend_len, encoder_len=encoder_len
        )
        batch = _make_batch([req], page_size, total_extend_tokens=extend_len)

        with self.assertRaises(AssertionError):
            self._run(batch, [array("q", [0] * extend_len)], [extend_len])

    def test_mixed_batch_encoder_and_no_encoder(self):
        """Multi-req batch [fresh-encoder req, no-encoder req]: encoder KV holds
        only the first req's true encoder_len, decoder slots concatenate in order,
        and the no-encoder req passes through untouched."""
        page_size, encoder_len = 128, 1500
        reserve = ceil_align(encoder_len, page_size)
        decode0, ext1 = 4, 10
        ext0 = reserve + decode0
        total = ext0 + ext1

        req0 = _make_req(prefix_len=0, extend_len=ext0, encoder_len=encoder_len)
        req1 = _make_req(prefix_len=0, extend_len=ext1, encoder_len=0)
        batch = _make_batch([req0, req1], page_size, total_extend_tokens=total)

        self._run(batch, [array("q", [0] * ext0), array("q", [0] * ext1)], [ext0, ext1])

        self.assertTrue(
            torch.equal(
                batch.encoder_out_cache_loc,
                torch.arange(encoder_len, dtype=torch.int64),
            )
        )
        expected_decoder = torch.cat(
            [
                torch.arange(reserve, ext0, dtype=torch.int64),
                torch.arange(ext0, total, dtype=torch.int64),
            ]
        )
        self.assertTrue(torch.equal(batch.out_cache_loc, expected_decoder))
        self.assertEqual(len(batch.out_cache_loc), batch.extend_num_tokens)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [decode0, ext1])
        self.assertEqual(req1.extend_range.start, 0)

    def test_logprob_ids_stripped_per_req_encoder_len(self):
        """extend_input_logprob_token_ids: the fresh-encoder req is stripped by its
        own reserve while the no-encoder req in the same batch is untouched
        (regression against using a stale/shared encoder_len)."""
        page_size, encoder_len = 128, 1500
        reserve = ceil_align(encoder_len, page_size)
        decode0, ext1 = 4, 6
        ext0 = reserve + decode0
        total = ext0 + ext1

        req0 = _make_req(prefix_len=0, extend_len=ext0, encoder_len=encoder_len)
        req1 = _make_req(prefix_len=0, extend_len=ext1, encoder_len=0)
        batch = _make_batch([req0, req1], page_size, total_extend_tokens=total)
        batch.extend_logprob_start_lens = [0, 0]
        batch.extend_input_logprob_token_ids = torch.arange(total, dtype=torch.int64)

        self._run(batch, [array("q", [0] * ext0), array("q", [0] * ext1)], [ext0, ext1])

        expected = torch.cat(
            [
                torch.arange(
                    reserve, ext0, dtype=torch.int64
                ),  # req0: reserve stripped
                torch.arange(ext0, total, dtype=torch.int64),  # req1: untouched
            ]
        )
        self.assertTrue(torch.equal(batch.extend_input_logprob_token_ids, expected))
        self.assertEqual(batch.extend_logprob_start_lens, [0, 0])


if __name__ == "__main__":
    unittest.main()
