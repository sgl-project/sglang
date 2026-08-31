"""The DeepSeek-V4-Flash-Vision sliding-window gather, widened over image spans.

A query inside an image's ``[IMAGE_START, IMAGE_END]`` span attends across the
whole span in both directions; everywhere else the window stays causal. The
gather that expresses this feeds ``flash_mla_with_kvcache``'s ``indices`` /
``topk_length``, so the properties worth pinning are: it degenerates exactly to
the causal builder when no image tokens are present, its triton and torch paths
agree, and a token in a span really does reach past the 128-token window.
"""

import dataclasses
import inspect
import re
import unittest

import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    ImageSpan,
    _vision_window_extent,
    build_causal_swa_page_indices_triton,
    build_image_visible_spans,
    build_vision_swa_page_indices,
    build_vision_swa_page_indices_triton,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
from sglang.srt.multimodal.deepseek_v4_vl_image_processing import COMPRESS_PAD_TO
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

SWA_WINDOW = 128
MAX_IMAGE_TOKENS = 384
WIDTH = SWA_WINDOW + MAX_IMAGE_TOKENS
PAGE_INDEX_ALIGNED_SIZE = 64
NUM_REQS, MAX_LEN, NUM_SLOTS = 8, 4096, 40000


class TestDeepseekV4VisionSwaGather(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        torch.manual_seed(0)
        cls.req_to_token = torch.randint(
            0, NUM_SLOTS, (NUM_REQS, MAX_LEN), dtype=torch.int32, device=cls.device
        )
        cls.full_to_swa = torch.randint(
            0, 9999, (NUM_SLOTS + 1,), dtype=torch.int32, device=cls.device
        )

    def _vision(self, builder, seq_lens, reqs, left, right):
        return builder(
            req_to_token=self.req_to_token,
            full_to_swa_mapping=self.full_to_swa,
            req_pool_indices_repeated=reqs,
            seq_lens_casual=seq_lens,
            visible_left=left,
            visible_right=right,
            swa_window=SWA_WINDOW,
            width=WIDTH,
        )

    def _image_batch(self, span_lens):
        """One request: text, then one image block per span length, then text.

        Returns the flat-batch positions plus the visible counts the backend
        would resolve from the items' offsets.
        """
        spans, cursor = [], 0
        for span_len in span_lens:
            cursor += 3  # leading text
            compress_pad = COMPRESS_PAD_TO - 1 - cursor % COMPRESS_PAD_TO
            row_start = cursor + compress_pad
            row_end = row_start + span_len - 1
            spans.append(
                ImageSpan(row_start=row_start, row_end=row_end, left_origin=row_start)
            )
            cursor = row_end + 1 + 2  # trailing text
        positions = torch.arange(cursor, dtype=torch.int32, device=self.device)
        left, right = build_image_visible_spans(
            spans=spans,
            num_tokens=cursor,
            max_image_tokens=MAX_IMAGE_TOKENS,
            device=self.device,
        )
        return positions, left, right, spans

    def test_degenerates_to_the_causal_window_without_images(self):
        seq_lens = torch.randint(1, 2000, (600,), dtype=torch.int32, device=self.device)
        reqs = torch.randint(0, NUM_REQS, (600,), dtype=torch.int32, device=self.device)
        zeros = torch.zeros_like(seq_lens)
        causal = build_causal_swa_page_indices_triton(
            req_to_token=self.req_to_token,
            full_to_swa_mapping=self.full_to_swa,
            req_pool_indices_repeated=reqs,
            seq_lens_casual=seq_lens,
            swa_window=SWA_WINDOW,
            page_index_aligned_size=PAGE_INDEX_ALIGNED_SIZE,
        )
        causal_lengths = torch.clamp(seq_lens, max=SWA_WINDOW)
        indices, lengths = self._vision(
            build_vision_swa_page_indices_triton, seq_lens, reqs, zeros, zeros
        )
        self.assertTrue(torch.equal(lengths, causal_lengths))
        for row in range(seq_lens.numel()):
            length = int(causal_lengths[row])
            # The causal builder walks the window newest-first and the widened
            # one oldest-first, so compare the visible KV set, not its order.
            self.assertEqual(
                sorted(causal[row, :length].tolist()),
                sorted(indices[row, :length].tolist()),
                f"row {row}",
            )

    def test_triton_matches_torch(self):
        positions, left, right, _ = self._image_batch((37, 200, 381, 5))
        seq_lens = positions + 1
        reqs = torch.zeros_like(seq_lens)
        torch_indices, torch_lengths = self._vision(
            build_vision_swa_page_indices, seq_lens, reqs, left, right
        )
        triton_indices, triton_lengths = self._vision(
            build_vision_swa_page_indices_triton, seq_lens, reqs, left, right
        )
        self.assertTrue(torch.equal(torch_lengths, triton_lengths))
        self.assertTrue(torch.equal(torch_indices, triton_indices))

    def test_gathered_slots_are_the_visible_positions(self):
        positions, left, right, _ = self._image_batch((37, 200))
        seq_lens = positions + 1
        reqs = torch.zeros_like(seq_lens)
        indices, lengths = self._vision(
            build_vision_swa_page_indices_triton, seq_lens, reqs, left, right
        )
        first_pos, extents = _vision_window_extent(
            seq_lens, left, right, SWA_WINDOW, WIDTH
        )
        self.assertTrue(torch.equal(lengths, extents))
        for row in range(positions.numel()):
            length = int(extents[row])
            wanted = torch.arange(
                int(first_pos[row]),
                int(first_pos[row]) + length,
                device=self.device,
            )
            expected = self.full_to_swa[self.req_to_token[0, wanted].to(torch.long)].to(
                torch.int32
            )
            self.assertTrue(torch.equal(indices[row, :length], expected), f"row {row}")
            tail = indices[row, length:]
            if tail.numel():
                self.assertEqual(int(tail.max()), -1, f"row {row} tail")

    def test_copy_field_lists_cover_every_metadata_field(self):
        """DSV4AttnMetadata.copy_ must name every field of the struct.

        copy_metadata asserts that its three field lists partition the struct,
        so a field added without being listed raises only on the CUDA-graph
        replay path -- reachable with speculative decoding and easy to miss
        otherwise. Checking the lists here fails at import cost instead.
        """
        all_fields = {field.name for field in dataclasses.fields(DSV4AttnMetadata)}
        named = set(
            re.findall(r'"([a-z0-9_]+)"', inspect.getsource(DSV4AttnMetadata.copy_))
        )
        self.assertEqual(all_fields - named, set(), "field(s) missing from copy_")
        self.assertEqual(named - all_fields, set(), "copy_ names a non-field")

    def test_topk_length_never_exceeds_the_indices_row(self):
        """The clamp is what stands between a bad span and an out-of-bounds read.

        flash_mla_with_kvcache reads ``indices[0:topk_length]`` with no bound of
        its own, so both builders must cap the length at the gather width even
        when handed visible counts a correct span layout could not produce.
        """
        rows = 64
        # Positions far enough in that the backward reach stays inside the
        # request, and far enough from the end that the forward reach does too:
        # clipping the reach to the sequence is the collector's job, and this
        # test is about the width bound alone.
        seq_lens = torch.full((rows,), 1200, dtype=torch.int32, device=self.device)
        reqs = torch.zeros_like(seq_lens)
        # Deliberately impossible for a real span: both directions maxed at once,
        # so left + right exceeds one block.
        left = torch.full(
            (rows,), MAX_IMAGE_TOKENS - 1, dtype=torch.int32, device=self.device
        )
        right = torch.full(
            (rows,), MAX_IMAGE_TOKENS, dtype=torch.int32, device=self.device
        )
        _, unclamped = _vision_window_extent(seq_lens, left, right, SWA_WINDOW, 1 << 30)
        self.assertGreater(int(unclamped.max()), WIDTH)
        for builder in (
            build_vision_swa_page_indices,
            build_vision_swa_page_indices_triton,
        ):
            with self.subTest(builder=builder.__name__):
                indices, lengths = self._vision(builder, seq_lens, reqs, left, right)
                self.assertEqual(indices.shape[-1], WIDTH)
                self.assertLessEqual(int(lengths.max()), indices.shape[-1])

    def test_a_span_is_visible_in_both_directions(self):
        span_len = 381
        # The leading span puts the span under test past position SWA_WINDOW, so
        # its backward reach is the window rather than the start of the sequence.
        positions, left, right, spans = self._image_batch((200, span_len))
        first_pos, extents = _vision_window_extent(
            positions + 1, left, right, SWA_WINDOW, WIDTH
        )
        start, end = spans[1].row_start, spans[1].row_end
        self.assertGreater(start, SWA_WINDOW)
        # Backward reach is unchanged at the span's first token, but it now sees
        # forward all the way to IMAGE_END.
        self.assertEqual(int(first_pos[start]), start - (SWA_WINDOW - 1))
        self.assertEqual(int(first_pos[start]) + int(extents[start]) - 1, end)
        # The span's last token reaches back over the whole span, past the window.
        self.assertEqual(int(extents[end]), span_len)
        self.assertGreater(int(extents[end]), SWA_WINDOW)
        # And the gather still fits the widened width.
        self.assertLessEqual(int(extents.max()), WIDTH)


if __name__ == "__main__":
    unittest.main()
