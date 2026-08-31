"""The DeepSeek-V4-Flash-Vision sliding-window gather, widened over image spans.

A query inside an image's ``[IMAGE_START, IMAGE_END]`` span attends across the
whole span in both directions; everywhere else the window stays causal. The
gather that expresses this feeds ``flash_mla_with_kvcache``'s ``indices`` /
``topk_length``, so the properties worth pinning are: it degenerates exactly to
the causal builder when no image tokens are present, its triton and torch paths
agree, and a token in a span really does reach past the 128-token window.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    _vision_window_extent,
    build_causal_swa_page_indices_triton,
    build_vision_swa_page_indices,
    build_vision_swa_page_indices_triton,
    compute_image_visible_spans,
)
from sglang.srt.multimodal.deepseek_v4_vl_image_processing import COMPRESS_PAD_TO
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

SWA_WINDOW = 128
MAX_IMAGE_TOKENS = 384
WIDTH = SWA_WINDOW + MAX_IMAGE_TOKENS
PAGE_INDEX_ALIGNED_SIZE = 64
VOCAB_SIZE = 129280
# What MultimodalDataItem.pad_value looks like: MM_PAD_SHIFT_VALUE + a hash.
PAD_SENTINEL = 1_000_000 + 5

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
        """One request: text, an image block per span length, text. Returns the
        flat input_ids / positions plus each span's index range."""
        input_ids, spans = [], []
        for span_len in span_lens:
            input_ids.extend((11, 12, 13))
            compress_pad = COMPRESS_PAD_TO - 1 - len(input_ids) % COMPRESS_PAD_TO
            start = len(input_ids) + compress_pad
            input_ids.extend([PAD_SENTINEL] * (compress_pad + span_len))
            spans.append((start, start + span_len - 1))
            input_ids.extend((14, 15))
        positions = torch.arange(len(input_ids), dtype=torch.int32, device=self.device)
        return (
            torch.tensor(input_ids, device=self.device),
            positions,
            spans,
        )

    def _spans(self, input_ids, positions):
        return compute_image_visible_spans(
            input_ids=input_ids,
            positions=positions,
            vocab_size=VOCAB_SIZE,
            compress_pad_to=COMPRESS_PAD_TO,
            max_image_tokens=MAX_IMAGE_TOKENS,
        )

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
        input_ids, positions, _ = self._image_batch((37, 200, 381, 5))
        left, right = self._spans(input_ids, positions)
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
        input_ids, positions, _ = self._image_batch((37, 200))
        left, right = self._spans(input_ids, positions)
        seq_lens = positions + 1
        reqs = torch.zeros_like(seq_lens)
        indices, lengths = self._vision(
            build_vision_swa_page_indices_triton, seq_lens, reqs, left, right
        )
        first_pos, extents = _vision_window_extent(seq_lens, left, right, SWA_WINDOW)
        self.assertTrue(torch.equal(lengths, extents))
        for row in range(input_ids.numel()):
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

    def test_a_span_is_visible_in_both_directions(self):
        span_len = 381
        # The leading span puts the span under test past position SWA_WINDOW, so
        # its backward reach is the window rather than the start of the sequence.
        input_ids, positions, spans = self._image_batch((200, span_len))
        left, right = self._spans(input_ids, positions)
        first_pos, extents = _vision_window_extent(
            positions + 1, left, right, SWA_WINDOW
        )
        start, end = spans[1]
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
