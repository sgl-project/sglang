"""Unit tests for DeepSeek-V4-Flash-Vision prompt and image preprocessing.

Everything checked here has a counterpart in the checkpoint's own reference
implementation (``inference/image_processor.py``, ``inference/model.py``,
``encoding/encoding_dsv4.py``), so the assertions are written against the
reference's formulas rather than against this implementation's output.
"""

import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    _vision_window_extent,
    build_image_visible_spans,
)
from sglang.srt.entrypoints.openai import encoding_dsv4
from sglang.srt.layers.attention.deepseek_v4_backend import collect_image_spans
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.models.deepseek_v4_vl import DeepseekV4VisionModel
from sglang.srt.multimodal.deepseek_v4_vl_image_processing import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    DeepseekV4VisionParams,
    build_image_block,
    grid_tokens,
    preprocess_image,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PARAMS = DeepseekV4VisionParams(
    patch_size=14,
    downsample_ratio=3,
    max_n_token=384,
    min_pixels=147456,
    max_wh_ratio=8,
)
# The widened gather's width: the plain window plus at most one whole block.
WIDTH = 128 + PARAMS.max_n_token


class TestDeepseekV4VLImagePreprocessing(CustomTestCase):
    def test_block_fits_the_per_image_token_budget(self):
        """Every solved grid must fit max_n_token including its lead padding."""
        sizes = [(1, 1), (64, 64), (1920, 1080), (1080, 1920), (333, 777), (4000, 30)]
        sizes += [(w, h) for w in (17, 128, 512, 2001) for h in (23, 256, 900)]
        for width, height in sizes:
            with self.subTest(size=(width, height)):
                _, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(
                    Image.new("RGB", (width, height)), PARAMS
                )
                for start_pos in range(COMPRESS_PAD_TO):
                    types, _, _ = build_image_block(n_llm_h, n_llm_w, start_pos)
                    self.assertLessEqual(len(types), PARAMS.max_n_token)

    def test_patches_match_the_solved_grid(self):
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(
            Image.new("RGB", (1920, 1080)), PARAMS
        )
        self.assertEqual(patches.shape, (n_vit_h * n_vit_w, 3, 14, 14))
        self.assertEqual(patches.dtype, torch.bfloat16)
        # The aligner folds each 3x3 block of ViT patches into one LLM token.
        self.assertEqual(n_llm_h, -(-n_vit_h // PARAMS.downsample_ratio))
        self.assertEqual(n_llm_w, -(-n_vit_w // PARAMS.downsample_ratio))

    def test_extreme_aspect_ratios_do_not_raise(self):
        """The reference raises inside PIL on very tall images; we squash instead."""
        for size in ((2, 50000), (1, 900), (50000, 2)):
            with self.subTest(size=size):
                patches, *_ = preprocess_image(Image.new("RGB", size), PARAMS)
                self.assertGreater(patches.shape[0], 0)

    def test_block_layout_matches_the_reference_formulas(self):
        for n_llm_h, n_llm_w in ((8, 12), (14, 21), (1, 1), (3, 7), (2, 2)):
            for start_pos in range(8):
                with self.subTest(grid=(n_llm_h, n_llm_w), start_pos=start_pos):
                    types, perm, compress_pad = build_image_block(
                        n_llm_h, n_llm_w, start_pos
                    )
                    self.assertEqual(
                        compress_pad,
                        COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO,
                    )
                    # grid_tokens counts the block without the lead padding.
                    _, _, num_tokens = grid_tokens(
                        n_llm_h * PARAMS.downsample_ratio * PARAMS.patch_size,
                        n_llm_w * PARAMS.downsample_ratio * PARAMS.patch_size,
                        PARAMS.patch_size,
                        PARAMS.downsample_ratio,
                    )
                    self.assertEqual(len(types), num_tokens + compress_pad)

                    # One aligner output per IMAGE slot, and perm is a bijection.
                    self.assertEqual(int((types == IMAGE).sum()), n_llm_h * n_llm_w)
                    self.assertEqual(len(perm), n_llm_h * n_llm_w)
                    self.assertEqual(
                        sorted(perm.tolist()), list(range(n_llm_h * n_llm_w))
                    )

                    # Framing: exactly one START at the end of the lead padding
                    # and one END last, and nothing else outside the grid kinds.
                    self.assertEqual(types[compress_pad].item(), IMAGE_START)
                    self.assertEqual(types[-1].item(), IMAGE_END)
                    self.assertEqual(int((types == IMAGE_START).sum()), 1)
                    self.assertEqual(int((types == IMAGE_END).sum()), 1)
                    self.assertEqual(
                        int((types == IMAGE_NEW_LINE).sum()),
                        # one row terminator per real grid row
                        n_llm_h,
                    )
                    self.assertTrue(
                        bool(
                            (types[:compress_pad] == IMAGE_PAD).all()
                            if compress_pad
                            else True
                        )
                    )

                    # The grid content starts on a compression-group boundary.
                    self.assertEqual(
                        (start_pos + compress_pad + 1) % COMPRESS_PAD_TO, 0
                    )


class TestDeepseekV4VisionModelBlock(CustomTestCase):
    """The tower must return one row per block slot, framing included.

    Those rows are scattered onto the block's placeholder positions in
    input_ids, so a row-count or ordering mistake here shows up as a
    device-side assert deep inside the multimodal scatter rather than as a
    wrong answer. A tiny config exercises the layout on CPU.
    """

    @staticmethod
    def _config():
        return SimpleNamespace(
            hidden_size=32,
            vision_n_layers=2,
            vision_dim=16,
            vision_n_heads=2,
            vision_inter_dim=24,
            vision_patch_size=PARAMS.patch_size,
            vision_rope_theta=10000.0,
            vision_downsample_ratio=PARAMS.downsample_ratio,
        )

    def test_block_rows_and_framing_slots(self):
        torch.manual_seed(0)
        config = self._config()
        model = DeepseekV4VisionModel(config).to(torch.float32)

        items, expected_rows = [], 0
        start_pos = 0
        for n_vit_h, n_vit_w in ((4, 7), (9, 9)):
            n_llm_h = -(-n_vit_h // config.vision_downsample_ratio)
            n_llm_w = -(-n_vit_w // config.vision_downsample_ratio)
            slot_types, aligner_perm, _ = build_image_block(n_llm_h, n_llm_w, start_pos)
            start_pos += len(slot_types)
            expected_rows += len(slot_types)
            items.append(
                SimpleNamespace(
                    feature=torch.randn(
                        n_vit_h * n_vit_w,
                        3,
                        config.vision_patch_size,
                        config.vision_patch_size,
                    ),
                    n_vit_h=n_vit_h,
                    n_vit_w=n_vit_w,
                    slot_types=slot_types,
                    aligner_perm=aligner_perm,
                )
            )

        with torch.no_grad():
            out = model(items)
        self.assertEqual(out.shape, (expected_rows, config.hidden_size))

        # Each non-IMAGE slot must carry its learned framing embedding verbatim.
        learned = {
            IMAGE_START: model.image_start,
            IMAGE_PAD: model.image_pad,
            IMAGE_NEW_LINE: model.image_newline,
            IMAGE_END: model.image_end,
        }
        row = 0
        for item in items:
            for kind in item.slot_types.tolist():
                if kind != IMAGE:
                    self.assertTrue(
                        torch.equal(out[row], learned[kind].detach()),
                        f"row {row} kind {kind}",
                    )
                row += 1
        self.assertEqual(row, expected_rows)


class TestDeepseekV4VLPromptEncoding(CustomTestCase):
    """The encoder must place one placeholder per image, in prompt order."""

    @staticmethod
    def _encode(messages):
        messages, num_images = encoding_dsv4.process_image_messages(messages)
        return (
            encoding_dsv4.encode_messages(messages, thinking_mode="chat"),
            num_images,
        )

    def test_image_between_text_parts(self):
        prompt, num_images = self._encode(
            [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image"},
                        {"type": "text", "text": "after"},
                    ],
                },
            ]
        )
        self.assertEqual(num_images, 1)
        self.assertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>before\n\n"
            f"{encoding_dsv4.IMAGE_PLACEHOLDER}\n\nafter"
            "<｜Assistant｜></think>",
        )

    def test_multiple_images_keep_prompt_order(self):
        prompt, num_images = self._encode(
            [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "middle"},
                        {"type": "image"},
                    ],
                },
            ]
        )
        self.assertEqual(num_images, 2)
        self.assertEqual(prompt.count(encoding_dsv4.IMAGE_PLACEHOLDER), 2)

    def test_image_returned_by_a_tool_keeps_its_place(self):
        prompt, num_images = self._encode(
            [
                {"role": "system", "content": ""},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "tool image"},
                    ],
                },
            ]
        )
        self.assertEqual(num_images, 1)
        self.assertIn(
            f"<tool_result>{encoding_dsv4.IMAGE_PLACEHOLDER}\n\ntool image</tool_result>",
            prompt,
        )

    def test_text_only_prompt_is_unchanged(self):
        prompt, num_images = self._encode(
            [{"role": "system", "content": ""}, {"role": "user", "content": "hello"}]
        )
        self.assertEqual(num_images, 0)
        self.assertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>",
        )

    def test_a_placeholder_in_user_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "image placeholder token"):
            self._encode(
                [
                    {
                        "role": "user",
                        "content": encoding_dsv4.IMAGE_PLACEHOLDER,
                    }
                ]
            )


class TestDeepseekV4VLImageSpanCollection(CustomTestCase):
    """Span extents come from the items' offsets, not from placeholder runs.

    Each case below is a layout where inferring "one image block in one
    request" from a maximal run of out-of-vocabulary ids at contiguous
    positions gives the wrong answer.
    """

    WINDOW = 128

    @staticmethod
    def _item(block_start, span_len):
        """One image whose block starts at ``block_start`` in its request."""
        compress_pad = COMPRESS_PAD_TO - 1 - block_start % COMPRESS_PAD_TO
        block_end = block_start + compress_pad + span_len - 1
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.zeros(1, 3, PARAMS.patch_size, PARAMS.patch_size),
            offsets=[(block_start, block_end)],
            model_specific_data={"compress_pad": compress_pad},
        )
        return item, block_start + compress_pad, block_end

    def _collect(self, mm_items_per_req, prefix_lens, extend_lens):
        mm_inputs = [
            None if items is None else MultimodalInputs(mm_items=items)
            for items in mm_items_per_req
        ]
        return collect_image_spans(
            mm_inputs=mm_inputs,
            extend_prefix_lens=prefix_lens,
            extend_seq_lens=extend_lens,
            swa_window=self.WINDOW,
        )

    def _visible(self, spans, num_tokens):
        return build_image_visible_spans(
            spans=spans,
            num_tokens=num_tokens,
            max_image_tokens=PARAMS.max_n_token,
            device="cpu",
        )

    def test_adjacent_blocks_stay_separate(self):
        """Two images can land back to back; they are two spans, not one.

        Merging them would let the first image attend into the second and, at
        full block size, push the gather length past the indices row it
        addresses.
        """
        span_len = PARAMS.max_n_token - (COMPRESS_PAD_TO - 1)
        first, first_start, first_end = self._item(0, span_len)
        second, second_start, second_end = self._item(first_end + 1, span_len)
        total = second_end + 1

        spans = self._collect([[first, second]], [0], [total])
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            [(s.row_start, s.row_end) for s in spans],
            [(first_start, first_end), (second_start, second_end)],
        )

        left, right = self._visible(spans, total)
        # The first block's last token sees nothing forward: its span ends there.
        self.assertEqual(int(right[first_end]), 0)
        self.assertEqual(int(left[second_start]), 0)
        _, lengths = _vision_window_extent(
            torch.arange(1, total + 1, dtype=torch.int32),
            left,
            right,
            self.WINDOW,
            WIDTH,
        )
        self.assertLessEqual(int(lengths.max()), WIDTH)

    def test_requests_with_contiguous_positions_stay_separate(self):
        """Request B's prefix can continue A's positions across the batch.

        A merge there would give A's tokens a forward reach past their own
        sequence end, into req_to_token slots A never wrote.
        """
        first, _, first_end = self._item(3, 16)
        len_a = first_end + 1 + 2
        second, _, second_end = self._item(0, 30)
        len_b = second_end + 1

        spans = self._collect([[first], [second]], [0, len_a], [len_a, len_b])
        self.assertEqual(len(spans), 2)
        left, right = self._visible(spans, len_a + len_b)

        # A's span ends inside A, and A's trailing text is on the plain window.
        self.assertEqual(spans[0].row_end, first_end)
        self.assertEqual(int(right[first_end]), 0)
        self.assertEqual(int(left[first_end + 1 : len_a].max()), 0)
        self.assertEqual(int(right[first_end + 1 : len_a].max()), 0)
        # B's span lives entirely in B's rows.
        self.assertGreaterEqual(spans[1].row_start, len_a)

    def test_a_split_block_keeps_its_true_start_and_stops_at_the_chunk(self):
        """A chunk opening mid-block must not re-anchor the span to the chunk.

        Backward reach is clipped to the oldest SWA-resident token rather than
        to the chunk start, and forward reach stops at the chunk's last token,
        whose KV is the newest that exists.
        """
        block_start, span_len = 100, 300
        item, span_start, block_end = self._item(block_start, span_len)
        prefix_len, extend_len = 250, 200
        chunk_last = prefix_len + extend_len - 1
        resident_first = max(0, prefix_len - (self.WINDOW - 1))

        spans = self._collect([[item]], [prefix_len], [extend_len])
        self.assertEqual(len(spans), 1)
        span = spans[0]
        # The span opened before this chunk, so left counts from before row 0.
        self.assertLess(span.left_origin, span.row_start)
        self.assertEqual(span.row_start, 0)
        self.assertEqual(span.row_end, min(block_end, chunk_last) - prefix_len)

        left, right = self._visible(spans, extend_len)
        self.assertEqual(
            int(left[span.row_start]), prefix_len - max(span_start, resident_first)
        )
        self.assertEqual(int(right[span.row_end]), 0)
        _, lengths = _vision_window_extent(
            torch.arange(
                prefix_len + 1, prefix_len + extend_len + 1, dtype=torch.int32
            ),
            left,
            right,
            self.WINDOW,
            WIDTH,
        )
        self.assertLessEqual(int(lengths.max()), WIDTH)

    def test_a_chunk_before_the_block_yields_no_span(self):
        item, _, _ = self._item(4000, 100)
        self.assertEqual(self._collect([[item]], [0], [512]), [])

    def test_text_only_requests_shift_the_row_cursor(self):
        """A text-only request between two image ones must still advance rows."""
        first, _, first_end = self._item(0, 8)
        second, _, second_end = self._item(0, 8)
        len_a, len_text, len_b = first_end + 1, 17, second_end + 1
        spans = self._collect(
            [[first], None, [second]], [0, 0, 0], [len_a, len_text, len_b]
        )
        self.assertEqual(len(spans), 2)
        # Identical geometry, so the second span sits at the same offset inside
        # its own request's rows as the first does inside its own.
        third_req_row = len_a + len_text
        self.assertEqual(spans[1].row_start - third_req_row, spans[0].row_start)
        self.assertLess(spans[1].row_end, third_req_row + len_b)

    def test_visibility_covers_the_span_and_nothing_else(self):
        item, span_start, block_end = self._item(3, 37)
        total = block_end + 1 + 5
        spans = self._collect([[item]], [0], [total])
        left, right = self._visible(spans, total)
        span_len = block_end - span_start + 1
        self.assertTrue(
            torch.equal(left[span_start : block_end + 1].long(), torch.arange(span_len))
        )
        self.assertTrue(
            torch.equal(
                right[span_start : block_end + 1].long(),
                torch.arange(span_len - 1, -1, -1),
            )
        )
        outside = torch.ones(total, dtype=torch.bool)
        outside[span_start : block_end + 1] = False
        self.assertEqual(int(left[outside].max()), 0)
        self.assertEqual(int(right[outside].max()), 0)

    def test_window_extent_matches_the_reference_formula(self):
        item, _, block_end = self._item(3, 200)
        total = block_end + 1
        spans = self._collect([[item]], [0], [total])
        left, right = self._visible(spans, total)
        positions = torch.arange(total, dtype=torch.int32)
        first_pos, lengths = _vision_window_extent(
            positions + 1, left, right, self.WINDOW, WIDTH
        )
        for i, position in enumerate(positions.tolist()):
            # Reference: start = idx - max(window - 1, left), end = idx + right.
            back = min(position, max(self.WINDOW - 1, int(left[i])))
            self.assertEqual(int(first_pos[i]), position - back)
            self.assertEqual(int(lengths[i]), back + 1 + int(right[i]))


if __name__ == "__main__":
    unittest.main()
