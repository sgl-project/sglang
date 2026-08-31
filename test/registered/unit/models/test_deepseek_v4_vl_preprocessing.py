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
    compute_image_visible_spans,
)
from sglang.srt.entrypoints.openai import encoding_dsv4
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
VOCAB_SIZE = 129280
# What MultimodalDataItem.pad_value looks like: MM_PAD_SHIFT_VALUE + a hash.
PAD_SENTINEL = 1_000_000 + 12345


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


class TestDeepseekV4VLImageVisibleSpans(CustomTestCase):
    """Bidirectional visibility must cover exactly [IMAGE_START, IMAGE_END]."""

    WINDOW = 128

    @staticmethod
    def _batch(spans, leading_text=3, trailing_text=2, first_position=0):
        """One request: text, then one block per span length, then text."""
        input_ids, expected = [], []
        for span_len in spans:
            input_ids.extend(range(100, 100 + leading_text))
            compress_pad = COMPRESS_PAD_TO - 1 - len(input_ids) % COMPRESS_PAD_TO
            span_start = len(input_ids) + compress_pad
            input_ids.extend([PAD_SENTINEL] * (compress_pad + span_len))
            expected.append((span_start, span_start + span_len - 1))
            input_ids.extend(range(200, 200 + trailing_text))
        positions = torch.arange(
            first_position, first_position + len(input_ids), dtype=torch.int32
        )
        return torch.tensor(input_ids), positions, expected

    def _spans(self, input_ids, positions):
        return compute_image_visible_spans(
            input_ids=input_ids,
            positions=positions,
            vocab_size=VOCAB_SIZE,
            compress_pad_to=COMPRESS_PAD_TO,
            max_image_tokens=PARAMS.max_n_token,
        )

    def test_visibility_covers_the_span_and_nothing_else(self):
        input_ids, positions, expected = self._batch([37, 200])
        left, right = self._spans(input_ids, positions)
        inside = torch.zeros(len(input_ids), dtype=torch.bool)
        for start, end in expected:
            inside[start : end + 1] = True
            self.assertTrue(
                torch.equal(left[start : end + 1].long(), torch.arange(end - start + 1))
            )
            self.assertTrue(
                torch.equal(
                    right[start : end + 1].long(),
                    torch.arange(end - start, -1, -1),
                )
            )
        # Text and the block's lead padding stay on the plain causal window.
        self.assertEqual(int(left[~inside].max()), 0)
        self.assertEqual(int(right[~inside].max()), 0)

    def test_runs_do_not_merge_across_requests(self):
        """A flat batch concatenates requests; positions restart at each one."""
        ids_a, pos_a, expected_a = self._batch([16])
        ids_b, pos_b, expected_b = self._batch([16])
        offset = len(ids_a)
        input_ids = torch.cat([ids_a, ids_b])
        positions = torch.cat([pos_a, pos_b])
        left, right = self._spans(input_ids, positions)
        for start, end in expected_a:
            self.assertEqual(int(left[end]), end - start)
        for start, end in expected_b:
            self.assertEqual(int(left[offset + end]), end - start)

    def test_gather_stays_within_the_widened_window(self):
        max_span = PARAMS.max_n_token - (COMPRESS_PAD_TO - 1)
        input_ids, positions, _ = self._batch([max_span])
        left, right = self._spans(input_ids, positions)
        _, lengths = _vision_window_extent(positions + 1, left, right, self.WINDOW)
        self.assertLessEqual(int(lengths.max()), self.WINDOW + PARAMS.max_n_token)

    def test_window_extent_matches_the_reference_formula(self):
        input_ids, positions, _ = self._batch([37, 200])
        left, right = self._spans(input_ids, positions)
        first_pos, lengths = _vision_window_extent(
            positions + 1, left, right, self.WINDOW
        )
        for i, position in enumerate(positions.tolist()):
            # Reference: start = idx - max(window - 1, left), end = idx + right.
            back = min(position, max(self.WINDOW - 1, int(left[i])))
            self.assertEqual(int(first_pos[i]), position - back)
            self.assertEqual(int(lengths[i]), back + 1 + int(right[i]))


if __name__ == "__main__":
    unittest.main()
