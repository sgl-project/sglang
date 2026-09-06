"""Atomic image boundaries shared by scheduling, cache matching and attention."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.dsv4.visible_window import (
    compute_visible_window_overrides,
    image_span_aligned_extend_end,
    image_span_cut_point,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


def image_inputs(*spans):
    return SimpleNamespace(
        mm_items=[
            SimpleNamespace(
                is_image=lambda: True,
                offsets=[(start, end - 1)],
                model_specific_data={"types": [], "perm": []},
            )
            for start, end in spans
        ]
    )


class TestImageBoundaries(CustomTestCase):
    def test_all_cut_points(self):
        spans = [(13, 393), (501, 1001), (1001, 1121)]
        mm = image_inputs(*spans)
        for position in range(1200):
            expected = next(
                (start for start, end in spans if start < position < end), position
            )
            self.assertEqual(image_span_aligned_extend_end(mm, position), expected)
            self.assertEqual(
                image_span_cut_point(mm, position),
                expected if expected < position else None,
            )

    def test_shrink_respects_kv_budget(self):
        mm = image_inputs((1100, 1480))
        end = image_span_aligned_extend_end(mm, 1024 + 256)
        self.assertEqual(end, 1100)
        self.assertLessEqual(end - 1024, 256)

    def test_chunking_makes_progress_for_every_start_alignment(self):
        for start in range(512):
            mm = image_inputs((start, start + 500), (start + 500, start + 880))
            prefix = 0
            chunks = []
            while prefix < start + 1000:
                end = image_span_aligned_extend_end(mm, min(prefix + 512, start + 1000))
                self.assertGreater(end, prefix)
                self.assertLessEqual(end - prefix, 512)
                chunks.append((prefix, end))
                prefix = end
            self.assertLessEqual(len(chunks), 4)

    def test_remaining_batch_budget_can_defer_image(self):
        mm = image_inputs((100, 480))
        self.assertEqual(image_span_aligned_extend_end(mm, 100 + 256), 100)
        self.assertEqual(image_span_aligned_extend_end(mm, 100 + 512), 612)

    def test_text_and_other_models_are_unchanged(self):
        other_mm = SimpleNamespace(
            mm_items=[
                SimpleNamespace(
                    is_image=lambda: True, offsets=[(0, 499)], model_specific_data={}
                )
            ]
        )
        for mm in (None, other_mm):
            self.assertEqual(image_span_aligned_extend_end(mm, 128), 128)
            self.assertIsNone(image_span_cut_point(mm, 128))

    def test_host_match_uses_device_plus_host_length(self):
        mm = image_inputs((100, 480), (520, 900))
        for device, host, expected in [
            (64, 64, 100),
            (128, 256, 100),
            (480, 64, 520),
            (64, 416, None),
        ]:
            self.assertEqual(image_span_cut_point(mm, device + host), expected)


class TestVisibleWindows(CustomTestCase):
    def test_matches_reference_for_all_four_compression_alignments(self):
        for start in range(128, 132):
            end = start + 380
            starts, lengths = compute_visible_window_overrides(
                mm_inputs=[image_inputs((start, end))],
                extend_prefix_lens=[64],
                extend_seq_lens=[512],
                swa_window=128,
                padded_num_tokens=520,
            )
            image_start = start + 3 - start % 4
            for row, pos in enumerate(range(64, 576)):
                in_image = image_start <= pos < end
                left = pos - image_start if in_image else 0
                right = end - 1 - pos if in_image else 0
                expected_start = max(0, pos - 127 - max(0, left - 127))
                self.assertEqual(
                    (starts[row], lengths[row]),
                    (expected_start, pos + right - expected_start + 1),
                )
            self.assertEqual(starts[512:], [0] * 8)
            self.assertEqual(lengths[512:], [1] * 8)

    def test_rejects_partial_images_instead_of_changing_attention(self):
        for prefix, end in [(0, 256), (101, 500), (256, 400)]:
            with self.assertRaisesRegex(ValueError, "crosses the prefill range"):
                compute_visible_window_overrides(
                    mm_inputs=[image_inputs((100, 480))],
                    extend_prefix_lens=[prefix],
                    extend_seq_lens=[end - prefix],
                    swa_window=128,
                    padded_num_tokens=end - prefix,
                )

    def test_cached_images_do_not_change_following_text(self):
        self.assertIsNone(
            compute_visible_window_overrides(
                mm_inputs=[image_inputs((100, 480))],
                extend_prefix_lens=[480],
                extend_seq_lens=[20],
                swa_window=128,
                padded_num_tokens=20,
            )
        )


if __name__ == "__main__":
    unittest.main()
