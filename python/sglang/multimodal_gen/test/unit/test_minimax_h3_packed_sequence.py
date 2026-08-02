# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for strict FL and text-only packed-sequence layouts."""

import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)


class TestMiniMaxH3PackedSequence(unittest.TestCase):
    def test_t2va_structure(self):
        built = minimax_h3_packed_sequence(
            text_len=97,
            latent_t=62,
            latent_h=48,
            latent_w=76,
            audio_t=348,
            include_keyframe_cond=False,
        )
        self.assertTrue(built["update_mask"].all())
        self.assertEqual(int(built["img_pos"].shape[0]), 62 * 24 * 38)
        self.assertEqual(int(built["seq_len"]) % 64, 0)
        self.assertEqual(built["token_tags"][built["audio_pos"]].unique().tolist(), [2])

    def test_fl2va_first_last_cond_blocks_use_exact_rope_span(self):
        text_len = 11
        latent_t = 37
        built = minimax_h3_packed_sequence(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=48,
            latent_w=76,
            audio_t=203,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=124,
        )
        frame_rows = 24 * 38
        cond_rows = 2 * frame_rows
        self.assertEqual(int((~built["update_mask"]).sum()), cond_rows)
        self.assertEqual(
            int(built["img_pos"].shape[0]),
            (2 + latent_t) * frame_rows,
        )
        cond_pos = built["img_pos"][:cond_rows].reshape(2, frame_rows)
        cond_t = [
            float(built["img_position_ids"][positions, 0].unique().item())
            for positions in cond_pos
        ]
        frame_rescale = 5.0 / 3.0
        temporal_span = sum(
            frame_rescale * (1, 4, 4, 4, 4)[index % 5] for index in range(latent_t)
        )
        self.assertEqual(cond_t[0], float(text_len))
        self.assertAlmostEqual(
            cond_t[1],
            float(text_len) + temporal_span - frame_rescale,
            places=12,
        )
        self.assertFalse(built["update_mask"][:cond_rows].any())
        self.assertTrue(built["update_mask"][cond_rows:].all())

    def test_i2va_and_l2va_single_cond_blocks_use_endpoint_rope(self):
        text_len = 11
        latent_t = 37
        frame_count = 124
        frame_rescale = 5.0 / 3.0
        temporal_span = sum(
            frame_rescale * (1, 4, 4, 4, 4)[index % 5] for index in range(latent_t)
        )
        for semantic_index, expected_t in (
            (0, float(text_len)),
            (-1, float(text_len) + temporal_span - frame_rescale),
        ):
            with self.subTest(semantic_index=semantic_index):
                built = minimax_h3_packed_sequence(
                    text_len=text_len,
                    latent_t=latent_t,
                    latent_h=48,
                    latent_w=76,
                    audio_t=203,
                    include_keyframe_cond=True,
                    keyframe_frame_indices=[semantic_index],
                    frame_count=frame_count,
                )
                frame_rows = 24 * 38
                self.assertEqual(int((~built["update_mask"]).sum()), frame_rows)
                positions = built["img_pos"][:frame_rows]
                cond_t = float(built["img_position_ids"][positions, 0].unique().item())
                self.assertAlmostEqual(cond_t, expected_t, places=12)

    def test_fl2va_keyframe_index_validation_is_defensive(self):
        common = dict(
            text_len=11,
            latent_t=37,
            latent_h=48,
            latent_w=76,
            audio_t=203,
            include_keyframe_cond=True,
            frame_count=124,
        )
        for frame_indices in (
            None,
            [1],
            [0, 52],
            [0, 52, -1],
            [-1, 0],
            [0, True],
            ["0", -1],
        ):
            with (
                self.subTest(frame_indices=frame_indices),
                self.assertRaises(ValueError),
            ):
                minimax_h3_packed_sequence(
                    **common,
                    keyframe_frame_indices=frame_indices,
                )
        with self.assertRaises(ValueError):
            minimax_h3_packed_sequence(
                **{**common, "frame_count": None},
                keyframe_frame_indices=[0, -1],
            )
        with self.assertRaises(ValueError):
            minimax_h3_packed_sequence(
                **{**common, "frame_count": 1},
                keyframe_frame_indices=[0, -1],
            )
