# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for MiniMax-H3 packed-sequence layouts."""

import unittest

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)


class TestMiniMaxH3PackedSequence(unittest.TestCase):
    def test_trailing_padding_layout(self):
        self.assertEqual(trailing_padding_used_len(64, 61, (0, 61, 64)), 61)
        self.assertIsNone(trailing_padding_used_len(64, 31, (0, 32, 64)))
        self.assertIsNone(trailing_padding_used_len(64, 61, (0, 61)))

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
        self.assertEqual(
            built["stream_layout"],
            {
                "target_shape": (62, 24, 38),
                "cond_image_shapes": (),
                "cond_image_roles": (),
                "cond_event_orders": (),
                "cond_audio_stream_lens": (),
            },
        )

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
        self.assertEqual(
            built["stream_layout"],
            {
                "target_shape": (37, 24, 38),
                "cond_image_shapes": ((1, 24, 38), (1, 24, 38)),
                "cond_image_roles": ("joint_cube", "joint_cube"),
                "cond_event_orders": (("imgvid", 0), ("imgvid", 1)),
                "cond_audio_stream_lens": (),
            },
        )

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
        for frame_indices in (None, [1], [0, 52, -1]):
            with (
                self.subTest(frame_indices=frame_indices),
                self.assertRaises(ValueError),
            ):
                minimax_h3_packed_sequence(
                    **common,
                    keyframe_frame_indices=frame_indices,
                )

    def test_ref2va_structure(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=97,
            latent_t=112,
            latent_h=48,
            latent_w=84,
            audio_t=631,
            ref_blocks=[
                {"kind": "image", "latent_h": 64, "latent_w": 48},
                {"kind": "audio", "ref_audio_t": 582},
            ],
        )

        self.assertEqual(int(built["seq_len"]) % 64, 0)
        self.assertEqual(int((~built["update_mask"]).sum()), 32 * 24)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 582 * 2)
        self.assertEqual(built["token_tags"][built["audio_pos"]].unique().tolist(), [2])

    def test_ref2va_mixed_media_preserves_temporal_origin(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
                {"kind": "audio", "ref_audio_t": 1},
            ],
        )

        self.assertEqual(int((~built["update_mask"]).sum()), 12)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 8)
        target_video_t0 = built["img_position_ids"][built["img_pos"][12], 0]
        target_audio_t0 = built["img_position_ids"][built["audio_pos"][8], 0]
        self.assertEqual(float(target_audio_t0), float(target_video_t0))
        self.assertEqual(
            built["stream_layout"],
            {
                "target_shape": (2, 2, 2),
                "cond_image_shapes": ((1, 2, 2), (2, 2, 2)),
                "cond_image_roles": ("dense_prefix", "independent_cube"),
                "cond_event_orders": (
                    ("imgvid", 0),
                    ("audio", 0),
                    ("imgvid", 1),
                    ("audio", 1),
                ),
                "cond_audio_stream_lens": (6, 2),
            },
        )

    def test_ref2va_hybrid_packs_keyframes_before_references(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 4}],
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )

        frame_rows = 4
        frozen_rows = 2 * frame_rows + frame_rows
        self.assertEqual(int((~built["update_mask"]).sum()), frozen_rows)
        keyframe_positions = built["img_pos"][: 2 * frame_rows].reshape(2, -1)
        keyframe_times = [
            float(built["img_position_ids"][positions, 0].unique().item())
            for positions in keyframe_positions
        ]
        target_origin = 6.0  # text origin 5 + one reference-image time slot
        self.assertEqual(keyframe_times[0], target_origin)
        self.assertAlmostEqual(
            keyframe_times[1], target_origin + 4 * (5.0 / 3.0), places=12
        )
        target_t0 = built["img_position_ids"][built["img_pos"][frozen_rows], 0]
        self.assertEqual(float(target_t0), target_origin)
