# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cosmos-Dreams frustum K/V retrieval: selector geometry, pose
recovery from unified rows, the per-layer archive view and request resolution."""

import math
import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.frustum import (
    POSE_CHUNK_FRAMES,
    FrustumHistorySelector,
    SelectedHistory,
    chunk_anchored_poses,
    denormalize_unified_slots,
    describe_selection,
    recover_unified_camera_poses,
    resolve_frustum_selector,
    rot6d_to_matrix,
)

IMAGE = (640, 360)
INTRINSICS = torch.tensor(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]], dtype=torch.float64
)


def _pose(translation=(0.0, 0.0, 0.0), yaw_degrees: float = 0.0) -> torch.Tensor:
    angle = math.radians(yaw_degrees)
    pose = torch.eye(4, dtype=torch.float64)
    pose[0, 0] = pose[2, 2] = math.cos(angle)
    pose[0, 2] = math.sin(angle)
    pose[2, 0] = -math.sin(angle)
    pose[:3, 3] = torch.tensor(translation, dtype=torch.float64)
    return pose


def _selector(poses, **sizes) -> FrustumHistorySelector:
    return FrustumHistorySelector(
        torch.stack(poses), INTRINSICS, IMAGE, 0.1, 10.0, **sizes
    )


class TestFrustumSelector(unittest.TestCase):
    def test_identity_and_disjoint_overlap(self):
        selector = _selector(
            [_pose(), _pose(), _pose((100.0, 0.0, 0.0)), _pose(yaw_degrees=180.0)]
        )
        scores = selector.overlap_scores([0], [1, 2, 3])
        self.assertAlmostEqual(scores[0].item(), 1.0)
        self.assertEqual(scores[1].item(), 0.0)
        self.assertEqual(scores[2].item(), 0.0)

    def test_partial_overlap_is_symmetric_and_shrinks_with_distance(self):
        selector = _selector([_pose(), _pose((0.5, 0.0, 0.0)), _pose((2.0, 0.0, 0.0))])
        near = selector.overlap_scores([0], [1]).item()
        far = selector.overlap_scores([0], [2]).item()
        self.assertAlmostEqual(near, selector.overlap_scores([1], [0]).item())
        self.assertTrue(0.0 < far < near < 1.0)

    def test_select_keeps_sinks_recent_and_best_candidates(self):
        # Frames 0..19 look at the origin, 20..47 are far away, 48.. are back home.
        poses = [_pose() for _ in range(20)]
        poses += [_pose((100.0, 0.0, 0.0)) for _ in range(28)]
        poses += [_pose() for _ in range(52)]
        selector = _selector(poses)
        selected = selector.select(80, 82)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertEqual(len(selected), 8 + 32 + 32)
        self.assertEqual(selected[:8], list(range(8)))
        self.assertEqual(selected[-32:], list(range(48, 80)))
        retrieved = [index for index in selected if 8 <= index < 48]
        # Every candidate that saw the origin wins; ties among the far frames go to the newest.
        self.assertTrue(set(range(8, 20)).issubset(retrieved))
        self.assertEqual(
            [index for index in retrieved if index >= 20], list(range(28, 48))
        )

    def test_ties_prefer_newer_frames_and_short_histories_use_everything(self):
        selector = _selector(
            [_pose() for _ in range(100)], sink_size=2, retrieved_size=3, recent_size=4
        )
        self.assertEqual(selector.select(50), [0, 1, 43, 44, 45, 46, 47, 48, 49])
        self.assertEqual(selector.select(5), [0, 1, 2, 3, 4])
        self.assertEqual(selector.select(0), [])
        zero = _selector(
            [_pose() for _ in range(20)], sink_size=0, retrieved_size=0, recent_size=0
        )
        self.assertEqual(zero.select(10), [])
        self.assertEqual(selector.partition(50), (2, 46))

    def test_multi_frame_query_averages_over_the_chunk(self):
        poses = (
            [_pose() for _ in range(8)]
            + [_pose((0.6, 0.0, 0.0))]
            + [_pose((100.0, 0.0, 0.0))] * 3
        )
        poses += [_pose(), _pose((0.6, 0.0, 0.0))]
        selector = _selector(poses, sink_size=0, retrieved_size=1, recent_size=0)
        single = selector.overlap_scores([12], [8]).item()
        pair = selector.overlap_scores([12, 13], [8]).item()
        self.assertAlmostEqual(
            pair, (single + selector.overlap_scores([13], [8]).item()) / 2
        )
        self.assertEqual(selector.select(12, 14), [8])
        with self.assertRaises(ValueError):
            selector.select(12, 12)
        with self.assertRaises(ValueError):
            selector.select(14)

    def test_from_rgb_geometry_maps_latent_frames_and_fov(self):
        poses = torch.stack([_pose((0.1 * index, 0.0, 0.0)) for index in range(13)])
        geometry = {
            "c2w": poses.tolist(),
            "image_size": [640, 360],
            "fps": 30,
            "horizontal_fov_degrees": 90.0,
            "near": 0.1,
            "far": 10.0,
        }
        selector = FrustumHistorySelector.from_rgb_geometry(
            geometry, num_latent_frames=4, temporal_compression_factor=4, fps=30.0
        )
        self.assertEqual(selector.num_frames, 4)
        torch.testing.assert_close(selector._poses, poses[[0, 4, 8, 12]])
        torch.testing.assert_close(selector._intrinsics[0], INTRINSICS)
        self.assertEqual(selector.sizes, (8, 32, 32))
        with self.assertRaisesRegex(ValueError, "shorter"):
            FrustumHistorySelector.from_rgb_geometry(
                geometry, num_latent_frames=5, temporal_compression_factor=4, fps=30.0
            )
        with self.assertRaisesRegex(ValueError, "FPS"):
            FrustumHistorySelector.from_rgb_geometry(
                geometry, num_latent_frames=4, temporal_compression_factor=4, fps=24.0
            )
        with self.assertRaisesRegex(ValueError, "not both"):
            FrustumHistorySelector.from_rgb_geometry(
                dict(geometry, intrinsics=INTRINSICS.tolist()),
                num_latent_frames=4,
                temporal_compression_factor=4,
                fps=30.0,
            )
        with self.assertRaisesRegex(ValueError, "missing fields"):
            FrustumHistorySelector.from_rgb_geometry(
                {"c2w": poses.tolist()},
                num_latent_frames=4,
                temporal_compression_factor=4,
                fps=30.0,
            )

    def test_invalid_geometry(self):
        poses = torch.stack([_pose(), _pose()])
        bad_rotation = poses.clone()
        bad_rotation[1, 0, 0] = 2.0
        bad_row = poses.clone()
        bad_row[1, 3, 0] = 1.0
        skewed = INTRINSICS.clone()
        skewed[1, 0] = 1.0
        cases = [
            dict(c2w=bad_rotation),
            dict(c2w=bad_row),
            dict(intrinsics=skewed),
            dict(intrinsics=-INTRINSICS),
            dict(image_size=(640.0, 360)),
            dict(near=1.0, far=0.5),
            dict(far=math.inf),
            dict(sink_size=-1),
        ]
        for overrides in cases:
            kwargs = dict(
                c2w=poses, intrinsics=INTRINSICS, image_size=IMAGE, near=0.1, far=10.0
            )
            kwargs.update(overrides)
            with self.subTest(overrides=list(overrides)), self.assertRaises(ValueError):
                FrustumHistorySelector(
                    kwargs.pop("c2w"),
                    kwargs.pop("intrinsics"),
                    kwargs.pop("image_size"),
                    kwargs.pop("near"),
                    kwargs.pop("far"),
                    **kwargs,
                )


def _encode_chunk_anchored(poses: torch.Tensor) -> torch.Tensor:
    """Reference encoding: row i is ``inv(T_anchor) @ T_{i+1}`` with the anchor at
    the start of each 16-frame chunk; rotation as the first two matrix columns."""
    rows = []
    for index in range(poses.shape[0] - 1):
        anchor = poses[POSE_CHUNK_FRAMES * (index // POSE_CHUNK_FRAMES)]
        delta = torch.linalg.inv(anchor) @ poses[index + 1]
        rows.append(torch.cat([delta[:3, 3], delta[:3, 0], delta[:3, 1]]))
    return torch.stack(rows)


class TestPoseRecovery(unittest.TestCase):
    def test_round_trip_through_normalized_unified_rows(self):
        generator = torch.Generator().manual_seed(0)
        poses = [torch.eye(4, dtype=torch.float64)]
        for index in range(1, 70):
            yaw = 2.0 * math.sin(index / 7.0)
            step = _pose((0.03 * math.cos(index / 9.0), 0.01, 0.05), yaw)
            poses.append(poses[-1] @ step)
        poses = torch.stack(poses)
        offset = torch.linspace(-0.2, 0.3, 59, dtype=torch.float64)
        scale = torch.linspace(0.5, 1.5, 59, dtype=torch.float64)
        raw = torch.zeros(69, 59, dtype=torch.float64)
        raw[:, :9] = _encode_chunk_anchored(poses)
        raw[:, 33] = torch.rand(69, generator=generator, dtype=torch.float64)
        normalized = torch.asinh((raw - offset) / scale) / math.asinh(1.0)
        recovered = recover_unified_camera_poses(
            normalized.to(torch.float32), offset=offset.tolist(), scale=scale.tolist()
        )
        self.assertEqual(recovered.shape, (70, 4, 4))
        torch.testing.assert_close(recovered, poses, atol=1e-4, rtol=0)
        torch.testing.assert_close(
            denormalize_unified_slots(
                normalized[:, :9], offset=offset[:9], scale=scale[:9]
            ),
            raw[:, :9],
        )

    def test_rot6d_projection_and_anchoring(self):
        rotation = _pose(yaw_degrees=30.0)[:3, :3]
        noisy = torch.cat(
            [rotation[:, 0] * 1.1, rotation[:, 1] + 0.05 * rotation[:, 0]]
        )
        projected = rot6d_to_matrix(noisy)
        torch.testing.assert_close(
            projected @ projected.T, torch.eye(3, dtype=torch.float64)
        )
        self.assertAlmostEqual(torch.linalg.det(projected).item(), 1.0)
        step = _pose((0.0, 0.0, 1.0))
        deltas = step.repeat(POSE_CHUNK_FRAMES + 1, 1, 1)
        absolute = chunk_anchored_poses(deltas)
        # Within the first chunk every row is relative to the identity anchor...
        torch.testing.assert_close(
            absolute[1 : POSE_CHUNK_FRAMES + 1, 2, 3],
            torch.ones(POSE_CHUNK_FRAMES, dtype=torch.float64),
        )
        # ...and the second chunk re-anchors on the last pose of the first.
        self.assertAlmostEqual(absolute[POSE_CHUNK_FRAMES + 1, 2, 3].item(), 2.0)


class TestSelectedHistory(unittest.TestCase):
    def test_gathers_selected_frames_per_layer(self):
        tokens = 3
        frames = torch.arange(6, dtype=torch.float32).repeat_interleave(tokens)
        archive = [
            (frames.view(1, -1, 1, 1) + layer * 100, -frames.view(1, -1, 1, 1))
            for layer in range(2)
        ]
        view = SelectedHistory(archive, frame_ids=[0, 3, 5], tokens_per_frame=tokens)
        self.assertEqual(len(view), 2)
        self.assertEqual(view.frame_ids, (0, 3, 5))
        key, value = view[1]
        self.assertEqual(
            key[0, :, 0, 0].tolist(), [100, 100, 100, 103, 103, 103, 105, 105, 105]
        )
        self.assertEqual(value[0, ::tokens, 0, 0].tolist(), [0, -3, -5])
        empty = SelectedHistory(archive, frame_ids=[], tokens_per_frame=tokens)
        self.assertEqual(empty[0][0].shape[1], 0)
        for bad in ([3, 0], [0, 0], [6], [-1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SelectedHistory(archive, frame_ids=bad, tokens_per_frame=tokens)

    def test_describe_selection(self):
        selector = _selector(
            [_pose() for _ in range(100)], sink_size=2, retrieved_size=3, recent_size=4
        )
        selected = selector.select(50)
        self.assertEqual(
            describe_selection(selector, 50, selected),
            "9 frames: 2 sinks, 4 recent, retrieved [43, 44, 45]",
        )


class TestResolveFrustumSelector(unittest.TestCase):
    GEOMETRY = {
        "pose_source": "agibot_unified_v1_chunk16",
        "image_size": [640, 360],
        "horizontal_fov_degrees": 90.0,
        "near": 0.1,
        "far": 10.0,
    }

    def test_pose_source_recovers_poses_from_rows_and_defaults_fps(self):
        rows = torch.zeros(12, 59)
        rows[:, 3] = rows[:, 7] = 1.0  # identity rotation under unit statistics
        stats = ([0.0] * 59, [1.0] * 59)
        selector = resolve_frustum_selector(
            self.GEOMETRY,
            rows=rows,
            normalizer_stats=stats,
            num_latent_frames=4,
            temporal_compression_factor=4,
            fps=30.0,
        )
        self.assertEqual(selector.num_frames, 4)
        self.assertEqual(selector.select(3, 4), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "FPS"):
            resolve_frustum_selector(
                dict(self.GEOMETRY, fps=24),
                rows=rows,
                normalizer_stats=stats,
                num_latent_frames=4,
                temporal_compression_factor=4,
                fps=30.0,
            )

    def test_rejections(self):
        rows = torch.zeros(12, 59)
        stats = ([0.0] * 59, [1.0] * 59)
        common = dict(num_latent_frames=4, temporal_compression_factor=4, fps=30.0)
        with self.assertRaisesRegex(ValueError, "action rows"):
            resolve_frustum_selector(
                self.GEOMETRY, rows=None, normalizer_stats=stats, **common
            )
        with self.assertRaisesRegex(ValueError, "unified"):
            resolve_frustum_selector(
                self.GEOMETRY, rows=rows, normalizer_stats=None, **common
            )
        with self.assertRaisesRegex(ValueError, "not both"):
            resolve_frustum_selector(
                dict(self.GEOMETRY, c2w=torch.eye(4).repeat(13, 1, 1).tolist()),
                rows=rows,
                normalizer_stats=stats,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "pose_source"):
            resolve_frustum_selector(
                dict(self.GEOMETRY, pose_source="odometry"),
                rows=rows,
                normalizer_stats=stats,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "JSON object"):
            resolve_frustum_selector(
                [1, 2], rows=rows, normalizer_stats=stats, **common
            )


if __name__ == "__main__":
    unittest.main()
