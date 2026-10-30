# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Cosmos3-Nano-Sim-Auto port: artifact validation, the
chunk schedule and packed layout, the replay visibility predicate, the
FlexAttention block mask against a dense oracle, the fixed-step sampler, the
KV ring, request validation and registry wiring."""

import copy
import json
import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.models.dits.cosmos3_sim_auto import (
    COSMOS3_SIM_AUTO_ARTIFACT_KEY,
    load_cosmos3_sim_auto_manifest,
    multiview_deployment_from_manifest,
    parse_cosmos3_sim_auto_manifest,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MADS_CAMERAS,
    MULTIVIEW_BACKEND_ENV_VAR,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_sim_auto import (
    Cosmos3SimAutoConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.cosmos3_sim_auto import (
    Cosmos3SimAutoSamplingParams,
)
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
)
from sglang.multimodal_gen.runtime.models.dits import cosmos3_sim_auto
from sglang.multimodal_gen.runtime.models.dits.cosmos3_sim_auto import (
    ChunkShard,
    UlyssesGroup,
    local_head_group,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_attention import (
    fa4_sparse_available,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_attention import (
    ReplayBlockMask,
    build_replay_block_mask,
    build_replay_fa4_mask,
    build_replay_mask,
    build_replay_plan,
    dense_token_mask,
    replay_attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_layout import (
    CAPTION_SCOPE_ALL,
    CAPTION_SCOPE_SAME_VIEW,
    PASS_CLEAN,
    PASS_NOISY,
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    SensorGrid,
    StreamItem,
    build_caption_documents,
    build_chunk_items,
    build_chunk_schedule,
    caption_metadata,
    chunk_token_count,
    chunk_token_metadata,
    committed_metadata,
    latent_frames_from_pixel_frames,
    lidar_num_sweeps,
    rgb_sensor_grid,
    ring_slot_for_step,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.sim_auto import (
    ReplayRing,
    _FlatLayout,
    camera_major_slice,
    camera_major_write,
    fixed_step_sample,
    initial_noise,
    sde_step_seed,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import RecordOps

EXPORT_DIR = "/mnt/kdub_disk/models/Cosmos3-Nano-Sim-Auto"

LIDAR_BLOCK = {
    "dtype": "float32",
    "sample_posterior": False,
    "apply_validity_mask": True,
    "fps": 10.0,
    "latent_channels": 128,
    "temporal_compression_factor": 1,
    "spatial_compression": [16, 16],
    "streaming_chunk_frames": 20,
    "streaming_context_frames": 21,
    "network_config": {
        "resolution": [128, 1808],
        "patch_size": [2, 2],
        "depths": [3, 3, 3, 3],
        "z_dim": 128,
        "in_channels": 3,
        "temporal_downsample": [False, False, False],
    },
    "range_projection": {
        "semantic_width": 1800,
        "model_width": 1808,
        "native_height": 128,
        "model_width_transform": "circular_pad",
        "intensity_encoding": "unit",
        "min_range_m": 5.0,
        "max_range_m": 100.0,
        "validity_threshold": 0.5,
    },
}


def _artifact(**overrides):
    cameras = list(COSMOS3_MADS_CAMERAS)
    artifact = {
        "schema_version": 1,
        "checkpoint_id": "sim_auto_joint_stage3_sfdmd_mads_all@iter_000001800",
        "checkpoint_iteration": 1800,
        "checkpoint_hash": "deadbeef",
        "attention_mode": "multiview",
        "video_temporal_causal": True,
        "cameras": cameras,
        "rgb": {
            "latent_channels": 48,
            "latent_patch_size": 2,
            "vae_spatial_compression_factor": 16,
            "temporal_compression_factor": 4,
        },
        "lidar": copy.deepcopy(LIDAR_BLOCK),
        "lidar_latent_patch_size_hw": [1, 1],
        "chunk": {"frames_per_chunk": 3},
        "ring": {"cache_chunks": 2, "sink_chunks": 1},
        "replay": {
            "control_visibility": "causal",
            "controls_read_strict_past_clean_rgb": True,
            "clean_pass_causality": "chunk",
            "attention_scope": "decomposed",
            "cross_view_past_window_seconds": 0.4,
            "deduplicate_cross_view": True,
            "lidar_attends_captions": True,
        },
        "prompt": {
            "policy": "segmented",
            "text_sink_tokens": 4,
            "text_position_mode": "fixed",
            "context_before": 0,
            "context_after": 0,
            "media_text_extent": None,
            "media_rope_scale": 1.0,
            "sink_age_cap_seconds": None,
            "use_first_chunk_only": True,
            "system_prompt_rgb": "av_multiview_transfer",
            "system_prompt_joint": "av_joint_camera_lidar_transfer",
            "fractional_duration": False,
            "emphasize_control_in_prompt": False,
            "text_cache_max_len": 16384,
        },
        "mrope": {
            "temporal_modality_margin": 15000,
            "reset_spatial_ids": True,
            "base_fps": 30.0,
            "enable_fps_modulation": True,
        },
        "fixed_step_sampler_config": {
            "num_train_timesteps": 1000,
            "sample_type": "sde",
            "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
        },
        "inference_defaults": {
            "resolution": "480",
            "fps": 30.0,
            "guidance": 1.0,
            "control_guidance": 1.0,
        },
        "rig_view_embedding": {
            "num_embeddings": 12,
            "camera_ids": {camera: index for index, camera in enumerate(cameras)},
            "lidar_id": 11,
        },
    }
    for key, value in overrides.items():
        artifact[key] = value
    return artifact


def _transformer_config(artifact=None):
    return {
        "backbone_type": "cosmos3_multiview",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        COSMOS3_SIM_AUTO_ARTIFACT_KEY: _artifact() if artifact is None else artifact,
    }


class TestManifest(unittest.TestCase):
    def test_parses_synthetic_artifact(self):
        manifest = parse_cosmos3_sim_auto_manifest(_artifact())
        self.assertTrue(manifest.joint)
        self.assertEqual(manifest.lidar_fps, 10.0)
        self.assertAlmostEqual(manifest.lidar_seconds_per_frame, 0.1)
        self.assertAlmostEqual(manifest.rgb_seconds_per_frame(30.0), 4 / 30)
        self.assertEqual(manifest.t_list, (1.0, 0.9375, 0.8333333333333334, 0.625))
        self.assertEqual(
            manifest.rig_camera_ids(
                ["camera_rear_tele_30fov", "camera_front_wide_120fov"]
            ),
            [3, 0],
        )
        self.assertEqual(
            manifest.system_prompt_id(joint=True), "av_joint_camera_lidar_transfer"
        )
        self.assertEqual(manifest.lidar_latent_patch_size_hw, (1, 1))
        self.assertIn("ring 2 chunks (1 sink)", manifest.describe())

    def test_rgb_only_artifact(self):
        manifest = parse_cosmos3_sim_auto_manifest(
            _artifact(lidar=None, lidar_latent_patch_size_hw=None)
        )
        self.assertFalse(manifest.joint)
        self.assertIsNone(manifest.lidar_seconds_per_frame)
        with self.assertRaisesRegex(ValueError, "requires a lidar block"):
            parse_cosmos3_sim_auto_manifest(_artifact(lidar=None))

    def test_rejects_field_set_mismatch_and_policy_drift(self):
        extra = _artifact()
        extra["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "unknown=\\['surprise'\\]"):
            parse_cosmos3_sim_auto_manifest(extra)
        missing = _artifact()
        del missing["ring"]
        with self.assertRaisesRegex(ValueError, "missing=\\['ring'\\]"):
            parse_cosmos3_sim_auto_manifest(missing)
        drift = _artifact()
        drift["replay"]["clean_pass_causality"] = "frame"
        with self.assertRaisesRegex(ValueError, "replay.clean_pass_causality"):
            parse_cosmos3_sim_auto_manifest(drift)
        prompt = _artifact()
        prompt["prompt"]["use_first_chunk_only"] = False
        with self.assertRaisesRegex(ValueError, "prompt.use_first_chunk_only"):
            parse_cosmos3_sim_auto_manifest(prompt)
        with self.assertRaisesRegex(ValueError, "schema_version=2"):
            parse_cosmos3_sim_auto_manifest(_artifact(schema_version=2))
        with self.assertRaisesRegex(ValueError, "rig_view_embedding"):
            parse_cosmos3_sim_auto_manifest(_artifact(rig_view_embedding=None))
        with self.assertRaisesRegex(ValueError, "backbone_type"):
            load_cosmos3_sim_auto_manifest(
                {"backbone_type": "cosmos3", COSMOS3_SIM_AUTO_ARTIFACT_KEY: _artifact()}
            )
        with self.assertRaisesRegex(ValueError, "requires transformer/config.json"):
            load_cosmos3_sim_auto_manifest({"backbone_type": "cosmos3_multiview"})

    def test_multiview_deployment_follows_the_student(self):
        deployment = multiview_deployment_from_manifest(
            parse_cosmos3_sim_auto_manifest(_artifact())
        )
        self.assertEqual(deployment.num_views, 11)
        self.assertEqual(deployment.inference_default("num_steps", None), 4)
        self.assertEqual(deployment.inference_default("guidance", None), 1.0)
        self.assertFalse(
            deployment.inference_default("emphasize_control_in_prompt", True)
        )
        self.assertEqual(deployment.inference_default("fps", None), 30.0)
        self.assertTrue(deployment.supports_lidar)
        self.assertEqual(deployment.rig_view_ids(["camera_cross_left_120fov"]), [5])

    @unittest.skipUnless(os.path.isdir(EXPORT_DIR), "Sim-Auto export not mounted")
    def test_real_export(self):
        with open(os.path.join(EXPORT_DIR, "transformer", "config.json")) as handle:
            manifest = load_cosmos3_sim_auto_manifest(json.load(handle))
        self.assertEqual(
            manifest.checkpoint_id,
            "sim_auto_joint_stage3_sfdmd_mads_all@iter_000001800",
        )
        self.assertEqual(len(manifest.cameras), 11)
        self.assertEqual(
            (manifest.frames_per_chunk, manifest.cache_chunks, manifest.sink_chunks),
            (3, 2, 1),
        )
        self.assertEqual(manifest.text_sink_tokens, 4)
        self.assertEqual(manifest.rig_view_embedding["lidar_id"], 11)
        self.assertTrue(manifest.joint)


class TestSchedule(unittest.TestCase):
    def test_reference_horizon(self):
        latents = latent_frames_from_pixel_frames(297, 4)
        sweeps = lidar_num_sweeps(297, camera_fps=30.0, lidar_fps=10.0)
        self.assertEqual((latents, sweeps), (75, 99))
        schedule = build_chunk_schedule(
            rgb_latents=latents,
            lidar_sweeps=sweeps,
            frames_per_chunk=3,
            rgb_seconds_per_frame=4 / 30,
            lidar_seconds_per_frame=0.1,
        )
        self.assertEqual(len(schedule), 26)
        self.assertEqual(
            (schedule[0].rgb_frames, schedule[0].lidar_sweeps), ((0,), (0,))
        )
        self.assertEqual(
            (schedule[1].rgb_frames, schedule[1].lidar_sweeps),
            ((1, 2, 3), (1, 2, 3, 4)),
        )
        self.assertEqual(
            (schedule[-1].rgb_frames, schedule[-1].lidar_sweeps), ((73, 74), (97, 98))
        )
        self.assertEqual(
            [f for c in schedule for f in c.rgb_frames], list(range(latents))
        )
        self.assertEqual(
            [s for c in schedule for s in c.lidar_sweeps], list(range(sweeps))
        )
        self.assertEqual([c.step for c in schedule], list(range(26)))

    def test_rgb_only_and_invalid_inputs(self):
        schedule = build_chunk_schedule(
            rgb_latents=5,
            lidar_sweeps=None,
            frames_per_chunk=3,
            rgb_seconds_per_frame=4 / 30,
        )
        self.assertEqual([c.rgb_frames for c in schedule], [(0,), (1, 2, 3), (4,)])
        self.assertTrue(all(c.lidar_sweeps == () for c in schedule))
        with self.assertRaisesRegex(ValueError, "must contain both"):
            build_chunk_schedule(
                rgb_latents=5,
                lidar_sweeps=1,
                frames_per_chunk=3,
                rgb_seconds_per_frame=4 / 30,
                lidar_seconds_per_frame=0.1,
            )
        with self.assertRaisesRegex(ValueError, "4k \\+ 1"):
            latent_frames_from_pixel_frames(16, 4)


def _grid():
    return SensorGrid(1, 1, 1, 1)


def _items(chunk, num_views=2, joint=False, condition=True):
    return build_chunk_items(
        chunk,
        num_views=num_views,
        rgb_grid=_grid(),
        rgb_seconds_per_frame=4 / 30,
        lidar_grid=_grid() if joint else None,
        lidar_seconds_per_frame=0.1 if joint else None,
        rgb_condition_frames=(0,) if condition else (),
        lidar_condition_sweeps=(),
    )


def _schedule(joint=False):
    return build_chunk_schedule(
        rgb_latents=7,
        lidar_sweeps=9 if joint else None,
        frames_per_chunk=3,
        rgb_seconds_per_frame=4 / 30,
        lidar_seconds_per_frame=0.1 if joint else None,
    )


class TestLayout(unittest.TestCase):
    def test_items_and_tokens(self):
        chunk = _schedule(joint=True)[1]
        items = _items(chunk, joint=True)
        self.assertEqual(
            [item.kind for item in items],
            ["rgb_control", "rgb_target", "lidar_control", "lidar_target"],
        )
        self.assertEqual(items[0].view_ids, (0, 1))
        self.assertEqual(items[2].view_ids, (2,))
        self.assertEqual(items[1].condition_frames, ())
        self.assertEqual(_items(_schedule()[0])[1].condition_frames, (0,))
        self.assertEqual(items[3].caption_scope, CAPTION_SCOPE_ALL)
        self.assertEqual(items[1].caption_scope, CAPTION_SCOPE_SAME_VIEW)
        self.assertEqual(chunk_token_count(items), 2 * 2 * 3 + 2 * 4)
        grid = rgb_sensor_grid(
            height=480, width=832, vae_spatial_compression=16, patch_size=2
        )
        self.assertEqual(
            (grid.grid_height, grid.grid_width, grid.tokens_per_frame), (15, 26, 390)
        )
        with self.assertRaisesRegex(ValueError, "fully observed"):
            StreamItem("rgb_control", (0,), (0,), _grid(), 0.1, (0,))

    def test_metadata_roles_and_commit(self):
        items = _items(_schedule()[0])
        noisy = chunk_token_metadata(items, step=0, pass_kind=PASS_NOISY)
        self.assertEqual(noisy.num_tokens, 4)
        self.assertEqual(
            noisy.role.tolist(),
            [ROLE_CONTROL, ROLE_CONTROL, ROLE_TARGET_CONDITION, ROLE_TARGET_CONDITION],
        )
        items = _items(_schedule()[1])
        noisy = chunk_token_metadata(items, step=1, pass_kind=PASS_NOISY)
        self.assertEqual(noisy.view_id.tolist(), [0, 0, 0, 1, 1, 1] * 2)
        self.assertEqual(noisy.frame_id.tolist(), [1, 2, 3, 1, 2, 3] * 2)
        self.assertEqual(noisy.role[6:].tolist(), [ROLE_CURRENT_TARGET] * 6)
        self.assertEqual(noisy.step.tolist(), [1] * 12)
        torch.testing.assert_close(
            noisy.timestamp[:3], torch.tensor([1, 2, 3], dtype=torch.float32) * (4 / 30)
        )
        clean = chunk_token_metadata(items, step=1, pass_kind=PASS_CLEAN)
        self.assertEqual(clean.role[6:].tolist(), [ROLE_CLEAN_TARGET] * 6)
        committed = committed_metadata(noisy)
        self.assertEqual(committed.role.tolist(), clean.role.tolist())

    def test_caption_documents(self):
        documents = build_caption_documents(
            lengths=(5, 7), sink_tokens=2, horizon_seconds=1.0
        )
        self.assertEqual(
            [(d.view_id, d.num_tokens, d.is_sink) for d in documents],
            [(0, 2, True), (0, 5, False), (1, 2, True), (1, 7, False)],
        )
        captions = caption_metadata(documents)
        self.assertEqual(captions.num_tokens, 16)
        self.assertEqual(captions.doc_offsets, (0, 2, 7, 9))
        self.assertEqual(captions.is_sink.tolist()[:3], [True, True, False])
        self.assertEqual(captions.end_seconds[2].item(), 1.0)
        self.assertEqual(captions.end_seconds[0].item(), math.inf)

    def test_ring_slots(self):
        self.assertEqual(
            [ring_slot_for_step(s, cache_chunks=2, sink_chunks=1) for s in range(5)],
            [0, 1, 1, 1, 1],
        )
        self.assertEqual(
            [ring_slot_for_step(s, cache_chunks=3, sink_chunks=1) for s in range(5)],
            [0, 1, 2, 1, 2],
        )


def _committed(chunk, step, **kwargs):
    return committed_metadata(
        chunk_token_metadata(_items(chunk, **kwargs), step=step, pass_kind=PASS_NOISY)
    )


def _plan(
    step,
    pass_kind,
    *,
    joint=False,
    lengths=(5, 6),
    sink=2,
    horizon=25 / 30,
    condition=True,
    num_views=2,
):
    """Plan of chunk ``step`` after committing every earlier chunk into a ring of two slots."""
    schedule = _schedule(joint=joint)
    items = _items(
        schedule[step], joint=joint, condition=condition, num_views=num_views
    )
    frame_unit = max(
        chunk_token_count(
            _items(c, joint=joint, condition=condition, num_views=num_views)
        )
        for c in schedule
    )
    history = []
    slots = {}
    for earlier in schedule[:step]:
        slots[ring_slot_for_step(earlier.step, cache_chunks=2, sink_chunks=1)] = (
            _committed(
                earlier,
                earlier.step,
                joint=joint,
                condition=condition,
                num_views=num_views,
            )
        )
    history = [(slot * frame_unit, slots[slot]) for slot in sorted(slots)]
    captions = caption_metadata(
        build_caption_documents(
            lengths=lengths, sink_tokens=sink, horizon_seconds=horizon
        )
    )
    current = chunk_token_metadata(items, step=step, pass_kind=pass_kind)
    plan = build_replay_plan(
        current=current,
        history=history,
        history_tokens=2 * frame_unit,
        captions=captions,
        window_seconds=0.4,
    )
    return plan, items, frame_unit, slots


def _token(items, kind, view, frame):
    """Packed index of (item kind, view, frame) in a chunk with one token per frame."""
    offset = 0
    for item in items:
        if item.kind == kind:
            return (
                offset
                + item.view_ids.index(view) * len(item.frames)
                + item.frames.index(frame)
            )
        offset += item.num_tokens
    raise KeyError(kind)


class TestPredicate(unittest.TestCase):
    """Token-level checks of the replay predicate on a two-camera rig (one token per frame).

    Chunk 2 (frames 4, 5, 6 at 0.533, 0.667, 0.8 s) reads the ring holding chunk 0
    (frame 0, sink slot) and chunk 1 (frames 1, 2, 3 at 0.133, 0.267, 0.4 s).
    """

    def setUp(self):
        self.plan, self.items, self.frame_unit, self.slots = _plan(2, PASS_NOISY)
        self.mask = dense_token_mask(self.plan)
        self.current = self.plan.num_current_tokens
        chunk0 = _items(_schedule()[0])
        chunk1 = _items(_schedule()[1])
        self.history_index = {}
        for slot, items in ((0, chunk0), (1, chunk1)):
            for item in items:
                for view in item.view_ids:
                    for frame in item.frames:
                        self.history_index[(item.kind, view, frame)] = (
                            self.current
                            + slot * self.frame_unit
                            + _token(items, item.kind, view, frame)
                        )
        self.caption_base = self.current + 2 * self.frame_unit

    def q(self, kind, view, frame):
        return _token(self.items, kind, view, frame)

    def k(self, kind, view, frame):
        return _token(self.items, kind, view, frame)

    def h(self, kind, view, frame):
        return self.history_index[(kind, view, frame)]

    def test_targets_read_own_controls_and_same_step_targets_in_window(self):
        q = self.q("rgb_target", 0, 4)
        self.assertTrue(
            all(self.mask[q, self.k("rgb_control", 0, f)] for f in (4, 5, 6))
        )
        self.assertFalse(
            any(self.mask[q, self.k("rgb_control", 1, f)] for f in (4, 5, 6))
        )
        self.assertTrue(self.mask[q, self.k("rgb_target", 1, 4)])
        self.assertFalse(self.mask[q, self.k("rgb_target", 1, 6)])
        self.assertTrue(
            self.mask[self.q("rgb_target", 0, 6), self.k("rgb_target", 1, 4)]
        )
        self.assertTrue(
            self.mask[self.q("rgb_target", 0, 6), self.k("rgb_target", 0, 4)]
        )

    def test_targets_read_clean_history_same_view_always_and_cross_view_in_window(self):
        q4 = self.q("rgb_target", 0, 4)
        self.assertTrue(
            all(self.mask[q4, self.h("rgb_target", 0, f)] for f in (0, 1, 2, 3))
        )
        self.assertTrue(
            all(self.mask[q4, self.h("rgb_target", 1, f)] for f in (1, 2, 3))
        )
        self.assertFalse(self.mask[q4, self.h("rgb_target", 1, 0)])
        q6 = self.q("rgb_target", 0, 6)
        self.assertFalse(self.mask[q6, self.h("rgb_target", 1, 1)])
        self.assertTrue(self.mask[q6, self.h("rgb_target", 1, 3)])
        self.assertTrue(self.mask[q6, self.h("rgb_target", 0, 0)])
        self.assertTrue(
            all(self.mask[q4, self.h("rgb_control", 0, f)] for f in (0, 1, 2, 3))
        )
        self.assertFalse(
            any(self.mask[q4, self.h("rgb_control", 1, f)] for f in (0, 1, 2, 3))
        )

    def test_controls_read_same_view_only(self):
        q = self.q("rgb_control", 0, 5)
        self.assertTrue(
            all(self.mask[q, self.h("rgb_target", 0, f)] for f in (0, 1, 2, 3))
        )
        self.assertFalse(
            any(self.mask[q, self.h("rgb_target", 1, f)] for f in (0, 1, 2, 3))
        )
        self.assertTrue(
            all(self.mask[q, self.k("rgb_control", 0, f)] for f in (4, 5, 6))
        )
        self.assertFalse(
            any(self.mask[q, self.k("rgb_control", 1, f)] for f in (4, 5, 6))
        )
        self.assertFalse(
            any(
                self.mask[q, self.k("rgb_target", v, f)]
                for v in (0, 1)
                for f in (4, 5, 6)
            )
        )

    def test_captions_follow_the_query_view(self):
        offsets = self.plan.key_runs.start[-4:].tolist()
        sink0, cap0, sink1, cap1 = offsets
        q0 = self.q("rgb_target", 0, 5)
        q1 = self.q("rgb_control", 1, 5)
        for q, own, other in (
            (q0, (sink0, cap0), (sink1, cap1)),
            (q1, (sink1, cap1), (sink0, cap0)),
        ):
            self.assertTrue(all(self.mask[q, start] for start in own))
            self.assertFalse(any(self.mask[q, start] for start in other))
        self.assertEqual(int(self.mask[:, self.caption_base :].sum(1).min()), 2 + 5)

    def test_clean_pass_is_chunk_causal(self):
        plan, items, _, _ = _plan(2, PASS_CLEAN)
        mask = dense_token_mask(plan)
        q4 = _token(items, "rgb_target", 0, 4)
        self.assertTrue(mask[q4, _token(items, "rgb_target", 0, 6)])
        self.assertFalse(mask[q4, _token(items, "rgb_target", 1, 6)])
        self.assertTrue(
            mask[_token(items, "rgb_target", 0, 6), _token(items, "rgb_target", 1, 4)]
        )
        noisy_plan, _, _, _ = _plan(2, PASS_NOISY)
        # The noisy pass of the same chunk reads no same-step clean target: there are none yet.
        self.assertFalse(
            bool(
                (
                    noisy_plan.key_runs.role[: noisy_plan.query_runs.num_runs]
                    == ROLE_CLEAN_TARGET
                ).any()
            )
        )

    def test_conditions_in_chunk_zero(self):
        plan, items, _, _ = _plan(0, PASS_NOISY, joint=True)
        mask = dense_token_mask(plan)
        cond = _token(items, "rgb_target", 0, 0)
        self.assertTrue(mask[cond, cond])
        self.assertTrue(mask[cond, _token(items, "rgb_control", 0, 0)])
        self.assertFalse(mask[cond, _token(items, "rgb_target", 1, 0)])
        self.assertFalse(mask[cond, _token(items, "lidar_target", 2, 0)])
        lidar = _token(items, "lidar_target", 2, 0)
        self.assertTrue(mask[lidar, cond])
        self.assertTrue(mask[lidar, _token(items, "lidar_control", 2, 0)])
        self.assertFalse(mask[lidar, _token(items, "rgb_control", 0, 0)])
        caption_base = plan.num_current_tokens + plan.num_history_tokens
        self.assertTrue(bool(mask[lidar, caption_base:].all()))
        self.assertEqual(int(mask[cond, caption_base:].sum()), 2 + 5)

    def test_joint_lidar_reads_rgb_within_window(self):
        # Chunk 2 sweeps 5..8 sit at 0.5..0.8 s; RGB frames 4, 5, 6 at 0.533, 0.667, 0.8 s.
        plan, items, _, _ = _plan(2, PASS_NOISY, joint=True)
        mask = dense_token_mask(plan)
        sweep7 = _token(items, "lidar_target", 2, 7)
        self.assertTrue(mask[sweep7, _token(items, "rgb_target", 0, 5)])
        self.assertTrue(mask[sweep7, _token(items, "rgb_target", 0, 4)])
        self.assertFalse(mask[sweep7, _token(items, "rgb_target", 0, 6)])
        self.assertTrue(
            mask[_token(items, "lidar_target", 2, 8), _token(items, "rgb_target", 1, 6)]
        )
        self.assertTrue(mask[sweep7, _token(items, "lidar_control", 2, 7)])
        self.assertFalse(mask[sweep7, _token(items, "rgb_control", 0, 5)])
        self.assertFalse(
            mask[_token(items, "rgb_target", 0, 4), _token(items, "lidar_target", 2, 8)]
        )
        self.assertTrue(
            mask[_token(items, "rgb_target", 0, 6), _token(items, "lidar_target", 2, 5)]
        )

    def test_caption_bounds_and_padding(self):
        plan, items, _, _ = _plan(2, PASS_NOISY, horizon=0.6)
        mask = dense_token_mask(plan)
        caption_base = plan.num_current_tokens + plan.num_history_tokens
        late = _token(items, "rgb_target", 0, 6)
        early = _token(items, "rgb_target", 0, 4)
        self.assertEqual(int(mask[late, caption_base:].sum()), 2)
        self.assertEqual(int(mask[early, caption_base:].sum()), 2 + 5)
        padding = plan.key_run_ids < 0
        self.assertTrue(bool(padding.any()))
        self.assertFalse(bool(mask[:, padding].any()))


def _dense_oracle(q, k, v, mask):
    ratio = q.shape[2] // k.shape[2]
    keys = k.repeat_interleave(ratio, dim=2).float()
    values = v.repeat_interleave(ratio, dim=2).float()
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), keys) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), values)


def _expand_fa4_mask(mask):
    """Token mask exactly as the CuTe kernel resolves it: skipped, full and partial tiles.

    Partial tiles read bit ``k_run % 32`` of word ``q_word_base[q] + k_run // 32``.
    """
    sparsity = mask.sparsity
    q_block, kv_block = sparsity.q_block_size, sparsity.kv_block_size
    k_runs = sparsity.k_group_ids.long()
    base = sparsity.q_word_base.long()
    words = sparsity.allowed_words.long() & 0xFFFFFFFF
    table = (
        (words[base[:, None] + k_runs[None, :] // 32] >> (k_runs[None, :] % 32)) & 1
    ).bool()
    dense = torch.zeros(mask.padded_q_len, mask.padded_kv_len, dtype=torch.bool)
    for q_index in range(mask.padded_q_len // q_block):
        rows = slice(q_index * q_block, (q_index + 1) * q_block)
        count = int(sparsity.full_counts[q_index])
        for k_index in sparsity.full_indices[q_index, :count].tolist():
            dense[rows, k_index * kv_block : (k_index + 1) * kv_block] = True
        count = int(sparsity.partial_counts[q_index])
        for k_index in sparsity.partial_indices[q_index, :count].tolist():
            cols = slice(k_index * kv_block, (k_index + 1) * kv_block)
            dense[rows, cols] = table[rows, cols]
    return dense[: mask.q_len, : mask.kv_len]


def _wide_plan(grid=SensorGrid(5, 7, 1, 1), lengths=(30, 40, 50)):
    """Chunk 1 of a three-camera RGB rollout whose runs span several tiles."""
    schedule = _schedule()
    items = build_chunk_items(
        schedule[1], num_views=3, rgb_grid=grid, rgb_seconds_per_frame=4 / 30
    )
    history = [
        (
            0,
            committed_metadata(
                chunk_token_metadata(
                    build_chunk_items(
                        schedule[0],
                        num_views=3,
                        rgb_grid=grid,
                        rgb_seconds_per_frame=4 / 30,
                    ),
                    step=0,
                    pass_kind=PASS_NOISY,
                )
            ),
        )
    ]
    captions = caption_metadata(
        build_caption_documents(lengths=lengths, sink_tokens=4, horizon_seconds=1.0)
    )
    return build_replay_plan(
        current=chunk_token_metadata(items, step=1, pass_kind=PASS_NOISY),
        history=history,
        history_tokens=2 * chunk_token_count(items),
        captions=captions,
        window_seconds=0.4,
    )


class TestBlockMask(unittest.TestCase):
    def _check(self, plan):
        device = torch.device("cpu")
        block = build_replay_block_mask(plan, device=device)
        heads, kv_heads, head_dim = 4, 2, 16
        q = torch.randn(1, plan.num_query_tokens, heads, head_dim)
        k = torch.randn(1, plan.num_key_tokens, kv_heads, head_dim)
        v = torch.randn(1, plan.num_key_tokens, kv_heads, head_dim)
        out = replay_attention(q, k, v, block)
        expected = _dense_oracle(q, k, v, dense_token_mask(plan))
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(block.padded_q_len % 64, 0)
        self.assertEqual(block.padded_kv_len % 64, 0)

    def test_matches_dense_oracle(self):
        torch.manual_seed(0)
        for step, pass_kind, joint in (
            (0, PASS_NOISY, True),
            (1, PASS_NOISY, False),
            (2, PASS_NOISY, True),
            (2, PASS_CLEAN, False),
        ):
            with self.subTest(step=step, pass_kind=pass_kind, joint=joint):
                plan, _, _, _ = _plan(
                    step, pass_kind, joint=joint, lengths=(9, 13), sink=4
                )
                self._check(plan)

    def test_wide_grid_crosses_blocks(self):
        torch.manual_seed(1)
        self._check(_wide_plan())

    def test_fa4_tiles_and_packed_table_resolve_the_predicate(self):
        """Hopper (128x128) and Blackwell (256x128) maps classify tiles and pack bits like the predicate."""
        plans = (
            _plan(2, PASS_NOISY, joint=True, lengths=(9, 13), sink=4)[0],
            _wide_plan(grid=SensorGrid(9, 11, 1, 1), lengths=(130, 70, 90)),
        )
        for block_size in ((256, 128), (128, 128)):
            for index, plan in enumerate(plans):
                with self.subTest(block_size=block_size, plan=index):
                    mask = build_replay_fa4_mask(
                        plan, device=torch.device("cpu"), block_size=block_size
                    )
                    self.assertEqual(mask.padded_q_len % block_size[0], 0)
                    self.assertEqual(mask.padded_kv_len % block_size[1], 0)
                    # FA4 wraps aux reads modulo the padded lengths; shorter tensors alias.
                    sparsity = mask.sparsity
                    self.assertEqual(sparsity.q_word_base.numel(), mask.padded_q_len)
                    self.assertEqual(sparsity.k_group_ids.numel(), mask.padded_kv_len)
                    torch.testing.assert_close(
                        _expand_fa4_mask(mask), dense_token_mask(plan)
                    )
                    if index == 1:
                        # The wide plan must exercise all three tile classes.
                        tiles = sparsity.full_indices.numel()
                        full = int(sparsity.full_counts.sum())
                        partial = int(sparsity.partial_counts.sum())
                        self.assertGreater(full, 0)
                        self.assertGreater(partial, 0)
                        self.assertLess(full + partial, tiles)

    def test_build_replay_mask_dispatch(self):
        plan = _plan(1, PASS_NOISY)[0]
        cpu = torch.device("cpu")
        self.assertIsInstance(
            build_replay_mask(plan, device=cpu, backend="triton"), ReplayBlockMask
        )
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            build_replay_mask(plan, device=cpu, backend="fa4")
        with self.assertRaisesRegex(ValueError, "must be one of"):
            build_replay_mask(plan, device=cpu, backend="flex")

    @unittest.skipUnless(
        torch.cuda.is_available() and fa4_sparse_available(torch.device("cuda")),
        "needs an SM90 or SM100 device with the flash-attn-4 CuTe package",
    )
    def test_cuda_fa4_matches_dense_oracle_and_triton(self):
        torch.manual_seed(2)
        device = torch.device("cuda")
        plan = _wide_plan(grid=SensorGrid(9, 11, 1, 1), lengths=(130, 70, 90))
        heads, kv_heads, head_dim = 4, 2, 128
        shape = (1, plan.num_key_tokens, kv_heads, head_dim)
        q = torch.randn(
            1,
            plan.num_query_tokens,
            heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        k = torch.randn(shape, device=device, dtype=torch.bfloat16)
        v = torch.randn(shape, device=device, dtype=torch.bfloat16)
        fa4 = replay_attention(
            q, k, v, build_replay_mask(plan, device=device, backend="fa4")
        )
        expected = _dense_oracle(q, k, v, dense_token_mask(plan).to(device))
        torch.testing.assert_close(fa4.float(), expected, atol=3e-2, rtol=3e-2)
        triton = replay_attention(
            q, k, v, build_replay_mask(plan, device=device, backend="triton")
        )
        torch.testing.assert_close(fa4.float(), triton.float(), atol=3e-2, rtol=3e-2)


class TestUlyssesShard(unittest.TestCase):
    """Token shards and head groups partition a chunk the way the per-layer exchange assumes."""

    def test_rank_splits_reassemble_the_chunk(self):
        tokens = torch.arange(38 * 3, dtype=torch.float32).view(1, 38, 3)
        positions = torch.arange(38)
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                shards = [
                    UlyssesGroup(rank=rank, world_size=world_size).shard(38)
                    for rank in range(world_size)
                ]
                local = shards[0].local_tokens
                self.assertEqual(local, -(-38 // world_size))
                self.assertEqual(shards[0].padded_tokens, local * world_size)
                hidden_parts = [shard.split(tokens, dim=1) for shard in shards]
                position_parts = [shard.split(positions, dim=0) for shard in shards]
                self.assertTrue(all(part.shape[1] == local for part in hidden_parts))
                torch.testing.assert_close(
                    torch.cat(hidden_parts, dim=1)[:, :38], tokens
                )
                torch.testing.assert_close(torch.cat(position_parts)[:38], positions)
                # The padding repeats the last token, so its RoPE position stays valid.
                tail = torch.cat(position_parts)[38:]
                self.assertTrue(bool((tail == 37).all()))

    def test_head_groups_follow_rank_order(self):
        kv_heads = 8
        for world_size in (2, 4, 8):
            groups = [
                UlyssesGroup(rank=rank, world_size=world_size)
                for rank in range(world_size)
            ]
            covered = [group.head_slice(kv_heads) for group in groups]
            self.assertEqual(covered[0].start, 0)
            self.assertEqual(covered[-1].stop, kv_heads)
            for earlier, later in zip(covered, covered[1:]):
                self.assertEqual(earlier.stop, later.start)
        layer_kv = [(torch.randn(1, 5, kv_heads, 4), torch.randn(1, 5, kv_heads, 4))]
        self.assertIs(local_head_group(layer_kv, None)[0][0], layer_kv[0][0])
        mine = local_head_group(layer_kv, UlyssesGroup(rank=1, world_size=4))
        torch.testing.assert_close(mine[0][1], layer_kv[0][1][:, :, 2:4])
        self.assertTrue(mine[0][0].is_contiguous())

    def test_rejects_bad_geometry(self):
        with self.assertRaisesRegex(ValueError, "Invalid Ulysses rank"):
            UlyssesGroup(rank=2, world_size=2)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            UlyssesGroup(rank=0, world_size=3).head_slice(8)
        with self.assertRaisesRegex(ValueError, "at least one token"):
            ChunkShard(group=UlyssesGroup(rank=0, world_size=2), num_tokens=0)
        shard = UlyssesGroup(rank=0, world_size=2).shard(5)
        with self.assertRaisesRegex(ValueError, "Expected 5 tokens"):
            shard.split(torch.zeros(1, 6, 2), dim=1)


class TestPositions(unittest.TestCase):
    """The chunk's absolute mRoPE ids equal the Cosmos3 helper's FPS-modulated ids per item."""

    def test_rgb_and_lidar_positions_match_the_base_helper(self):
        from sglang.multimodal_gen.runtime.models.dits.cosmos3_sim_auto import (
            Cosmos3SimAutoTransformer,
        )
        from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
            compute_mrope_position_ids_vision,
        )

        fake = SimpleNamespace(
            temporal_margin=15000, base_fps=30.0, temporal_compression_factor=4
        )
        rgb = SensorGrid(30, 52, 2, 2)
        lidar = SensorGrid(8, 113, 1, 1)
        chunk = _schedule(joint=True)[2]
        items = build_chunk_items(
            chunk,
            num_views=2,
            rgb_grid=rgb,
            rgb_seconds_per_frame=4 / 30,
            lidar_grid=lidar,
            lidar_seconds_per_frame=0.1,
        )
        ids = Cosmos3SimAutoTransformer.chunk_position_ids(
            fake, items, text_extent=1115, device=torch.device("cpu")
        )
        self.assertEqual(ids.shape, (3, chunk_token_count(items)))
        offset = 1115 + 15000
        expected_rgb, _ = compute_mrope_position_ids_vision(
            3,
            15,
            26,
            offset,
            torch.device("cpu"),
            fps=30.0,
            base_fps=30.0,
            temporal_compression_factor=4,
            start_frame_offset=chunk.rgb_frames[0],
        )
        expected_lidar, _ = compute_mrope_position_ids_vision(
            4,
            8,
            113,
            offset,
            torch.device("cpu"),
            fps=10.0,
            base_fps=30.0,
            temporal_compression_factor=1,
            base_temporal_compression_factor=4,
            start_frame_offset=chunk.lidar_sweeps[0],
        )
        per_view = 3 * 15 * 26
        control = ids[:, : 2 * per_view]
        torch.testing.assert_close(
            control[:, :per_view], expected_rgb.float(), atol=1e-3, rtol=0
        )
        torch.testing.assert_close(
            control[:, per_view:], expected_rgb.float(), atol=1e-3, rtol=0
        )
        lidar_start = 4 * per_view
        torch.testing.assert_close(
            ids[:, lidar_start : lidar_start + 4 * 904],
            expected_lidar.float(),
            atol=1e-3,
            rtol=0,
        )
        # Temporal ids: RGB latent f at offset + f, LiDAR sweep s at offset + 0.75 s.
        self.assertAlmostEqual(float(ids[0, 0]), offset + 4.0, places=2)
        self.assertAlmostEqual(float(ids[0, lidar_start]), offset + 0.75 * 5, places=2)
        self.assertEqual(float(ids[1, 0]), 0.0)
        self.assertEqual(float(ids[2, 1]), 1.0)


class TestSampler(unittest.TestCase):
    def test_seeds_and_schedule(self):
        self.assertEqual(sde_step_seed(7, 3, 0), 7 + 3 * 1_000_003 + 9_176)
        a = initial_noise((1, 8), seed=11, sample_index=2, device=torch.device("cpu"))
        b = initial_noise((1, 8), seed=13, sample_index=0, device=torch.device("cpu"))
        torch.testing.assert_close(a, b)
        noise = torch.randn(1, 16)
        calls = []

        def velocity(x, timestep):
            calls.append(float(timestep[0, 0]))
            return x

        ode = fixed_step_sample(
            velocity,
            noise,
            t_list=(1.0, 0.5),
            num_train_timesteps=1000,
            seed=0,
            sample_index=0,
            sample_type="ode",
        )
        self.assertEqual(calls, [1000.0, 500.0])
        # Euler with v = x: x1 = x0 + (0.5 - 1) * x0 = 0.5 x0, then x2 = x1 - 0.5 * x1.
        torch.testing.assert_close(ode, noise * 0.25)
        first = fixed_step_sample(
            velocity,
            noise,
            t_list=(1.0, 0.5),
            num_train_timesteps=1000,
            seed=3,
            sample_index=4,
            sample_type="sde",
        )
        again = fixed_step_sample(
            velocity,
            noise,
            t_list=(1.0, 0.5),
            num_train_timesteps=1000,
            seed=3,
            sample_index=4,
            sample_type="sde",
        )
        other = fixed_step_sample(
            velocity,
            noise,
            t_list=(1.0, 0.5),
            num_train_timesteps=1000,
            seed=3,
            sample_index=5,
            sample_type="sde",
        )
        torch.testing.assert_close(first, again)
        self.assertFalse(torch.allclose(first, other))
        with self.assertRaisesRegex(ValueError, "flat"):
            fixed_step_sample(
                velocity,
                noise.view(1, 1, 16),
                t_list=(1.0,),
                num_train_timesteps=1000,
                seed=0,
                sample_index=0,
                sample_type="sde",
            )

    def test_camera_major_helpers(self):
        latents = torch.arange(2 * 4).float().view(1, 1, 8, 1, 1)
        picked = camera_major_slice(
            latents, num_views=2, frames_per_view=4, frames=torch.tensor([1, 3])
        )
        self.assertEqual(picked.flatten().tolist(), [1, 3, 5, 7])
        camera_major_write(
            latents,
            torch.zeros(1, 1, 4, 1, 1),
            num_views=2,
            frames_per_view=4,
            frames=torch.tensor([1, 3]),
        )
        self.assertEqual(latents.flatten().tolist(), [0, 0, 2, 0, 4, 0, 6, 0])


class TestRing(unittest.TestCase):
    def test_publish_and_history(self):
        ring = ReplayRing(
            num_layers=2,
            cache_chunks=2,
            frame_unit=6,
            num_kv_heads=1,
            head_dim=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        schedule = _schedule()
        meta0 = _committed(schedule[0], 0)
        kv = [(torch.ones(1, 4, 1, 2), torch.full((1, 4, 1, 2), 2.0)) for _ in range(2)]
        ring.publish(0, kv, meta0)
        self.assertEqual(ring.history(), [(0, meta0)])
        self.assertEqual(ring.signature(), ((0, 4, True),))
        self.assertEqual(ring.keys[0][0, :6, 0, 0].tolist(), [1, 1, 1, 1, 0, 0])
        meta1 = _committed(schedule[1], 1)
        with self.assertRaisesRegex(ValueError, "Cannot publish"):
            ring.publish(1, kv, meta1)


def _config():
    config = Cosmos3SimAutoConfig()
    with (
        mock.patch(
            "sglang.multimodal_gen.configs.pipeline_configs.cosmos3_sim_auto._transformer_config",
            return_value=_transformer_config(),
        ),
        mock.patch(
            "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.get_distilled_sigmas",
            return_value=[1.0, 0.9375, 0.8333333333333334, 0.625],
        ),
        mock.patch(
            "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.is_edge_checkpoint",
            return_value=False,
        ),
        mock.patch(
            "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.is_nano_checkpoint",
            return_value=True,
        ),
    ):
        config.update_config_from_dict({"model_path": "/models/sim-auto"})
    return config


def _adjusted(params, config):
    deployment = config.multiview_deployment
    params._apply_deployment_defaults(deployment)
    params._resolve_canvas(deployment)
    params._apply_guidance_policy(deployment)
    params._enforce_student_contract(config.sim_auto_manifest)
    return params


class TestConfigAndSamplingParams(unittest.TestCase):
    def test_attention_backend_switch_is_shared_with_transfer_auto(self):
        """Config field, then the multiview env var, then auto; unknown kernels are rejected."""
        config = _config()
        with mock.patch.dict(os.environ, {MULTIVIEW_BACKEND_ENV_VAR: ""}):
            self.assertEqual(config.resolved_multiview_backend(), "auto")
        with mock.patch.dict(os.environ, {MULTIVIEW_BACKEND_ENV_VAR: "fa4"}):
            self.assertEqual(config.resolved_multiview_backend(), "fa4")
            config.multiview_attention_backend = "triton"
            self.assertEqual(config.resolved_multiview_backend(), "triton")
            config.multiview_attention_backend = None
        with mock.patch.dict(os.environ, {MULTIVIEW_BACKEND_ENV_VAR: "maskless"}):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                config.resolved_multiview_backend()

    def test_config_resolves_manifest_and_sequence_parallel_rules(self):
        """Ulysses only, equal to sp_degree, dividing the heads with TP; gather and ring are refused."""
        config = _config()
        self.assertEqual(config.sim_auto_manifest.frames_per_chunk, 3)
        self.assertEqual(config.multiview_deployment.num_views, 11)
        self.assertEqual(config.sim_auto_heads, (32, 8))
        self.assertIsNone(config.scheduler_class_override)
        deployment = config.get_model_deployment_config()
        self.assertFalse(deployment.supports_cfg_parallel)
        base = dict(
            tp_size=1,
            sp_degree=1,
            ulysses_degree=1,
            ring_degree=1,
            kv_gather_degree=1,
            enable_cfg_parallel=False,
            num_gpus=1,
        )
        config.validate_server_args(SimpleNamespace(**base))
        config.validate_server_args(SimpleNamespace(**{**base, "tp_size": 2}))
        for accepted in (
            {"sp_degree": 2, "ulysses_degree": 2},
            {"sp_degree": 8, "ulysses_degree": 8},
            {"tp_size": 2, "sp_degree": 4, "ulysses_degree": 4},
        ):
            config.validate_server_args(SimpleNamespace(**{**base, **accepted}))
        for rejected, message in (
            ({"ring_degree": 2}, "ring_degree"),
            (
                {"kv_gather_degree": 2, "ulysses_degree": 2, "sp_degree": 2},
                "kv_gather_degree",
            ),
            ({"sp_degree": 2}, "must equal"),
            ({"sp_degree": 3, "ulysses_degree": 3}, "divide"),
            ({"tp_size": 4, "sp_degree": 4, "ulysses_degree": 4}, "divide"),
        ):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(ValueError, message):
                    config.validate_server_args(SimpleNamespace(**{**base, **rejected}))
        with (
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3_sim_auto._transformer_config",
                return_value=_transformer_config(),
            ),
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.get_distilled_sigmas",
                return_value=[1.0, 0.5],
            ),
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.is_edge_checkpoint",
                return_value=False,
            ),
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.is_nano_checkpoint",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "different fixed-step schedules"):
                Cosmos3SimAutoConfig().update_config_from_dict(
                    {"model_path": "/models/sim-auto"}
                )

    def test_defaults_follow_the_student(self):
        config = _config()
        params = _adjusted(Cosmos3SimAutoSamplingParams(), config)
        self.assertEqual((params.width, params.height), (832, 480))
        self.assertEqual(params.num_frames, 297)
        self.assertEqual(params.fps, 30)
        self.assertEqual(params.num_inference_steps, 4)
        self.assertEqual(params.guidance_scale, 1.0)
        self.assertEqual(params.flow_shift, 1.0)
        self.assertFalse(params.emphasize_control_in_prompt)
        self.assertTrue(params.resolved_condition_video_as_image())
        self.assertTrue(Cosmos3SimAutoSamplingParams.prompt_optional)

    def test_rejects_student_contract_violations(self):
        config = _config()
        for kwargs, message in (
            ({"guidance_scale": 3.0}, "guidance_scale=1.0 only"),
            ({"num_inference_steps": 8}, "fixed 4-step"),
            ({"flow_shift": 5.0}, "no flow_shift"),
            (
                {
                    "multiview": {
                        "views": [
                            {
                                "camera_key": "camera_front_wide_120fov",
                                "control_path": "c.mp4",
                                "prompt": "p",
                            }
                        ],
                        "condition_video_as_image": False,
                    }
                },
                "first frame only",
            ),
            (
                {
                    "multiview": {
                        "views": [
                            {
                                "camera_key": "camera_front_wide_120fov",
                                "control_path": "c.mp4",
                                "prompt": "p",
                            }
                        ],
                        "condition_frame_indexes_vision": [0, 1],
                    }
                },
                "latent frame 0",
            ),
            ({"lidar": {"control_path": "hdmap.safetensors"}, "fps": 24}, "fps=30"),
        ):
            with self.subTest(kwargs=kwargs):
                params = Cosmos3SimAutoSamplingParams(**kwargs)
                params._explicit_fields = set(kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    _adjusted(params, config)
        with self.assertRaisesRegex(ValueError, "exactly one measured LiDAR sweep"):
            Cosmos3SimAutoSamplingParams(
                lidar={
                    "control_path": "h.safetensors",
                    "condition_path": "m.safetensors",
                    "num_conditional_sweeps": 2,
                }
            )
        with self.assertRaisesRegex(ValueError, "no unconditional branch"):
            Cosmos3SimAutoSamplingParams(per_view_negative_prompt="blurry")
        with self.assertRaisesRegex(TypeError, "requires Cosmos3SimAutoConfig"):
            Cosmos3SimAutoSamplingParams()._adjust(
                SimpleNamespace(pipeline_config=Cosmos3Config())
            )


class TestRegistry(unittest.TestCase):
    def test_pipeline_and_transformer_are_discoverable(self):
        _discover_and_register_pipelines()
        self.assertIn("Cosmos3NanoSimAutoPipeline", _PIPELINE_REGISTRY)
        self.assertIn("Cosmos3SimAutoTransformer", ModelRegistry.get_supported_archs())

    def test_release_path_resolves_here_and_neighbours_stay_put(self):
        for model_path in (
            "nvidia/Cosmos3-Nano-Sim-Auto",
            "/models/Cosmos3-Nano-Sim-Auto",
        ):
            with self.subTest(model_path=model_path):
                info = _get_config_info(model_path)
                self.assertIs(info.pipeline_config_cls, Cosmos3SimAutoConfig)
                self.assertIs(info.sampling_param_cls, Cosmos3SimAutoSamplingParams)
        info = _get_config_info("nvidia/Cosmos3-Nano")
        self.assertIs(info.pipeline_config_cls, Cosmos3Config)
        self.assertIs(info.sampling_param_cls, Cosmos3SamplingParams)
        self.assertIsNot(
            _get_config_info("nvidia/Cosmos3-Nano-Transfer-Auto").pipeline_config_cls,
            Cosmos3SimAutoConfig,
        )
        with mock.patch(
            "sglang.multimodal_gen.registry.maybe_download_model_index",
            return_value={"_class_name": "Cosmos3NanoSimAutoPipeline"},
        ):
            info = _get_config_info("acme/renamed-sim-auto")
        self.assertIs(info.pipeline_config_cls, Cosmos3SimAutoConfig)


class _Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        return tokens, None


def _stream_items(condition_frames=()):
    grid = SensorGrid(2, 2, 1, 1)
    return [
        StreamItem("rgb_control", (0, 1), (0, 1, 2), grid, 0.1),
        StreamItem("rgb_target", (0, 1), (0, 1, 2), grid, 0.1, condition_frames),
        StreamItem("lidar_control", (2,), (0, 1), grid, 0.1),
        StreamItem("lidar_target", (2,), (0, 1), grid, 0.1),
    ]


def _sim_auto_model():
    model = cosmos3_sim_auto.Cosmos3SimAutoTransformer.__new__(
        cosmos3_sim_auto.Cosmos3SimAutoTransformer
    )
    torch.nn.Module.__init__(model)
    model._rope_inputs_key = model._rope_inputs = None
    model._generated_masks_key = model._generated_masks = None
    return model


class TestHostOverhead(unittest.TestCase):
    def test_steps_scatter_and_gather_without_uploads(self):
        items = _stream_items(condition_frames=(0,))
        latents = [
            torch.randn(1, 3, 6, 2, 2),
            torch.randn(1, 3, 6, 2, 2),
            torch.randn(1, 4, 2, 2, 2),
            torch.randn(1, 4, 2, 2, 2),
        ]
        layout = _FlatLayout.of(
            items, latents, num_views=2, noisy_rgb=(1, 2), noisy_sweeps=(1,)
        )
        x = torch.randn(1, layout.size)
        with RecordOps() as ops:
            current = layout.scatter(latents, x)
            flat = layout.gather([current[1], current[3]])
        self.assertEqual(ops.uploads(), [])
        # the generated frames hold x and read back as x; the others are untouched
        rgb = current[1].view(1, 3, 2, 3, 2, 2)
        torch.testing.assert_close(
            (flat, rgb[:, :, :, 0], current[3][:, :, 0]),
            (
                x,
                latents[1].view(1, 3, 2, 3, 2, 2)[:, :, :, 0],
                latents[3][:, :, 0],
            ),
            rtol=0,
            atol=0,
        )

    def test_time_embed_masks_built_once_per_chunk(self):
        model = _sim_auto_model()
        model.proj_in = _Projection()
        model.patchify = lambda latent, frames, h, w: latent.flatten(2).transpose(1, 2)
        model.time_embedder = lambda timestep: timestep.view(1, 1).expand(1, 3)
        model.rig_view_embed = torch.nn.Embedding(4, 3)
        model.rig_lidar_id = 3
        items = _stream_items(condition_frames=(0,))[:2]
        latents = [torch.randn(1, 3, 6, 2, 2), torch.randn(1, 3, 6, 2, 2)]
        rig_view_ids = torch.tensor([0, 1])

        def embed(timestep):
            with mock.patch.object(
                cosmos3_sim_auto, "add_rig_view_rows", lambda hidden, rows, n: hidden
            ):
                return model.embed_items(
                    items, latents, timestep=timestep, rig_view_ids=rig_view_ids
                )

        timesteps = [torch.tensor(500.0), torch.tensor(250.0)]
        with mock.patch.object(torch, "tensor", wraps=torch.tensor) as upload:
            steps = [embed(timestep) for timestep in timesteps]
        # one generated-frame mask (the target's) for the whole chunk
        self.assertEqual(upload.call_count, 1)
        model._generated_masks_key = None
        torch.testing.assert_close(steps[1], embed(timesteps[1]), rtol=0, atol=0)

    def test_rope_inputs_built_once_per_chunk(self):
        model = _sim_auto_model()
        items = _stream_items()[::2]  # controls only: no output projection
        num_tokens = sum(item.num_tokens for item in items)
        model.num_hidden_layers = 0
        model.gen_layers = []
        model.ulysses = None
        model.norm_moe_gen = lambda hidden: hidden
        model.embed_items = lambda items, latents, **_: torch.zeros(1, num_tokens, 3)
        model.chunk_position_ids = mock.Mock(return_value=torch.zeros(3, num_tokens))
        model.language_model = SimpleNamespace(
            rotary_emb=SimpleNamespace(
                build_rope_cache_inputs=lambda ids, cache_dtype: (ids, ids)
            )
        )
        for timestep in (torch.ones(1), torch.zeros(1), None):
            model(
                items=items,
                latents=[None, None],
                timestep=timestep,
                text_extent=7,
                und_kv=[],
                history_kv=[],
                mask=None,
                rig_view_ids=None,
            )
        # the chunk's denoise steps and its clean refresh share the RoPE inputs
        self.assertEqual(model.chunk_position_ids.call_count, 1)


if __name__ == "__main__":
    unittest.main()
