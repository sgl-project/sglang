# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Cosmos-Dreams port: artifact validation, causal frame
partitioning, action packing, interleaved mRoPE ids, K/V history, and registry
wiring."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    ActionNormalizerContract,
    CosmosDreamsActionContract,
    canonical_sha256,
    float32_value,
    load_cosmos_dreams_manifest,
    load_unified_normalizer,
    parse_cosmos_dreams_manifest,
    resolve_inference_profile,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams_realtime import (
    CosmosDreamsRealtimeConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.cosmos_dreams import (
    CosmosDreamsSamplingParams,
)
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
    build_cosmos_dreams_position_ids,
    interleave_action_vision_tokens,
    null_action_token_positions,
    split_interleaved_action_vision_tokens,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams import (
    actions_for_frames,
    append_kv_history,
    closest_canvas,
    crop_geometry_to_content,
    domains_for_frames,
    fit_image_to_canvas,
    format_dreams_prompt,
    frame_domain_ids,
    iter_ar_chunk_ranges,
    iter_clean_commit_frames,
    latent_frame_count,
    load_action_rows,
    normalize_action_rows,
    pad_action_rows,
    prepare_action_rows,
    resolve_geometry,
    sde_step_generator,
)
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    _lighter_valid_num_frames,
)

# ``transformer/config.json["cosmos3_nano_sim_bimanual"]`` of Cosmos3-Nano-Sim-Bimanual
# (checkpoint causal_8b_sf_dmd_max_4step_cam_chunk4_480p_961f@iter_000002000).
# The two SHA-256 digests are exporter-published and pin the hash
# canonicalization, so they must not be regenerated from this code.
CHECKPOINT_ARTIFACT = {
    "attention_mode": "three_way",
    "base_fps": 24.0,
    "checkpoint_hash": "9540afba2ac43d36b213a847f227b8011f5419feb670be722b518debdc22f63e",
    "checkpoint_id": "causal_8b_sf_dmd_max_4step_cam_chunk4_480p_961f@iter_000002000",
    "checkpoint_iteration": 2000,
    "chunk_size": 4,
    "conditioning": {
        "action_tokens_per_frame": 4,
        "contract_sha256": "2c75f43813f507ab5d382d5d64c41955b3871ee164c093fa098c8429cb8547db",
        "default_embodiment": "camera_pose",
        "embodiments": {
            "camera_pose": {
                "domain_id": 2,
                "layout": {
                    "delta_equation": "T_i^-1 @ T_{i+1}",
                    "fields": [
                        {
                            "name": "camera_translation",
                            "offset": 0,
                            "size": 3,
                            "unit": "meter",
                        },
                        {
                            "name": "camera_rotation",
                            "offset": 3,
                            "representation": "rot6d_columns",
                            "size": 6,
                            "unit": "dimensionless",
                        },
                    ],
                    "id": "camera_pose_backward_framewise_rot6d_v1",
                    "pose_convention": "backward_framewise",
                    "rotation_representation": "rot6d_columns",
                },
                "normalizer": {
                    "derivation": {"rotation_scale": 1.0, "translation_scale": 10.0},
                    "method": "pose_scale",
                    "schema_version": 1,
                    "training_config": {
                        "experiment": "causal_8b_sf_dmd_max_4step_cam_chunk4_480p_961f",
                        "repository_revision": "483375f79279832b52d9e5c3598345a86f7137d5",
                        "resolved_sha256": "0e7b2be45c82b13c6f1454cdafd7e75d1941a6565e3b72b15a03a99d65177f29",
                    },
                    "transform": {
                        "forward_clamp": False,
                        "offset": [0.0] * 9,
                        "scale": [0.10000000149011612] * 3 + [1.0] * 6,
                        "type": "affine",
                    },
                    "transform_sha256": "bb8102c686b4813c1cdcc82b78cb174225fa35f8bfafd07d36b81a767bb6e698",
                },
                "raw_action_dim": 9,
            }
        },
        "mode": "action",
        "model_action_dim": 64,
        "num_embodiment_domains": 32,
        "padding": {"stage": "after_normalization", "value": 0.0},
        "schema_version": 3,
        "training_config_excerpt": {
            "datasets": [
                {
                    "apply_forward_clamp": False,
                    "dataset_class": "CameraDatasetSharded",
                    "embodiment": "camera_pose",
                    "method": "pose_scale",
                    "mode": "forward_dynamics",
                    "pose_convention": "backward_framewise",
                    "rotation_format": "rot6d",
                    "rotation_scale": 1.0,
                    "translation_scale": 10.0,
                }
            ],
            "experiment": "causal_8b_sf_dmd_max_4step_cam_chunk4_480p_961f",
        },
    },
    "enable_fps_modulation": True,
    "fixed_step_sampler_config": {
        "num_train_timesteps": 1000,
        "sample_type": "sde",
        "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
    },
    "latent_patch_size": 2,
    "schema_version": 1,
    "sink_frames": 0,
    "temporal_compression_factor": 4,
    "temporal_modality_margin": 15000,
    "text_cache_max_len": 512,
    "unified_3d_mrope_reset_spatial_ids": True,
    "vae_spatial_compression_factor": 16,
    "video_temporal_causal": True,
    "window_frames": 96,
}

MANIFEST = parse_cosmos_dreams_manifest(CHECKPOINT_ARTIFACT)
CAMERA_TRANSFORM = MANIFEST.conditioning.embodiments["camera_pose"].normalizer.transform


def _artifact() -> dict:
    return copy.deepcopy(CHECKPOINT_ARTIFACT)


AGIBOT_LAYOUT = {
    "id": "agibot_backward_framewise_rot6d_v1",
    "pose_convention": "backward_framewise",
    "rotation_representation": "rot6d_columns",
    "delta_equation": "T_i^-1 @ T_{i+1}",
    "fields": [
        {"name": "head_translation", "offset": 0, "size": 3, "unit": "meter"},
        {
            "name": "head_rotation",
            "offset": 3,
            "size": 6,
            "unit": "dimensionless",
            "representation": "rot6d_columns",
        },
        {"name": "right_translation", "offset": 9, "size": 3, "unit": "meter"},
        {
            "name": "right_rotation",
            "offset": 12,
            "size": 6,
            "unit": "dimensionless",
            "representation": "rot6d_columns",
        },
        {
            "name": "right_gripper",
            "offset": 18,
            "size": 1,
            "unit": "open_fraction",
            "closed_value": 0.0,
            "open_value": 1.0,
        },
        {"name": "left_translation", "offset": 19, "size": 3, "unit": "meter"},
        {
            "name": "left_rotation",
            "offset": 22,
            "size": 6,
            "unit": "dimensionless",
            "representation": "rot6d_columns",
        },
        {
            "name": "left_gripper",
            "offset": 28,
            "size": 1,
            "unit": "open_fraction",
            "closed_value": 0.0,
            "open_value": 1.0,
        },
    ],
}


def _two_embodiment_artifact() -> dict:
    """The camera fixture plus a synthetic 29-D AgiBot embodiment, hashes recomputed.

    Mirrors the release export (default embodiment agibotworld, domain 15, a
    quantile normalizer resolved to an unclamped affine transform).
    """
    artifact = _artifact()
    conditioning = artifact["conditioning"]
    conditioning["default_embodiment"] = "agibotworld"
    conditioning["embodiments"]["agibotworld"] = {
        "domain_id": 15,
        "raw_action_dim": 29,
        "layout": copy.deepcopy(AGIBOT_LAYOUT),
        "normalizer": {
            "schema_version": 1,
            "method": "quantile_rot",
            "derivation": {"low_key": "q01", "high_key": "q99", "range_floor": 1e-8},
            "transform": {
                "type": "affine",
                "forward_clamp": False,
                # The exporter publishes float32-exact values; the validator checks it.
                "offset": [float32_value(0.001 * i) for i in range(29)],
                "scale": [float32_value(0.01 * (i + 1)) for i in range(29)],
            },
            "transform_sha256": "0" * 64,
        },
    }
    for embodiment in conditioning["embodiments"].values():
        normalizer = embodiment["normalizer"]
        parsed = msgspec.convert(normalizer, type=ActionNormalizerContract)
        normalizer["transform_sha256"] = canonical_sha256(parsed.behavioral_payload())
    conditioning["contract_sha256"] = "0" * 64
    parsed = msgspec.convert(conditioning, type=CosmosDreamsActionContract)
    conditioning["contract_sha256"] = canonical_sha256(parsed.behavioral_payload())
    return artifact


class TestCosmosDreamsManifest(unittest.TestCase):
    def test_checkpoint_artifact_parses_with_published_hashes(self):
        self.assertEqual(MANIFEST.chunk_size, 4)
        self.assertEqual(MANIFEST.window_frames, 96)
        self.assertEqual(MANIFEST.t_list, (1.0, 0.9375, 0.8333333333333334, 0.625))
        self.assertEqual(MANIFEST.action_tokens_per_frame, 4)
        self.assertEqual(MANIFEST.max_action_dim, 64)
        self.assertEqual(MANIFEST.conditioning.embodiment_to_domain, {"camera_pose": 2})

    def test_tampered_contract_fails_hash_verification(self):
        """A stale or hand-edited normalizer/layout must not load: the published
        digests cover exactly the payload that changes model output."""
        tampered_scale = _artifact()
        embodiment = tampered_scale["conditioning"]["embodiments"]["camera_pose"]
        embodiment["normalizer"]["transform"]["scale"][3] = 0.5
        with self.assertRaises(ValueError):
            parse_cosmos_dreams_manifest(tampered_scale)

        tampered_layout = _artifact()
        embodiment = tampered_layout["conditioning"]["embodiments"]["camera_pose"]
        embodiment["layout"]["fields"][0]["unit"] = "millimeter"
        with self.assertRaises(ValueError):
            parse_cosmos_dreams_manifest(tampered_layout)

    def test_rejects_non_causal_or_incomplete_artifacts(self):
        for mutate in (
            lambda a: a.__setitem__("attention_mode", "full"),
            lambda a: a.__setitem__("video_temporal_causal", False),
            lambda a: a["fixed_step_sampler_config"].__setitem__("sample_type", "ode"),
            lambda a: a["fixed_step_sampler_config"].__setitem__(
                "t_list", [1.0, 0.5, 0.7]
            ),
            lambda a: a.pop("window_frames"),
            lambda a: a.__setitem__("unexpected_field", 1),
            lambda a: a["conditioning"].__setitem__("action_tokens_per_frame", 2),
        ):
            artifact = _artifact()
            mutate(artifact)
            with self.subTest(artifact=artifact):
                with self.assertRaises(ValueError):
                    parse_cosmos_dreams_manifest(artifact)

    def test_load_requires_artifact_block(self):
        with self.assertRaises(ValueError):
            load_cosmos_dreams_manifest({"hidden_size": 4096})

    def test_load_reads_the_exporter_envelope_only(self):
        # One envelope serves Sim-Bimanual and Sim-Transfer exports; the legacy
        # ``cosmos_dreams`` block and unnamed blocks are rejected with a re-export hint.
        manifest = load_cosmos_dreams_manifest(
            {"hidden_size": 4096, "cosmos3_nano_sim_bimanual": _artifact()}
        )
        self.assertEqual(manifest.checkpoint_iteration, 2000)
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            load_cosmos_dreams_manifest({"cosmos_dreams": _artifact()})
        with self.assertRaisesRegex(ValueError, "cosmos3_nano_sim_bimanual"):
            load_cosmos_dreams_manifest({"cosmos3_nano_sim_x": _artifact()})

    def test_pre_normalized_rows_skip_the_affine_but_keep_checks(self):
        manifest = parse_cosmos_dreams_manifest(_two_embodiment_artifact())
        contract = manifest.conditioning.embodiments["agibotworld"]
        rows = (torch.arange(4 * 29, dtype=torch.float32).view(4, 29) / 50.0).tolist()
        common = dict(manifest=manifest, embodiment="agibotworld", target_frame=2)
        raw, domains = prepare_action_rows(rows, pre_normalized=False, **common)
        pre, _ = prepare_action_rows(rows, pre_normalized=True, **common)
        self.assertIsNone(domains)
        self.assertEqual(tuple(raw.shape), (4, 64))
        self.assertEqual(tuple(pre.shape), (4, 64))
        torch.testing.assert_close(pre[:, :29], torch.tensor(rows))
        torch.testing.assert_close(
            raw[:, :29],
            normalize_action_rows(torch.tensor(rows), contract.normalizer.transform),
        )
        self.assertEqual(pre[:, 29:].abs().sum().item(), 0.0)
        with self.assertRaises(ValueError):
            prepare_action_rows(
                (torch.arange(4 * 9, dtype=torch.float32).view(4, 9)).tolist(),
                pre_normalized=True,
                **common,
            )
        with self.assertRaises(ValueError):
            prepare_action_rows(rows[:2], pre_normalized=True, **common)
        with self.assertRaisesRegex(ValueError, "unified"):
            prepare_action_rows(
                rows,
                pre_normalized=True,
                domain_names=[
                    "agibotworld",
                    "camera_pose",
                    "agibotworld",
                    "agibotworld",
                ],
                **common,
            )

    def test_two_embodiment_contract_resolves_and_normalizes(self):
        manifest = parse_cosmos_dreams_manifest(_two_embodiment_artifact())
        contract = manifest.conditioning
        self.assertEqual(
            contract.embodiment_to_domain, {"agibotworld": 15, "camera_pose": 2}
        )
        self.assertEqual(contract.resolve_embodiment(None, None), "agibotworld")
        self.assertEqual(contract.resolve_embodiment(None, 15), "agibotworld")
        self.assertEqual(contract.resolve_embodiment(None, 2), "camera_pose")
        self.assertEqual(
            contract.resolve_embodiment("camera_pose", None), "camera_pose"
        )
        with self.assertRaises(ValueError):
            contract.resolve_embodiment("agibotworld", 2)
        agibot = contract.embodiments["agibotworld"]
        rows = torch.arange(2 * 29, dtype=torch.float32).view(2, 29) / 100.0
        normalized = normalize_action_rows(rows, agibot.normalizer.transform)
        expected = (
            rows - torch.tensor(agibot.normalizer.transform.offset)
        ) / torch.tensor(agibot.normalizer.transform.scale)
        torch.testing.assert_close(normalized, expected)
        self.assertEqual(pad_action_rows(normalized, 64).shape, (2, 64))
        with self.assertRaises(ValueError):
            normalize_action_rows(rows[:, :9], agibot.normalizer.transform)

    def test_resolve_embodiment_by_name_or_domain(self):
        contract = MANIFEST.conditioning
        self.assertEqual(contract.resolve_embodiment(None, None), "camera_pose")
        self.assertEqual(contract.resolve_embodiment(None, 2), "camera_pose")
        self.assertEqual(contract.resolve_embodiment("Camera_Pose", 2), "camera_pose")
        with self.assertRaises(ValueError):
            contract.resolve_embodiment(None, 15)
        with self.assertRaises(ValueError):
            contract.resolve_embodiment("camera_pose", 15)
        with self.assertRaises(ValueError):
            contract.resolve_embodiment("agibotworld", None)


class TestCosmosDreamsActions(unittest.TestCase):
    def test_pose_scale_normalizer_scales_translation_only(self):
        rows = torch.tensor([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        normalized = normalize_action_rows(rows, CAMERA_TRANSFORM)
        torch.testing.assert_close(
            normalized,
            torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            atol=1e-5,
            rtol=0,
        )
        padded = pad_action_rows(normalized, 64)
        self.assertEqual(tuple(padded.shape), (1, 64))
        self.assertEqual(padded[0, 9:].abs().sum().item(), 0.0)
        with self.assertRaises(ValueError):
            normalize_action_rows(torch.zeros(1, 29), CAMERA_TRANSFORM)

    def test_actions_for_frames_uses_global_row_layout(self):
        rows = (
            torch.arange(16, dtype=torch.float32)
            .view(16, 1)
            .expand(16, 64)
            .contiguous()
        )
        block, null_indexes = actions_for_frames(
            rows,
            frame_start=1,
            frame_end=5,
            action_tokens_per_frame=4,
            model_action_dim=64,
        )
        self.assertEqual(tuple(block.shape), (1, 16, 64))
        torch.testing.assert_close(
            block[0, :, 0], torch.arange(16, dtype=torch.float32)
        )
        self.assertEqual(null_indexes, ())

        block, null_indexes = actions_for_frames(
            rows,
            frame_start=0,
            frame_end=1,
            action_tokens_per_frame=4,
            model_action_dim=64,
        )
        self.assertEqual(block.abs().sum().item(), 0.0)
        self.assertEqual(null_indexes, (0,))

        with self.assertRaises(ValueError):
            actions_for_frames(
                rows,
                frame_start=5,
                frame_end=9,
                action_tokens_per_frame=4,
                model_action_dim=64,
            )

        block, null_indexes = actions_for_frames(
            None,
            frame_start=5,
            frame_end=9,
            action_tokens_per_frame=4,
            model_action_dim=64,
        )
        self.assertEqual(tuple(block.shape), (1, 16, 64))
        self.assertEqual(null_indexes, (0, 1, 2, 3))

    def test_load_action_rows_accepts_json_and_files(self):
        rows = [[0.0] * 9 for _ in range(4)]
        self.assertEqual(tuple(load_action_rows(json.dumps(rows)).shape), (4, 9))
        self.assertEqual(tuple(load_action_rows([rows]).shape), (4, 9))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "actions.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(rows, handle)
            self.assertEqual(tuple(load_action_rows(path).shape), (4, 9))
        with self.assertRaises(ValueError):
            load_action_rows([1.0, 2.0])


class TestCosmosDreamsChunking(unittest.TestCase):
    def test_chunk_ranges_follow_training_partition(self):
        self.assertEqual(list(iter_ar_chunk_ranges(0, 9, 4)), [(0, 1), (1, 5), (5, 9)])
        self.assertEqual(list(iter_ar_chunk_ranges(1, 7, 4)), [(1, 5), (5, 7)])
        self.assertEqual(list(iter_ar_chunk_ranges(0, 1, 4)), [(0, 1)])
        self.assertEqual(list(iter_ar_chunk_ranges(3, 3, 4)), [])

    def test_clean_commit_skips_globally_final_frame(self):
        self.assertEqual(
            list(iter_clean_commit_frames(5, 9, target_frame=9)),
            [(0, 5), (1, 6), (2, 7)],
        )
        self.assertEqual(
            list(iter_clean_commit_frames(1, 5, target_frame=9)),
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        self.assertEqual(list(iter_clean_commit_frames(0, 1, target_frame=1)), [])

    def test_latent_frame_count_and_geometry(self):
        self.assertEqual(latent_frame_count(81, 4), 21)
        self.assertEqual(latent_frame_count(17, 4), 5)
        geometry = resolve_geometry(
            height=720, width=1280, manifest=MANIFEST, max_pixels=921_600
        )
        self.assertEqual((geometry.latent_height, geometry.latent_width), (45, 80))
        self.assertEqual((geometry.grid_height, geometry.grid_width), (23, 40))
        self.assertEqual(geometry.tokens_per_frame(4), 924)
        with self.assertRaises(ValueError):
            resolve_geometry(
                height=736, width=1280, manifest=MANIFEST, max_pixels=921_600
            )
        with self.assertRaises(ValueError):
            resolve_geometry(
                height=480, width=1280, manifest=MANIFEST, max_pixels=921_600
            )

    def test_crop_geometry_to_content_drops_padded_latents(self):
        canvas = resolve_geometry(
            height=480, width=832, manifest=MANIFEST, max_pixels=921_600
        )
        # 1596x980 fitted to 832x480 keeps 782 content columns -> 48 latent columns.
        cropped = crop_geometry_to_content(
            canvas, content_size=(480, 782), manifest=MANIFEST
        )
        self.assertEqual((cropped.height, cropped.width), (480, 768))
        self.assertEqual((cropped.latent_height, cropped.latent_width), (30, 48))
        self.assertEqual((cropped.grid_height, cropped.grid_width), (15, 24))
        # Odd latent sizes stay odd; the transformer pads them to the patch grid.
        odd = crop_geometry_to_content(
            canvas, content_size=(470, 775), manifest=MANIFEST
        )
        self.assertEqual((odd.latent_height, odd.latent_width), (29, 48))
        self.assertEqual((odd.grid_height, odd.grid_width), (15, 24))
        self.assertEqual((odd.height, odd.width), (464, 768))
        self.assertEqual(
            crop_geometry_to_content(
                canvas, content_size=(480, 832), manifest=MANIFEST
            ),
            canvas,
        )
        with self.assertRaises(ValueError):
            crop_geometry_to_content(canvas, content_size=(480, 848), manifest=MANIFEST)
        with self.assertRaises(ValueError):
            crop_geometry_to_content(canvas, content_size=(0, 832), manifest=MANIFEST)


class TestCosmosDreamsPositionIds(unittest.TestCase):
    OFFSET = 10 + 15000

    def _ids(self, **overrides):
        kwargs = dict(
            frame_start=1,
            num_frames=4,
            grid_h=2,
            grid_w=3,
            text_length=10,
            temporal_modality_margin=15000,
            fps=24.0,
            base_fps=24.0,
            temporal_compression_factor=4,
            action_tokens_per_frame=4,
            null_action_frames=(),
            device=torch.device("cpu"),
        )
        kwargs.update(overrides)
        ids = build_cosmos_dreams_position_ids(**kwargs)
        tokens_per_frame = (
            kwargs["action_tokens_per_frame"] + kwargs["grid_h"] * kwargs["grid_w"]
        )
        return ids.view(3, kwargs["num_frames"], tokens_per_frame)

    def test_action_tokens_trail_their_frame_by_quarter_steps(self):
        ids = self._ids()
        for local, frame in enumerate(range(1, 5)):
            expected_action_t = (
                self.OFFSET + frame + torch.tensor([-0.75, -0.5, -0.25, 0.0])
            )
            torch.testing.assert_close(ids[0, local, :4], expected_action_t)
            torch.testing.assert_close(
                ids[0, local, 4:], torch.full((6,), float(self.OFFSET + frame))
            )
            torch.testing.assert_close(
                ids[1, local, 4:], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            )
            torch.testing.assert_close(
                ids[2, local, 4:], torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
            )
            self.assertEqual(ids[1:, local, :4].abs().sum().item(), 0.0)

    def test_null_first_frame_colocates_actions_with_vision(self):
        ids = self._ids(frame_start=0, null_action_frames=(0,))
        torch.testing.assert_close(ids[0, 0, :4], torch.full((4,), float(self.OFFSET)))
        # Later frames keep the real-action ids even when they are null too.
        torch.testing.assert_close(
            self._ids(frame_start=0, null_action_frames=(0, 1))[0, 1, :4],
            self.OFFSET + 1 + torch.tensor([-0.75, -0.5, -0.25, 0.0]),
        )

    def test_single_frame_commit_keeps_fps_modulation(self):
        """A one-frame clean commit at fps != base_fps must use the same scaled
        time as the frame had inside its denoised chunk."""
        ids = self._ids(frame_start=5, num_frames=1, fps=15.0)
        torch.testing.assert_close(ids[0, 0, 4:], torch.full((6,), self.OFFSET + 8.0))
        torch.testing.assert_close(
            ids[0, 0, :4], self.OFFSET + torch.tensor([6.8, 7.2, 7.6, 8.0])
        )

    def test_interleave_split_roundtrip_and_null_positions(self):
        action = torch.randn(1, 3, 4, 8)
        vision = torch.randn(1, 3, 6, 8)
        packed = interleave_action_vision_tokens(action, vision)
        self.assertEqual(tuple(packed.shape), (1, 30, 8))
        unpacked_action, unpacked_vision = split_interleaved_action_vision_tokens(
            packed, num_frames=3, action_tokens_per_frame=4, vision_tokens_per_frame=6
        )
        torch.testing.assert_close(unpacked_action, action)
        torch.testing.assert_close(unpacked_vision, vision)
        self.assertEqual(
            null_action_token_positions(
                (0, 2), num_frames=3, tokens_per_frame=10, action_tokens_per_frame=4
            ),
            [0, 1, 2, 3, 20, 21, 22, 23],
        )
        with self.assertRaises(ValueError):
            null_action_token_positions(
                (3,), num_frames=3, tokens_per_frame=10, action_tokens_per_frame=4
            )


class TestCosmosDreamsSdeSeeding(unittest.TestCase):
    def test_step_seed_mixes_seed_chunk_and_step_like_the_reference(self):
        device = torch.device("cpu")
        g = sde_step_generator(seed=42, frame_start=5, step_index=2, device=device)
        self.assertEqual(g.initial_seed(), 42 + 5 * 1_000_003 + 3 * 9_176)
        # Step 0 of frame 0 must not reuse the initial chunk noise seed (``seed``).
        first = sde_step_generator(seed=42, frame_start=0, step_index=0, device=device)
        self.assertNotEqual(first.initial_seed(), 42)
        # Distinct (chunk, step) pairs never collide for the four-step schedule.
        seeds = {
            sde_step_generator(
                seed=0, frame_start=f, step_index=k, device=device
            ).initial_seed()
            for f in range(0, 200)
            for k in range(4)
        }
        self.assertEqual(len(seeds), 800)


class TestCosmosDreamsKVHistory(unittest.TestCase):
    @staticmethod
    def _frame_kv(frame: int, layers: int = 2, tokens_per_frame: int = 2):
        value = torch.full((1, tokens_per_frame, 1, 1), float(frame))
        return [(value.clone(), value.clone()) for _ in range(layers)]

    def _rollout(self, *, sink_frames: int, window_frames: int, frames: int = 5):
        history = None
        for frame in range(frames):
            history = append_kv_history(
                history,
                self._frame_kv(frame),
                tokens_per_frame=2,
                sink_frames=sink_frames,
                window_frames=window_frames,
            )
        return history

    def test_window_keeps_latest_frames(self):
        history = self._rollout(sink_frames=0, window_frames=3)
        self.assertEqual(len(history), 2)
        for key, value in history:
            torch.testing.assert_close(
                key[0, :, 0, 0], torch.tensor([2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
            )
            torch.testing.assert_close(key, value)

    def test_sink_frames_survive_the_window(self):
        history = self._rollout(sink_frames=1, window_frames=3)
        torch.testing.assert_close(
            history[0][0][0, :, 0, 0],
            torch.tensor([0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
        )

    def test_layer_count_change_is_rejected(self):
        history = self._rollout(sink_frames=0, window_frames=3, frames=1)
        with self.assertRaises(ValueError):
            append_kv_history(
                history,
                self._frame_kv(1, layers=3),
                tokens_per_frame=2,
                sink_frames=0,
                window_frames=3,
            )


UNIFIED_FIELDS = [
    ("ego_pose", 0, 9),
    ("right_wrist_pose", 9, 9),
    ("right_fingertips", 18, 15),
    ("right_gripper_width_m", 33, 1),
    ("left_wrist_pose", 34, 9),
    ("left_fingertips", 43, 15),
    ("left_gripper_width_m", 58, 1),
]
UNIFIED_STATS_SHA = hashlib.sha256(
    json.dumps({"global": {"q01": [-1.0] * 59, "q99": [1.0] * 59}}).encode()
).hexdigest()


def _unified_artifact() -> dict:
    """A schema-5 (unified_v1) export: chunk 2, two-step sampler, unbounded history."""
    artifact = _artifact()
    artifact.update(
        checkpoint_id="simulation_sfdmd_bimanual_prod_64N_chunk2_v001@iter_000001000",
        checkpoint_iteration=1000,
        chunk_size=2,
        window_frames=None,
        sink_frames=0,
    )
    artifact["fixed_step_sampler_config"]["t_list"] = [1.0, 0.8333333333333334]
    conditioning = {
        "mode": "action",
        "schema_version": 5,
        "action_tokens_per_frame": 4,
        "model_action_dim": 64,
        "num_embodiment_domains": 32,
        "default_embodiment": "agibotworld",
        "embodiments": {
            "agibotworld": {"domain_id": 15, "input_action_dim": 59},
            "agibot_gear_gripper": {"domain_id": 15, "input_action_dim": 59},
            "camera_pose": {"domain_id": 2, "input_action_dim": 59},
            "hand_pose": {"domain_id": 3, "input_action_dim": 59},
        },
        "padding": {"stage": "after_normalization_and_validity_mask", "value": 0.0},
        "input_contract": {
            "action_space": "normalized_unified_v1",
            "action_dim": 59,
            "domain_routing": "scalar_or_per_action_row",
            "validity": "masked_to_zero_after_normalization_by_source",
            "runtime_normalization": False,
            "model_mode": "forward_dynamics",
        },
        "layout": {
            "id": "unified_v1",
            "pose_convention": "backward_chunk_anchored_16f",
            "rotation_representation": "rot6d_columns",
            "fields": [
                {"name": name, "offset": offset, "size": size}
                for name, offset, size in UNIFIED_FIELDS
            ],
        },
        "normalizer": {
            "method": "global_asinh_unified_v1",
            "runtime_application": False,
            "source": {
                "artifact_path": f"cosmos3_nano_sim_bimanual_action_sources/{UNIFIED_STATS_SHA}.json",
                "sha256": UNIFIED_STATS_SHA,
                "path": "projects/cosmos3/cosmos3/datasets/action/normalizers/global_asinh_v1.json",
                "repository_revision": "9368f59bb86ce27dbdc6110b327fe9a0ba154d2b",
                "pose_convention_authority": "resolved_canonical_lance_training_config",
            },
        },
        "training_config_excerpt": {
            "experiment": "simulation_sfdmd_bimanual_prod_64N_chunk2_v001",
            "loader": {"action_schema": "unified_v1", "format_prompt_as_json": False},
        },
        "contract_sha256": "0" * 64,
    }
    payload = {
        k: v for k, v in conditioning.items() if k not in ("mode", "contract_sha256")
    }
    conditioning["contract_sha256"] = canonical_sha256(payload)
    artifact["conditioning"] = conditioning
    return artifact


UNIFIED_MANIFEST = parse_cosmos_dreams_manifest(_unified_artifact())


def _write_unified_export(root: str) -> None:
    """Minimal export dir with the normalizer statistics the fixture names."""
    sources = os.path.join(root, "cosmos3_nano_sim_bimanual_action_sources")
    os.makedirs(sources, exist_ok=True)
    with open(os.path.join(sources, f"{UNIFIED_STATS_SHA}.json"), "w") as handle:
        handle.write(json.dumps({"global": {"q01": [-1.0] * 59, "q99": [1.0] * 59}}))


class TestCosmosDreamsUnifiedContract(unittest.TestCase):
    """Schema-5 exports: one normalized 59-D row layout, per-row domains,
    unbounded history and a two-step sampler must load and validate."""

    def test_unified_manifest_parses(self):
        manifest = UNIFIED_MANIFEST
        self.assertTrue(manifest.unified_actions)
        self.assertIsNone(manifest.window_frames)
        self.assertEqual(manifest.t_list, (1.0, 0.8333333333333334))
        self.assertEqual(manifest.action_input_dim("camera_pose"), 59)
        self.assertFalse(manifest.training_prompt_as_json)
        contract = manifest.action_contract
        self.assertEqual(contract.camera_domain_id, 2)
        self.assertEqual(contract.resolve_embodiment(None, None), "agibotworld")
        self.assertEqual(contract.resolve_embodiment(None, 15), "agibotworld")
        self.assertEqual(contract.resolve_embodiment("hand_pose", 3), "hand_pose")
        with self.assertRaises(ValueError):
            contract.resolve_embodiment("camera_pose", 15)
        # Two sigmas apply to every chunk; no step42 split remains to derive.
        profile = CosmosDreamsConfig().inference_profile(manifest)
        self.assertEqual(profile.frame_sigma_schedules, ((1.0, 0.8333333333333334),))
        self.assertEqual(profile.window_frames, 226)
        with self.assertRaisesRegex(ValueError, "window_frames=None"):
            resolve_inference_profile(
                manifest, history_mode="sliding", history_max_frames=901
            )

    def test_unified_contract_rejections(self):
        def broken(mutate):
            artifact = _unified_artifact()
            mutate(artifact)
            conditioning = artifact["conditioning"]
            payload = {
                k: v
                for k, v in conditioning.items()
                if k not in ("mode", "contract_sha256")
            }
            conditioning["contract_sha256"] = canonical_sha256(payload)
            return artifact

        cases = {
            "runtime normalization": lambda a: a["conditioning"]["normalizer"].update(
                runtime_application=True
            ),
            "other action space": lambda a: a["conditioning"]["input_contract"].update(
                action_space="raw"
            ),
            "framewise layout": lambda a: a["conditioning"]["layout"].update(
                pose_convention="backward_framewise"
            ),
            "layout gap": lambda a: a["conditioning"]["layout"]["fields"].pop(2),
            "narrow embodiment": lambda a: a["conditioning"]["embodiments"][
                "camera_pose"
            ].update(input_action_dim=9),
            "sinks without window": lambda a: a.update(sink_frames=1),
            "unknown schema": lambda a: a["conditioning"].update(schema_version=4),
        }
        for label, mutate in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                parse_cosmos_dreams_manifest(broken(mutate))
        tampered = _unified_artifact()
        tampered["conditioning"]["training_config_excerpt"]["experiment"] = "other"
        with self.assertRaisesRegex(ValueError, "contract_sha256"):
            parse_cosmos_dreams_manifest(tampered)

    def test_unified_rows_route_per_row_and_keep_camera_slots_clean(self):
        rows = torch.zeros(8, 59)
        rows[:, 3] = 1.0
        rows[:4, 12] = 1.0  # robot rows use wrist slots
        payload = {
            "action_space": "normalized_unified_v1",
            "action": rows.tolist(),
            "domain_names": ["agibotworld"] * 4 + ["camera_pose"] * 4,
        }
        padded, domains = prepare_action_rows(
            payload,
            manifest=UNIFIED_MANIFEST,
            embodiment="agibotworld",
            target_frame=3,
            pre_normalized=True,
        )
        self.assertEqual(tuple(padded.shape), (8, 64))
        self.assertEqual(domains.tolist(), [15] * 4 + [2] * 4)
        # Frame 0 borrows the first real row's domain; the robot chunk collapses to one id.
        self.assertEqual(
            domains_for_frames(
                domains,
                scalar_domain=15,
                frame_start=0,
                frame_end=1,
                action_tokens_per_frame=4,
            ).tolist(),
            [15],
        )
        mixed = domains_for_frames(
            domains,
            scalar_domain=15,
            frame_start=1,
            frame_end=3,
            action_tokens_per_frame=4,
        )
        self.assertEqual(tuple(mixed.shape), (1, 8))
        self.assertEqual(frame_domain_ids(mixed, 1, 4).tolist(), [[2, 2, 2, 2]])
        self.assertEqual(frame_domain_ids(torch.tensor([15]), 1, 4).tolist(), [15])
        # A single name or a uniform default-domain list means scalar routing.
        _, scalar = prepare_action_rows(
            dict(payload, domain_names=["agibotworld"]),
            manifest=UNIFIED_MANIFEST,
            embodiment="agibotworld",
            target_frame=3,
            pre_normalized=True,
        )
        self.assertIsNone(scalar)
        # Camera rows may not carry robot slots.
        with self.assertRaisesRegex(ValueError, "camera rows"):
            prepare_action_rows(
                dict(payload, domain_names=["camera_pose"] * 8),
                manifest=UNIFIED_MANIFEST,
                embodiment="agibotworld",
                target_frame=3,
                pre_normalized=True,
            )
        with self.assertRaisesRegex(ValueError, "59 slots"):
            prepare_action_rows(
                torch.zeros(8, 29).tolist(),
                manifest=UNIFIED_MANIFEST,
                embodiment="agibotworld",
                target_frame=3,
                pre_normalized=True,
            )
        with self.assertRaisesRegex(ValueError, "action_space"):
            prepare_action_rows(
                dict(payload, action_space="raw"),
                manifest=UNIFIED_MANIFEST,
                embodiment="agibotworld",
                target_frame=3,
                pre_normalized=True,
            )

    def test_per_token_projection_matches_scalar_routing(self):
        from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
            DomainAwareLinear,
        )
        from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
            project_actions,
        )

        torch.manual_seed(0)
        projection = DomainAwareLinear(64, 16, 32)
        with torch.no_grad():
            projection.fc.weight.normal_()
            projection.bias.weight.normal_()
        tokens = torch.randn(1, 8, 64)
        scalar = project_actions(projection, tokens, torch.tensor([15]), hidden_size=16)
        same = project_actions(
            projection, tokens, torch.tensor([[15] * 8]), hidden_size=16
        )
        torch.testing.assert_close(scalar, same)
        mixed = project_actions(
            projection, tokens, torch.tensor([[15] * 4 + [2] * 4]), hidden_size=16
        )
        camera = project_actions(
            projection, tokens[:, 4:], torch.tensor([2]), hidden_size=16
        )
        torch.testing.assert_close(mixed[:, :4], scalar[:, :4])
        torch.testing.assert_close(mixed[:, 4:], camera)
        with self.assertRaises(ValueError):
            project_actions(projection, tokens, torch.tensor([15, 2]), hidden_size=16)

    def test_unified_normalizer_identity_rows(self):
        with tempfile.TemporaryDirectory() as root:
            _write_unified_export(root)
            normalizer = load_unified_normalizer(root, UNIFIED_MANIFEST.action_contract)
        self.assertEqual(normalizer.offset, (0.0,) * 59)
        self.assertEqual(normalizer.scale, (1.0,) * 59)
        # q01=-1, q99=1: y = asinh(x) / asinh(1), so the unit value maps to 1.
        self.assertAlmostEqual(normalizer.normalize([1.0] + [0.0] * 58)[0], 1.0)
        camera = normalizer.identity_row(
            UNIFIED_MANIFEST.action_contract.layout, embodiment="camera_pose"
        )
        self.assertEqual([i for i, v in enumerate(camera) if v], [3, 7])
        robot = normalizer.identity_row(
            UNIFIED_MANIFEST.action_contract.layout, embodiment="agibotworld"
        )
        self.assertEqual([i for i, v in enumerate(robot) if v], [3, 7, 12, 16, 37, 41])
        bad = _unified_artifact()
        with (
            tempfile.TemporaryDirectory() as root,
            self.assertRaisesRegex(ValueError, "sha256"),
        ):
            _write_unified_export(root)
            path = os.path.join(
                root,
                "cosmos3_nano_sim_bimanual_action_sources",
                f"{UNIFIED_STATS_SHA}.json",
            )
            with open(path, "a") as handle:
                handle.write(" ")
            load_unified_normalizer(
                root, parse_cosmos_dreams_manifest(bad).action_contract
            )

    def test_unified_normalizer_resolves_hub_ids_through_the_snapshot(self):
        # --model-path nvidia/Cosmos3-Nano-Sim-Bimanual: the statistics file lives in
        # the cached snapshot, not under the literal id.
        requested = []

        def fake_component_path(component: str) -> str:
            requested.append(component)
            return os.path.join(root, "cosmos3_nano_sim_bimanual_action_sources")

        with tempfile.TemporaryDirectory() as root:
            _write_unified_export(root)
            with mock.patch(
                "sglang.multimodal_gen.runtime.utils.hf_diffusers_utils."
                "prepare_diffusers_component_path_for_loading",
                side_effect=fake_component_path,
            ):
                normalizer = load_unified_normalizer(
                    "nvidia/Cosmos3-Nano-Sim-Bimanual",
                    UNIFIED_MANIFEST.action_contract,
                )
        self.assertEqual(
            requested,
            [
                "nvidia/Cosmos3-Nano-Sim-Bimanual/cosmos3_nano_sim_bimanual_action_sources"
            ],
        )
        self.assertEqual(normalizer.scale, (1.0,) * 59)

    def test_unified_request_defaults_follow_the_training_loader(self):
        config = CosmosDreamsConfig()
        config.unified_actions = True
        config.prompt_json_caption = False
        params = CosmosDreamsSamplingParams(prompt="x", num_frames=9)
        params._apply_checkpoint_defaults(config)
        self.assertTrue(params.actions_pre_normalized)
        self.assertFalse(params.format_prompt_as_json)
        explicit = CosmosDreamsSamplingParams(
            prompt="x", num_frames=9, format_prompt_as_json=True
        )
        explicit._explicit_fields = {"format_prompt_as_json"}
        explicit._apply_checkpoint_defaults(config)
        self.assertTrue(explicit.format_prompt_as_json)
        rejected = CosmosDreamsSamplingParams(prompt="x", num_frames=9)
        rejected._explicit_fields = {"action_normalization"}
        with self.assertRaisesRegex(ValueError, "normalized unified_v1"):
            rejected._apply_checkpoint_defaults(config)
        native = CosmosDreamsSamplingParams(prompt="x", num_frames=9)
        native._apply_checkpoint_defaults(CosmosDreamsConfig())
        self.assertFalse(native.actions_pre_normalized)
        self.assertTrue(native.format_prompt_as_json)


class TestCosmosDreamsInferenceProfile(unittest.TestCase):
    """Per-chunk sigma schedules and history length are deployment settings the
    artifact does not carry; the defaults must not change the exported rollout
    except for keeping the whole-clip history training used."""

    STEP42 = [[1.0, 0.9375, 0.8333333333333334, 0.625], [1.0, 0.8333333333333334]]

    def test_defaults_apply_the_step42_budget_with_full_history(self):
        profile = CosmosDreamsConfig().inference_profile(MANIFEST)
        t_list = tuple(MANIFEST.t_list)
        self.assertEqual(
            profile.frame_sigma_schedules, (t_list, (t_list[0], t_list[2]))
        )
        self.assertEqual(profile.sigmas_for_frame(0), t_list)
        for frame in (1, 7, 500):
            self.assertEqual(profile.sigmas_for_frame(frame), (1.0, 0.8333333333333334))
        # 901 pixel frames = 226 latent frames: a whole training clip fits.
        self.assertEqual(profile.history_mode, "full")
        self.assertEqual(profile.window_frames, 226)
        self.assertEqual(profile.sink_frames, MANIFEST.sink_frames)
        self.assertEqual(profile.max_steps, 4)

    def test_artifact_schedule_on_every_chunk_is_an_explicit_override(self):
        config = CosmosDreamsConfig()
        config.update_pipeline_config(
            {"frame_sigma_schedules": [list(MANIFEST.t_list)]}
        )
        profile = config.inference_profile(MANIFEST)
        for frame in (0, 1, 9):
            self.assertEqual(profile.sigmas_for_frame(frame), tuple(MANIFEST.t_list))
        self.assertEqual(profile.max_steps, 4)

    def test_per_frame_schedules_select_by_chunk_start(self):
        config = CosmosDreamsConfig()
        # The same shape a --pipeline-config-path JSON file takes.
        config.update_pipeline_config(
            {"frame_sigma_schedules": self.STEP42, "history_mode": "sliding"}
        )
        profile = config.inference_profile(MANIFEST)
        self.assertEqual(len(profile.sigmas_for_frame(0)), 4)
        for frame in (1, 2, 3, 225):
            self.assertEqual(profile.sigmas_for_frame(frame), (1.0, 0.8333333333333334))
        self.assertEqual(profile.max_steps, 4)
        self.assertEqual(profile.window_frames, MANIFEST.window_frames)
        self.assertEqual(profile.history_mode, "sliding")
        with self.assertRaises(ValueError):
            profile.sigmas_for_frame(-1)

    def test_realtime_config_shares_the_profile_fields(self):
        config = CosmosDreamsRealtimeConfig()
        config.update_pipeline_config({"history_max_frames": 33})
        self.assertEqual(config.inference_profile(MANIFEST).window_frames, 9)

    def test_rejects_malformed_schedules_and_history_settings(self):
        bad_schedules = [
            [],
            [[]],
            [[0.9, 0.5]],
            [[1.0, 0.625, 0.625]],
            [[1.0, 1.5]],
            [[1.0, 0.0]],
            [[1.0, True]],
            [[1.0, 0.7]],
            [[1.0, 0.625, 0.9375]],
            "1.0, 0.5",
        ]
        for schedules in bad_schedules:
            with self.subTest(schedules=schedules), self.assertRaises(ValueError):
                resolve_inference_profile(
                    MANIFEST, frame_sigma_schedules=schedules, history_max_frames=901
                )
        with self.assertRaisesRegex(ValueError, "history_mode"):
            resolve_inference_profile(
                MANIFEST, history_mode="window", history_max_frames=901
            )
        for frames in (0, 1, 900, True, 5.0):
            with (
                self.subTest(frames=frames),
                self.assertRaisesRegex(ValueError, "history_max_frames"),
            ):
                resolve_inference_profile(MANIFEST, history_max_frames=frames)
        with self.assertRaisesRegex(ValueError, "not in the artifact"):
            resolve_inference_profile(
                MANIFEST, frame_sigma_schedules=[[1.0, 0.7]], history_max_frames=901
            )
        # Sliding mode ignores the cap entirely.
        profile = resolve_inference_profile(
            MANIFEST, history_mode="sliding", history_max_frames=900
        )
        self.assertEqual(profile.window_frames, MANIFEST.window_frames)


class TestCosmosDreamsRegistry(unittest.TestCase):
    def test_pipeline_and_transformer_are_discoverable(self):
        _discover_and_register_pipelines()
        self.assertIn("CosmosDreamsPipeline", _PIPELINE_REGISTRY)
        # The release export declares the model-card class name.
        self.assertTrue(
            issubclass(
                _PIPELINE_REGISTRY["Cosmos3NanoSimBimanualPipeline"],
                _PIPELINE_REGISTRY["CosmosDreamsPipeline"],
            )
        )
        self.assertIn("CosmosDreamsTransformer", ModelRegistry.get_supported_archs())

    def test_release_path_resolves_to_dreams_and_nano_stays_cosmos3(self):
        """The Dreams release name contains "Cosmos3-Nano"; the longer registered
        path must win the partial match without stealing plain Nano paths."""
        for model_path in (
            "nvidia/Cosmos3-Nano-Sim-Bimanual",
            "/models/Cosmos3-Nano-Sim-Bimanual",
        ):
            with self.subTest(model_path=model_path):
                config_info = _get_config_info(model_path)
                self.assertIs(config_info.pipeline_config_cls, CosmosDreamsConfig)
                self.assertIs(
                    config_info.sampling_param_cls, CosmosDreamsSamplingParams
                )
        config_info = _get_config_info("nvidia/Cosmos3-Nano")
        self.assertIs(config_info.pipeline_config_cls, Cosmos3Config)
        self.assertIs(config_info.sampling_param_cls, Cosmos3SamplingParams)

    def test_class_name_detector_matches_dreams_checkpoints(self):
        for class_name in ("CosmosDreamsPipeline", "Cosmos3NanoSimBimanualPipeline"):
            with mock.patch(
                "sglang.multimodal_gen.registry.maybe_download_model_index",
                return_value={"_class_name": class_name},
            ):
                config_info = _get_config_info("acme/renamed-interactive-ckpt")
            self.assertIs(config_info.pipeline_config_cls, CosmosDreamsConfig)

    def test_config_swaps_transformer_and_prompt_templates(self):
        config = CosmosDreamsConfig()
        self.assertEqual(config.transformer_class_override, "CosmosDreamsTransformer")
        self.assertFalse(config.use_duration_template)
        self.assertFalse(config.supports_action_endpoint())
        self.assertIsNone(Cosmos3Config().transformer_class_override)


class TestCosmosDreamsSamplingParams(unittest.TestCase):
    def test_rejects_unsupported_request_modes(self):
        for kwargs in (
            {"video_path": "clip.mp4"},
            {"control_path": "edge.mp4"},
            {"sound_duration": 1.0},
            {"action_mode": "policy"},
            {"num_frames": 1},
            {"image_path": ["a.png", "b.png"]},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CosmosDreamsSamplingParams(prompt="drive forward", **kwargs)
        params = CosmosDreamsSamplingParams(
            prompt="drive forward", action_mode="forward_dynamics"
        )
        self.assertEqual(params.fps, 24)
        self.assertEqual(params.num_frames, 81)
        self.assertEqual(params.supported_resolutions[0], (832, 480))
        self.assertTrue(params.format_prompt_as_json)
        self.assertFalse(params.canvas_from_image)
        self.assertEqual(CosmosDreamsConfig().canvas_tier, "480")

    def test_distilled_checkpoints_never_use_cfg_parallel(self):
        deployment = CosmosDreamsConfig().get_model_deployment_config()
        self.assertFalse(deployment.auto_enable_cfg_parallel)
        self.assertFalse(deployment.supports_cfg_parallel)
        self.assertIn("dit", deployment.keep_resident_components)
        # The launcher's auto-CFG heuristic reads the default params.
        self.assertEqual(CosmosDreamsSamplingParams().guidance_scale, 1.0)

    def test_frame_rounding_never_yields_a_single_frame(self):
        config = CosmosDreamsConfig()
        self.assertEqual([config.adjust_num_frames(n) for n in (1, 2, 4)], [5, 5, 5])
        self.assertEqual(
            [config.adjust_num_frames(n) for n in (5, 9, 61, 83)], [5, 9, 61, 81]
        )
        # The server warmup's residency ladder halves frames through this contract.
        server_args = SimpleNamespace(pipeline_config=config)
        self.assertEqual(_lighter_valid_num_frames(server_args, 17), 9)
        self.assertEqual(_lighter_valid_num_frames(server_args, 9), 5)
        self.assertEqual(_lighter_valid_num_frames(server_args, 5), 5)

    def test_sequence_parallel_layouts_are_rejected_with_a_hint(self):
        # `--num-gpus 2` alone resolves to sp_degree=2 in the launcher.
        for config in (CosmosDreamsConfig(), CosmosDreamsRealtimeConfig()):
            config.validate_server_args(SimpleNamespace(sp_degree=1, num_gpus=1))
            with self.assertRaisesRegex(ValueError, "--dp-size"):
                config.validate_server_args(SimpleNamespace(sp_degree=2, num_gpus=2))


class TestCosmosDreamsPromptAndCanvas(unittest.TestCase):
    def test_prompt_matches_training_json_caption(self):
        """The checkpoint trained on ``wrap_dataset(format_prompt_as_json=True,
        append_duration_fps_timestamps=False)`` captions: framing, description,
        resolution and aspect ratio with default ``json.dumps`` separators and
        no duration/fps fields."""
        prompt = format_dreams_prompt(
            "A robot arm pushes a box", view_point="ego_view", height=480, width=832
        )
        self.assertEqual(
            prompt,
            '{"cinematography": {"framing": "This video is captured from a first-person '
            'perspective looking at the scene."}, "actions": [{"description": "A robot arm '
            'pushes a box."}], "resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9"}',
        )
        self.assertEqual(
            format_dreams_prompt("  ", view_point="ego_view", height=480, width=832), ""
        )
        structured = json.loads(
            format_dreams_prompt(
                json.dumps(
                    {
                        "cinematography": {"framing": "Custom."},
                        "actions": [{"description": "Turn left."}],
                    }
                ),
                view_point="ego_view",
                height=480,
                width=832,
            )
        )
        self.assertEqual(structured["cinematography"], {"framing": "Custom."})
        self.assertEqual(structured["resolution"], {"H": 480, "W": 832})
        self.assertEqual(structured["aspect_ratio"], "16,9")
        self.assertIn(
            '"aspect_ratio": "13,10"',
            format_dreams_prompt("x", view_point="ego_view", height=640, width=832),
        )
        with self.assertRaises(ValueError):
            format_dreams_prompt("x", view_point="drone_view", height=480, width=832)

    def test_closest_canvas_snaps_by_aspect(self):
        self.assertEqual(closest_canvas(height=360, width=640, tier="480"), (832, 480))
        self.assertEqual(closest_canvas(height=640, width=360, tier="480"), (480, 832))
        self.assertEqual(closest_canvas(height=500, width=500, tier="480"), (640, 640))
        self.assertEqual(closest_canvas(height=600, width=800, tier="480"), (736, 544))
        self.assertEqual(closest_canvas(height=800, width=600, tier="480"), (544, 736))
        with self.assertRaises(ValueError):
            closest_canvas(height=1, width=1, tier="1080p")

    def test_fit_image_to_canvas_pads_bottom_right_like_training(self):
        # Smaller frames are padded at native size, never upscaled (training
        # caps the scale at 1.0); the AgiBot clips are 640x360 in an 832x480 canvas.
        image = torch.rand(3, 360, 640)
        fitted, content = fit_image_to_canvas(
            image, target_height=480, target_width=832
        )
        self.assertEqual(tuple(fitted.shape), (3, 480, 832))
        self.assertEqual(content, (360, 640))
        torch.testing.assert_close(fitted[:, :360, :640], image)
        # Reflection padding mirrors the rows just above the content edge.
        torch.testing.assert_close(
            fitted[:, 360:480, :640], torch.flip(fitted[:, 239:359, :640], dims=[1])
        )
        # Larger frames are downscaled to fit: the camera_stone parity geometry.
        _, content = fit_image_to_canvas(
            torch.rand(3, 980, 1596), target_height=480, target_width=832
        )
        self.assertEqual(content, (480, 782))
        # A pad at least as large as the content falls back to edge replication.
        fitted, content = fit_image_to_canvas(
            torch.rand(3, 2, 2), target_height=8, target_width=20
        )
        self.assertEqual(content, (2, 2))
        torch.testing.assert_close(
            fitted[:, :2, 2:], fitted[:, :2, 1:2].expand(-1, -1, 18)
        )
        torch.testing.assert_close(
            fitted[:, 2:, :], fitted[:, 1:2, :].expand(-1, 6, -1)
        )


if __name__ == "__main__":
    unittest.main()
