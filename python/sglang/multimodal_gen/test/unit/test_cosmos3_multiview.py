# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Cosmos3 Multiview-AV port: visibility predicate against
the spec truth table, run-compressed block sparsity, padded flex attention
against a dense masked GQA oracle, camera-major helpers, wrapped temporal
positions, deployment-contract parsing, request validation, and registry
wiring."""

import copy
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MADS_CAMERAS,
    Cosmos3MultiviewConfig,
    parse_multiview_deployment_config,
    validate_lidar_config,
)
from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
    Cosmos3MultiviewSamplingParams,
    clamp_multiview_guidance_scale,
    closest_multiview_aspect_ratio,
    normalize_multiview_aspect_ratio,
    parse_local_condition_indexes,
    validate_lidar_request,
    validate_multiview_request,
)
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
)
from sglang.multimodal_gen.runtime.models.dits import (
    cosmos3_multiview_maskless as maskless_module,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    pack_state,
    unpack_state,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_layout import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    expand_multiview_condition_frame_indexes,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_maskless import (
    MASKLESS_KERNEL_ENV_VAR,
    build_multiview_maskless_plan,
    maskless_unavailable_reason,
    merge_attentions,
    multiview_maskless_attention,
    resolve_maskless_kernel,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    compute_mrope_position_ids_vision,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar_decoder import (
    _DepthToPixels,
    crop_lidar_width,
    depth_to_space,
    lidar_network_to_metric,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar_encoder import (
    Cosmos3LidarEncoder,
    _SpaceToDepth,
    pad_lidar_sweeps,
    required_lidar_sweeps,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_lidar_outputs import (
    lidar_output_payload,
    lidar_payload_for_response,
    pool_lidar_azimuth,
    render_lidar_bev_frames,
    render_lidar_range_frames,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_multiview import (
    AV_SYSTEM_PROMPTS_BY_VARIANT,
    COSMOS3_MULTIVIEW_EMPHASIS,
    Cosmos3MultiviewInputStage,
    fit_uint8_cthw,
    format_multiview_prompts,
    format_per_view_prompts,
    format_separate_view_captions,
    media_kind,
    pad_view_frames_uint8,
    synthetic_multiview_pixels,
)

# ``transformer/config.json["multiview"]`` of the Sep-14 2026 export (HF 3e7d669): masked
# Triton backend, unversioned. Kept only to pin that such exports are refused.
LEGACY_BLOCK = {
    "align_temporal_positions_across_views": True,
    "attention_scope": "decomposed",
    "backend": "triton",
    "cameras": list(COSMOS3_MADS_CAMERAS),
    "causal_training_strategy": "none",
    "control_attends_sensor": True,
    "decomposed_temporal_window_seconds": None,
    "max_views": 11,
    "share_vision_temporal_positions": True,
}


# The V1.2 LiDAR contract of the joint export (transformer/config.json multiview.lidar).
LIDAR_BLOCK = {
    "apply_validity_mask": True,
    "dtype": "float32",
    "fps": 10.0,
    "latent_channels": 128,
    "network_config": {
        "base_channels": 128,
        "bottleneck_3d": True,
        "bottleneck_3d_causal_time": True,
        "bottleneck_3d_max_t": 32,
        "bottleneck_3d_rope": True,
        "depths": [3, 3, 3, 3],
        "dilation": [1, 1, 1, 1],
        "formulation": "VAE",
        "in_channels": 3,
        "mapping_depth": 2,
        "mask_as_input": False,
        "mlp_ratio": 3.0,
        "num_heads": [4, 4, 8, 8],
        "out_channels": 3,
        "patch_size": [2, 2],
        "positional_embedding": "learnable_embedding",
        "resolution": [128, 1808],
        "temporal_downsample": [False, False, False],
        "temporal_upsample": [False, False, False],
        "window_size": [5, 45],
        "z_dim": 128,
    },
    "range_projection": {
        "azimuth_end_degrees": -180.0,
        "azimuth_endpoint": False,
        "azimuth_start_degrees": 180.0,
        "coordinate_system": "x_forward_y_left_z_up",
        "intensity_encoding": "unit",
        "invalid_range_m": 0.0,
        "max_range_m": 100.0,
        "min_range_m": 5.0,
        "model_width": 1808,
        "model_width_transform": "circular_pad",
        "native_height": 128,
        "native_width": 3600,
        "return_selection": "nearest",
        "semantic_height": 128,
        "semantic_width": 1800,
        "sensor": "pandar128",
        "validity_threshold": 0.5,
    },
    "sample_posterior": False,
    "spatial_compression": [16, 16],
    "streaming_chunk_frames": 9,
    "streaming_context_frames": 9,
    "temporal_compression_factor": 1,
    "version": "1.2",
}
INFERENCE_DEFAULTS = {
    "control_guidance": 1.0,
    "control_guidance_interval": None,
    "emphasize_control_in_prompt": True,
    "fps": 30.0,
    "guidance": 6.0,
    "guidance_interval": None,
    "negative_metadata_mode": "none",
    "normalize_cfg": False,
    "num_steps": 35,
    "resolution": "480",
    "shift": 10.0,
    "sigma_max": 80.0,
}
# ``transformer/config.json["multiview"]`` of the maskless joint export (HF 75f2199 / e0496d5).
SCHEMA2_BLOCK = {
    **LEGACY_BLOCK,
    "backend": "maskless",
    "decomposed_temporal_window_seconds": None,
    "schema_version": 2,
    "separate_view_text_tokenization": True,
    "variable_view_count": True,
    "inference_defaults": INFERENCE_DEFAULTS,
    "lidar": LIDAR_BLOCK,
    "lidar_attends_captions": True,
}
DEPLOYMENT_BLOCK = SCHEMA2_BLOCK


def _adjust_multiview(params, deployment):
    """The multiview-specific half of ``_adjust``; the base adjustment needs live ServerArgs."""
    params._apply_deployment_defaults(deployment)
    params._resolve_canvas(deployment)
    params._apply_guidance_policy(deployment)
    return params


def _transformer_config(multiview=None, backbone_type="cosmos3_multiview"):
    return {
        "backbone_type": backbone_type,
        "multiview": copy.deepcopy(
            DEPLOYMENT_BLOCK if multiview is None else multiview
        ),
    }


class TestLayoutHelpers(unittest.TestCase):
    def test_expand_condition_indexes_camera_major(self):
        self.assertEqual(
            expand_multiview_condition_frame_indexes([0], 11, 11 * 24),
            [view * 24 for view in range(11)],
        )
        self.assertEqual(
            expand_multiview_condition_frame_indexes([1, 0, 99], 2, 6), [0, 1, 3, 4]
        )
        self.assertEqual(expand_multiview_condition_frame_indexes(None, 2, 6), [])
        with self.assertRaises(ValueError):
            expand_multiview_condition_frame_indexes([0], 2, 5)

    def test_layout_geometry_and_validation(self):
        layout = MultiviewLayout(
            num_views=11, latent_frames=264, patch_height=15, patch_width=26
        )
        self.assertEqual(layout.frames_per_view, 24)
        self.assertEqual(layout.item_tokens, 102_960)
        self.assertEqual(layout.gen_tokens, 205_920)
        with self.assertRaisesRegex(ValueError, "divisible by num_views"):
            MultiviewLayout(
                num_views=11, latent_frames=263, patch_height=15, patch_width=26
            )
        with self.assertRaisesRegex(ValueError, "attention_scope"):
            MultiviewLayout(
                num_views=1,
                latent_frames=1,
                patch_height=1,
                patch_width=1,
                attention_scope="all",
            )

    def test_temporal_position_period_wraps_camera_major_frames(self):
        ids, _ = compute_mrope_position_ids_vision(
            6,
            1,
            1,
            temporal_offset=0,
            device=torch.device("cpu"),
            temporal_position_period=3,
        )
        self.assertEqual(ids[0].tolist(), [0, 1, 2, 0, 1, 2])
        ids_fps, _ = compute_mrope_position_ids_vision(
            6,
            1,
            1,
            temporal_offset=0,
            device=torch.device("cpu"),
            fps=30.0,
            base_fps=24.0,
            temporal_compression_factor=4,
            temporal_position_period=3,
        )
        torch.testing.assert_close(
            ids_fps[0], torch.tensor([0.0, 0.8, 1.6, 0.0, 0.8, 1.6])
        )
        plain, _ = compute_mrope_position_ids_vision(
            6, 1, 1, temporal_offset=0, device=torch.device("cpu")
        )
        self.assertEqual(plain[0].tolist(), [0, 1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_mrope_position_ids_vision(
                6,
                1,
                1,
                temporal_offset=0,
                device=torch.device("cpu"),
                temporal_position_period=0,
            )


class TestPixelHelpersAndPrompt(unittest.TestCase):
    def test_fisheye_canvas_crop_rounds_half_to_even(self):
        """1720x1080 fisheye sources resize to 523 rows for a 480x832 canvas; the
        21.5-row crop offset must round to 22 like imaginaire4. A floor offset
        shifted every fisheye anchor by one row (28 dB instead of 45 dB parity)."""
        ramp = (
            torch.arange(1080, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .view(1, 1, 1080, 1)
            .expand(3, 1, 1080, 1720)
        )

        actual = fit_uint8_cthw(ramp, height=480, width=832)

        resized = torch.nn.functional.interpolate(
            ramp.permute(1, 0, 2, 3).float(),
            size=(523, 832),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        expected = resized[0, :, 22:502].round().clamp(0, 255).to(torch.uint8)
        self.assertEqual(tuple(actual.shape), (3, 1, 480, 832))
        self.assertTrue(torch.equal(actual[:, 0], expected))
        self.assertFalse(
            torch.equal(actual[:, 0], resized[0, :, 21:501].round().to(torch.uint8))
        )

    def test_pad_view_frames_truncates_or_repeats_last_frame(self):
        frames = torch.stack(
            [torch.full((3, 2, 3), value, dtype=torch.uint8) for value in (10, 20)],
            dim=1,
        )
        padded = pad_view_frames_uint8(frames, num_frames=5)
        self.assertEqual(tuple(padded.shape), (3, 5, 2, 3))
        self.assertEqual(padded[:, 0].unique().tolist(), [10])
        self.assertEqual(padded[:, 1:].unique().tolist(), [20])
        truncated = pad_view_frames_uint8(frames, num_frames=1)
        self.assertEqual(truncated.unique().tolist(), [10])
        with self.assertRaisesRegex(ValueError, "zero frames"):
            pad_view_frames_uint8(frames[:, :0], num_frames=3)

    def test_synthetic_pixels_and_media_kind(self):
        pixels = synthetic_multiview_pixels(
            num_views=3, num_frames=5, height=4, width=6
        )
        self.assertEqual(tuple(pixels.shape), (1, 3, 15, 4, 6))
        self.assertEqual(pixels.dtype, torch.uint8)
        self.assertEqual(media_kind("a/front.PNG"), "image")
        self.assertEqual(media_kind("a/front.mp4"), "video")

    def test_prose_prompt_gets_metadata_and_emphasis(self):
        positive, negative = format_multiview_prompts(
            "A rainy street.",
            "",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
        )
        self.assertEqual(
            positive,
            "A rainy street. The video is 3.1 seconds long and is of 30 FPS. "
            "This video is of 480x832 resolution. " + COSMOS3_MULTIVIEW_EMPHASIS,
        )
        self.assertEqual(
            negative,
            "The video is 3.1 seconds long and is of 30 FPS. This video is of 480x832 resolution.",
        )
        _, inverse = format_multiview_prompts(
            "x",
            "bad",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
            negative_metadata_mode="inverse",
        )
        self.assertIn("is not 3.1 seconds long", inverse)
        _, none = format_multiview_prompts(
            "x",
            "bad",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
            negative_metadata_mode="none",
        )
        self.assertEqual(none, "bad")

    def test_json_prompt_gets_metadata_fields(self):
        import json

        positive, _ = format_multiview_prompts(
            json.dumps({"description": "a bus"}),
            "",
            num_frames=93,
            fps=30.0,
            height=480,
            width=832,
        )
        json_part, _, suffix = positive.partition("} ")
        caption = json.loads(json_part + "}")
        self.assertEqual(caption["duration"], "3s")
        self.assertEqual(caption["fps"], 30.0)
        self.assertEqual(caption["resolution"], {"H": 480, "W": 832})
        self.assertEqual(caption["aspect_ratio"], "16,9")
        self.assertEqual(suffix, COSMOS3_MULTIVIEW_EMPHASIS)


class TestDeploymentConfig(unittest.TestCase):
    def test_accepts_the_maskless_contract(self):
        # The Sep-18 export: backend maskless, no temporal window, LiDAR reads captions.
        block = {
            **SCHEMA2_BLOCK,
            "backend": "maskless",
            "decomposed_temporal_window_seconds": None,
            "lidar_attends_captions": True,
        }
        deployment = parse_multiview_deployment_config(_transformer_config(block))
        self.assertEqual(deployment.backend, "maskless")
        self.assertTrue(deployment.lidar_attends_captions)
        # Every maskless export trained under the Sep-15 prompt wording; only an
        # export that records another variant changes it.
        self.assertEqual(deployment.system_prompt_variant, "wsm_controls")
        self.assertEqual(
            parse_multiview_deployment_config(
                _transformer_config(
                    {**block, "system_prompt_variant": "provided_controls"}
                )
            ).system_prompt_variant,
            "provided_controls",
        )
        with self.assertRaisesRegex(ValueError, "system_prompt_variant"):
            parse_multiview_deployment_config(
                _transformer_config({**block, "system_prompt_variant": "latest"})
            )
        block["lidar_attends_captions"] = False
        self.assertFalse(
            parse_multiview_deployment_config(
                _transformer_config(block)
            ).lidar_attends_captions
        )
        self.assertFalse(
            parse_multiview_deployment_config(
                _transformer_config({**block, "control_attends_sensor": False})
            ).control_attends_sensor
        )
        # The Sep-22 export renamed the caption-layout key; both spellings parse and
        # a disagreement between them is an error.
        renamed = {
            k: v for k, v in block.items() if k != "separate_view_text_tokenization"
        }
        renamed["per_view_captions"] = True
        self.assertTrue(
            parse_multiview_deployment_config(
                _transformer_config(renamed)
            ).per_view_captions
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            parse_multiview_deployment_config(
                _transformer_config({**block, "per_view_captions": False})
            )
        with self.assertRaisesRegex(ValueError, "per_view_captions"):
            parse_multiview_deployment_config(
                _transformer_config(
                    {
                        k: v
                        for k, v in block.items()
                        if k != "separate_view_text_tokenization"
                    }
                )
            )
        for field, value, message in (
            ("decomposed_temporal_window_seconds", 0.4, "temporal window"),
            ("attention_scope", "all_views", "all_views"),
            ("lidar_attends_captions", "yes", "boolean"),
            ("schema_version", None, "schema_version=2"),
        ):
            with self.subTest(field=field):
                bad = {**block, field: value}
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    parse_multiview_deployment_config(_transformer_config(bad))

    def test_refuses_masked_exports_with_the_revision_hint(self):
        # Exports trained under the masked FlexAttention backends (the Sep-14
        # revision and earlier) are a different attention this build no longer
        # implements; the error names the first maskless revision.
        with self.assertRaisesRegex(ValueError, "75f2199"):
            parse_multiview_deployment_config(_transformer_config(LEGACY_BLOCK))
        for backend in ("triton", "fa4"):
            with self.subTest(backend=backend):
                block = {
                    **SCHEMA2_BLOCK,
                    "backend": backend,
                    "decomposed_temporal_window_seconds": 0.4,
                }
                with self.assertRaisesRegex(ValueError, "75f2199"):
                    parse_multiview_deployment_config(_transformer_config(block))

    def test_accepts_the_schema2_joint_contract(self):
        deployment = parse_multiview_deployment_config(
            _transformer_config(SCHEMA2_BLOCK)
        )
        self.assertEqual(deployment.schema_version, 2)
        self.assertFalse(deployment.is_legacy)
        self.assertTrue(deployment.per_view_captions)
        self.assertTrue(deployment.variable_view_count)
        self.assertTrue(deployment.supports_lidar)
        self.assertIsNone(deployment.decomposed_temporal_window_seconds)
        self.assertEqual(deployment.cameras, COSMOS3_MADS_CAMERAS)
        self.assertTrue(deployment.control_attends_sensor)
        self.assertTrue(deployment.align_temporal_positions_across_views)
        self.assertEqual(deployment.inference_default("num_steps", 1), 35)
        self.assertEqual(deployment.inference_default("resolution", "720"), "480")
        self.assertEqual(deployment.inference_default("guidance_interval", "x"), "x")
        self.assertEqual(deployment.lidar["fps"], 10.0)
        self.assertTrue(deployment.lidar_attends_captions)
        # Schema 2 may reorder or subset the rig; legacy exports may not.
        block = copy.deepcopy(SCHEMA2_BLOCK)
        block["cameras"] = list(reversed(COSMOS3_MADS_CAMERAS))
        self.assertEqual(
            parse_multiview_deployment_config(_transformer_config(block)).cameras,
            tuple(reversed(COSMOS3_MADS_CAMERAS)),
        )
        for field, value, message in (
            ("schema_version", 3, "schema_version"),
            ("inference_defaults", {"resolution": "480"}, "inference_defaults"),
            ("separate_view_text_tokenization", "yes", "boolean"),
        ):
            with self.subTest(field=field):
                block = copy.deepcopy(SCHEMA2_BLOCK)
                block[field] = value
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    parse_multiview_deployment_config(_transformer_config(block))
        block = copy.deepcopy(LEGACY_BLOCK)
        block["lidar"] = LIDAR_BLOCK
        with self.assertRaisesRegex(ValueError, "schema_version=2"):
            parse_multiview_deployment_config(_transformer_config(block))

    def test_lidar_contract_validation(self):
        self.assertEqual(validate_lidar_config(LIDAR_BLOCK)["version"], "1.2")
        for path, value, message in (
            (("version",), "1.1", "V1.2"),
            (("range_projection", "model_width"), 1800, "circularly padded"),
            (("network_config", "patch_size"), [2, 4], "disagree"),
            (("streaming_context_frames",), 4, "chunk length"),
        ):
            with self.subTest(path=path):
                block = copy.deepcopy(LIDAR_BLOCK)
                target = block
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_lidar_config(block)

    def test_rejects_wrong_backbone_before_multiview_fields(self):
        with self.assertRaisesRegex(ValueError, "backbone_type"):
            parse_multiview_deployment_config(
                {"backbone_type": None, "multiview": None}
            )

    def test_rejects_malformed_fields(self):
        bad_cases = {
            "causal_training_strategy": ("teacher_forcing", ValueError),
            "attention_scope": ("everything", ValueError),
            "decomposed_temporal_window_seconds": (-1.0, ValueError),
            "control_attends_sensor": ("yes", TypeError),
            "share_vision_temporal_positions": (False, ValueError),
            "max_views": (10, ValueError),
            "backend": ("cuda", ValueError),
        }
        for field, (value, error) in bad_cases.items():
            with self.subTest(field=field):
                block = copy.deepcopy(SCHEMA2_BLOCK)
                block[field] = value
                with self.assertRaises(error):
                    parse_multiview_deployment_config(_transformer_config(block))
        # Every field of the original contract is still required.
        for field in LEGACY_BLOCK:
            with self.subTest(missing=field):
                block = copy.deepcopy(SCHEMA2_BLOCK)
                del block[field]
                with self.assertRaises(ValueError):
                    parse_multiview_deployment_config(_transformer_config(block))

    def test_config_flags_and_parallelism_limits(self):
        config = Cosmos3MultiviewConfig()
        self.assertEqual(
            config.transformer_class_override, "Cosmos3MultiviewTransformer"
        )
        self.assertTrue(config.use_system_prompt)
        self.assertFalse(config.supports_action_endpoint())
        self.assertFalse(config.supports_dynamic_batching())
        with (
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview._transformer_config",
                return_value=_transformer_config(),
            ),
            mock.patch(
                "sglang.multimodal_gen.configs.pipeline_configs.cosmos3.get_distilled_sigmas",
                return_value=None,
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
            config.update_config_from_dict({"model_path": "/models/mv"})
        self.assertEqual(config.multiview_deployment.num_views, 11)
        self.assertEqual(config.multiview_deployment.backend, "maskless")
        for name in ("tp_size", "sp_degree", "ulysses_degree", "ring_degree"):
            with self.subTest(arg=name):
                server_args = SimpleNamespace(
                    tp_size=1,
                    sp_degree=1,
                    ulysses_degree=1,
                    ring_degree=1,
                    enable_cfg_parallel=False,
                )
                setattr(server_args, name, 2)
                with self.assertRaises(ValueError):
                    config.validate_server_args(server_args)
        # CFG parallel only splits the two branches across ranks; it is allowed.
        config.validate_server_args(
            SimpleNamespace(
                tp_size=1,
                sp_degree=1,
                ulysses_degree=1,
                ring_degree=1,
                enable_cfg_parallel=True,
            )
        )


def _views(cameras, *, vision=False):
    views = []
    for index, camera in enumerate(cameras):
        view = {"camera_key": camera, "control_path": f"control_{index}.mp4"}
        if vision:
            view["vision_path"] = f"vision_{index}.png"
        views.append(view)
    return views


class TestSamplingParamsAndInputStage(unittest.TestCase):
    def test_defaults_follow_the_reference(self):
        params = Cosmos3MultiviewSamplingParams()
        # The canvas is a bucket choice made at adjustment, not a field default.
        self.assertEqual((params.width, params.height), (None, None))
        self.assertEqual(params.num_frames, 93)
        self.assertEqual(params.fps, 30)
        self.assertEqual(params.guidance_scale, 6.0)
        self.assertEqual(params.num_inference_steps, 35)
        self.assertEqual(params.negative_metadata_mode, "same")
        self.assertEqual(params.resolve_views(), [])
        for name in ("multiview", "lidar", "resolution", "aspect_ratio"):
            self.assertIn(
                name, Cosmos3MultiviewSamplingParams.video_request_extra_fields()
            )

    def test_adjust_resolves_canvas_and_checkpoint_defaults(self):
        # Without a deployment (client side) the v1 defaults and guidance clamp apply.
        params = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(guidance_scale=9.0), None
        )
        self.assertEqual((params.width, params.height), (832, 480))
        self.assertEqual(params.num_frames, 93)
        self.assertEqual(params.guidance_scale, 7.0)

        schema2 = parse_multiview_deployment_config(_transformer_config(SCHEMA2_BLOCK))
        params = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(guidance_scale=9.0), schema2
        )
        self.assertEqual((params.width, params.height), (832, 480))
        self.assertEqual(params.num_frames, 201)
        self.assertEqual(params.guidance_scale, 9.0)
        self.assertEqual(params.flow_shift, 10.0)
        self.assertEqual(params.aspect_ratio, "auto")
        portrait = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(resolution=720, aspect_ratio="9:16"), schema2
        )
        self.assertEqual((portrait.width, portrait.height), (720, 1280))
        explicit = _adjust_multiview(
            Cosmos3MultiviewSamplingParams(width=1104, height=832, resolution="720"),
            schema2,
        )
        self.assertEqual(explicit.aspect_ratio, "4,3")
        with self.assertRaisesRegex(ValueError, "requires width=1280"):
            _adjust_multiview(
                Cosmos3MultiviewSamplingParams(
                    width=832, height=480, resolution="720", aspect_ratio="16:9"
                ),
                schema2,
            )
        self.assertEqual(normalize_multiview_aspect_ratio("1920:1080"), "16,9")
        self.assertEqual(closest_multiview_aspect_ratio(1084, 1924, "480"), "16,9")
        self.assertEqual(closest_multiview_aspect_ratio(1000, 1000, "720"), "1,1")

    def test_views_from_multiview_object_and_control_path_list(self):
        params = Cosmos3MultiviewSamplingParams(
            multiview={
                "views": _views(("front", "left"), vision=True),
                "condition_video_as_image": True,
            }
        )
        views = params.resolve_views()
        self.assertEqual([view.camera_key for view in views], ["front", "left"])
        self.assertEqual(
            [view.vision for view in views], ["vision_0.png", "vision_1.png"]
        )
        self.assertTrue(params.resolved_condition_video_as_image())
        self.assertIsNone(params.resolved_local_condition_indexes())
        t2v = Cosmos3MultiviewSamplingParams(control_path=["a.mp4", "b.mp4"])
        self.assertEqual(
            [view.control for view in t2v.resolve_views()], ["a.mp4", "b.mp4"]
        )
        self.assertTrue(all(view.camera_key is None for view in t2v.resolve_views()))

    def test_request_validation(self):
        # Partial vision is view completion, admitted here and checked per checkpoint later.
        views = _views(("front", "left"))
        views[0]["vision_path"] = "front.mp4"
        self.assertEqual(len(validate_multiview_request({"views": views})), 2)
        with self.assertRaisesRegex(ValueError, "every view or none"):
            views = _views(("front", "left"))
            views[0]["prompt"] = "A car."
            validate_multiview_request({"views": views})
        with self.assertRaisesRegex(ValueError, "control_path"):
            validate_lidar_request({"control_path": "sweeps.tar"})
        self.assertEqual(
            validate_lidar_request({"control_path": "sweeps.safetensors"}),
            {"control_path": "sweeps.safetensors", "decode": False},
        )
        with self.assertRaisesRegex(ValueError, "control_path"):
            validate_multiview_request({"views": [{"camera_key": "front"}]})
        with self.assertRaisesRegex(ValueError, "Unsupported Cosmos3 multiview fields"):
            validate_multiview_request({"views": _views(("front",)), "lidar": {}})
        with self.assertRaisesRegex(ValueError, "resolution"):
            validate_multiview_request({"views": _views(("front",)), "resolution": 704})
        with self.assertRaisesRegex(ValueError, "aspect_ratio"):
            validate_multiview_request(
                {"views": _views(("front",)), "aspect_ratio": "2:1"}
            )
        with self.assertRaisesRegex(ValueError, "wsm"):
            Cosmos3MultiviewSamplingParams(wsm={"strength": 1.0})
        Cosmos3MultiviewSamplingParams(wsm={"weight": 1.0})
        with self.assertRaisesRegex(ValueError, "negative_metadata_mode"):
            Cosmos3MultiviewSamplingParams(negative_metadata_mode="sometimes")
        with self.assertRaisesRegex(ValueError, "max_sequence_length"):
            Cosmos3MultiviewSamplingParams(max_sequence_length=5000)
        self.assertEqual(parse_local_condition_indexes("1, 0"), [0, 1])
        self.assertEqual(clamp_multiview_guidance_scale(9.0), 7.0)
        self.assertEqual(clamp_multiview_guidance_scale(-1.0), 0.0)

    def test_lower_video_request_kwargs_parses_json_strings(self):
        request = SimpleNamespace(model_extra={}, num_frames=None, fps=None)
        kwargs = Cosmos3MultiviewSamplingParams.lower_video_request_kwargs(
            request,
            {
                "num_frames": 93,
                "fps": 30,
                "multiview": '{"views": [{"camera_key": "front", "control_path": "c.mp4"}]}',
                "wsm": "{}",
                "condition_video_as_image": "true",
                "negative_metadata_mode": "Same",
            },
        )
        self.assertEqual(kwargs["multiview"]["views"][0]["camera_key"], "front")
        self.assertEqual(kwargs["wsm"], {})
        self.assertIs(kwargs["condition_video_as_image"], True)
        self.assertEqual(kwargs["negative_metadata_mode"], "same")

    def test_input_stage_resolves_a_reordered_subset_with_captions(self):
        deployment = parse_multiview_deployment_config(_transformer_config())
        stage = Cosmos3MultiviewInputStage(deployment)

        def batch_for(params, is_warmup=False):
            return SimpleNamespace(sampling_params=params, is_warmup=is_warmup)

        # Schema 2 with variable_view_count admits a reordered subset but
        # insists on a caption per camera.
        subset = _views((COSMOS3_MADS_CAMERAS[3], COSMOS3_MADS_CAMERAS[0]))
        with self.assertRaisesRegex(ValueError, "prompt"):
            stage._resolve_views(
                batch_for(Cosmos3MultiviewSamplingParams(multiview={"views": subset}))
            )
        for view in subset:
            view["prompt"] = "A car."
        resolved = stage._resolve_views(
            batch_for(Cosmos3MultiviewSamplingParams(multiview={"views": subset}))
        )
        self.assertEqual(
            [view.camera_key for view in resolved],
            [COSMOS3_MADS_CAMERAS[3], COSMOS3_MADS_CAMERAS[0]],
        )
        with self.assertRaisesRegex(ValueError, "all images or all videos"):
            views = _views(COSMOS3_MADS_CAMERAS)
            for view in views:
                view["prompt"] = "A car."
            views[3]["control_path"] = "c.png"
            stage._resolve_views(
                batch_for(Cosmos3MultiviewSamplingParams(multiview={"views": views}))
            )
        self.assertIsNone(
            stage._resolve_views(
                batch_for(Cosmos3MultiviewSamplingParams(), is_warmup=True)
            )
        )
        with self.assertRaisesRegex(ValueError, "multiview.views"):
            stage._resolve_views(batch_for(Cosmos3MultiviewSamplingParams()))


class TestPerViewCaptions(unittest.TestCase):
    CAMERAS = ("camera_front_wide_120fov", "camera_rear_tele_30fov")
    RIG = (
        "This multiview driving sequence contains time-aligned recordings from 2 "
        "vehicle-mounted cameras: front wide-angle camera (forward-facing, 120\u00b0 FOV); "
        "rear telephoto camera (backward-facing, 30\u00b0 FOV)."
    )

    def test_rig_header_matches_the_training_formatter(self):
        # Golden strings from imaginaire4 caption_format.format_separate_view_captions.
        separate = format_separate_view_captions(["A car.", "Trees."], self.CAMERAS)
        self.assertEqual(
            separate[0],
            f"{self.RIG}\n\nThe description below is for the front wide-angle camera mounted "
            "on the vehicle. This camera is facing forward and has a 120\u00b0 field of "
            "view:\n\nA car.",
        )
        self.assertEqual(
            separate[1],
            f"{self.RIG}\n\nThe description below is for the rear telephoto camera mounted "
            "on the vehicle. This camera is facing backward and has a 30\u00b0 field of "
            "view:\n\nTrees.",
        )
        prompts = format_per_view_prompts(
            ["A car.", "Trees."],
            self.CAMERAS,
            num_frames=17,
            fps=30.0,
            height=480,
            width=832,
            emphasis=COSMOS3_MULTIVIEW_EMPHASIS,
        )
        # Whole seconds, as the training augmentor wrote them (17 frames -> 0.0 s).
        self.assertTrue(
            prompts[0].endswith(
                "A car. The video is 0.0 seconds long and is of 30 FPS. This video is of "
                f"480x832 resolution. {COSMOS3_MULTIVIEW_EMPHASIS}"
            )
        )

    def test_wsm_system_prompts_match_imaginaire4(self):
        # Verbatim from imaginaire4 datasets/augmentors/text_tokenizer.py after
        # commit 86041fb1b52 (Sep 15 2026), the wording the maskless export trained under.
        instruction = (
            "Follow WSM controls for vehicles (including trucks), cyclists, pedestrians, "
            "traffic lights, traffic signs, road markings, lane boundaries, and road "
            "boundaries. Do not add objects or road features in these categories that are "
            "absent from WSM. Use captions for appearance and unconstrained background "
            "details; WSM takes precedence in any conflict."
        )
        camera, joint = AV_SYSTEM_PROMPTS_BY_VARIANT["wsm_controls"]
        self.assertEqual(
            camera,
            "You are a helpful assistant that generates temporally synchronized, "
            "geometrically consistent autonomous-driving videos from per-camera scene "
            "descriptions and World Scenario Map (WSM) control videos depicting the "
            "controlled objects and road layout. Treat all camera views as simultaneous "
            "observations of the same driving scene, preserving each camera's viewpoint, "
            "shared ego motion, road layout, object identity and motion, weather, "
            f"lighting, and cross-view consistency.\n\n{instruction}",
        )
        self.assertEqual(
            joint,
            "You are a helpful assistant that jointly generates temporally synchronized, "
            "geometrically consistent autonomous-driving camera videos and LiDAR "
            "range-view sequences from per-camera scene descriptions and provided control "
            "signals: per-camera World Scenario Map (WSM) control videos depicting the "
            "controlled objects and road layout, and an HD-map control for LiDAR. Treat "
            "all camera views and LiDAR sweeps as synchronized observations of the same "
            "driving scene, preserving each camera's viewpoint, shared ego motion, road "
            "layout, object identity and motion, weather, lighting, cross-view "
            f"consistency, and camera-LiDAR alignment.\n\n{instruction}",
        )
        legacy_camera, legacy_joint = AV_SYSTEM_PROMPTS_BY_VARIANT["provided_controls"]
        self.assertIn("provided control signals.", legacy_camera)
        self.assertNotIn("WSM", legacy_camera)
        self.assertNotIn("Follow WSM controls", legacy_joint)
        with self.assertRaisesRegex(ValueError, "match the selected cameras"):
            format_separate_view_captions(["A car."], self.CAMERAS)

    def test_caption_scoping_in_the_plan(self):
        # Two cameras, one frame, one patch, packed [C0 C1 | W0 W1] with caption
        # lengths (2, 1): each camera's control and target read only that camera's caption.
        control = MaskItem(token_shape=(2, 1, 1), num_views=2, is_control=True)
        target = MaskItem(token_shape=(2, 1, 1), num_views=2)
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=2,
            patch_height=1,
            patch_width=1,
            control_attends_sensor=True,
            items=(control, target),
            caption_lengths=(2, 1),
        )
        plan = build_multiview_maskless_plan(
            layout, und_tokens=3, batch_size=1, device=torch.device("cpu")
        )
        self.assertEqual(plan.same_view_gather.tolist(), [0, 2, 1, 3])
        self.assertEqual(plan.caption_q_gather.tolist(), [0, 2, 1, 3])
        self.assertEqual(plan.caption_q_offsets.tolist(), [0, 2, 4])
        self.assertEqual(plan.caption_kv_gather.tolist(), [0, 1, 2])
        self.assertEqual(plan.caption_kv_offsets.tolist(), [0, 2, 3])
        with self.assertRaisesRegex(ValueError, "partition"):
            build_multiview_maskless_plan(
                msgspec.structs.replace(layout, caption_lengths=(2, 2)),
                und_tokens=3,
                batch_size=1,
                device=torch.device("cpu"),
            )


class TestLidarItems(unittest.TestCase):
    def test_layout_items_drive_plan_offsets_and_tokens(self):
        camera = MaskItem((2, 1, 2), 2, seconds_per_frame=0.5)
        control = MaskItem((2, 1, 2), 2, is_control=True, seconds_per_frame=0.5)
        lidar = MaskItem(
            (3, 1, 3), 1, view_offset=2, seconds_per_frame=0.1, is_lidar=True
        )
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=2,
            patch_height=1,
            patch_width=2,
            seconds_per_frame=0.5,
            control_attends_sensor=True,
            items=(control, camera, lidar),
            caption_lengths=(2, 2),
        )
        self.assertEqual(layout.gen_tokens, 4 + 4 + 9)
        default = MultiviewLayout(
            num_views=2, latent_frames=2, patch_height=1, patch_width=2
        )
        self.assertEqual([item.is_control for item in default.items], [True, False])
        self.assertNotEqual(layout.cache_key(), default.cache_key())
        plan = build_multiview_maskless_plan(
            layout, und_tokens=4, batch_size=1, device=torch.device("cpu")
        )
        # Same-view groups: each camera view's control + target (2 + 2 tokens),
        # then the LiDAR item (9 tokens); LiDAR reads both captions.
        self.assertEqual(plan.same_view_offsets.tolist(), [0, 4, 8, 17])
        self.assertEqual(plan.caption_kv_offsets.tolist(), [0, 2, 4, 8])
        self.assertEqual(plan.caption_kv_gather.tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            MultiviewLayout(
                num_views=2,
                latent_frames=2,
                patch_height=1,
                patch_width=2,
                caption_lengths=(0,),
            )

    def test_pack_unpack_roundtrip(self):
        camera = torch.arange(24.0).view(1, 2, 3, 2, 2)
        lidar = torch.arange(100.0, 118.0).view(1, 2, 3, 1, 3)
        packed = pack_state([camera, lidar])
        self.assertEqual(tuple(packed.shape), (1, 42))
        back = unpack_state(packed, (tuple(camera.shape[1:]), tuple(lidar.shape[1:])))
        torch.testing.assert_close(back[0], camera)
        torch.testing.assert_close(back[1], lidar)
        with self.assertRaisesRegex(ValueError, "declared geometries"):
            unpack_state(packed, ((2, 3, 2, 2),))

    def test_sweep_helpers_and_input_normalization(self):
        self.assertEqual(required_lidar_sweeps(17, 30.0, 10.0), 6)
        self.assertEqual(required_lidar_sweeps(201, 30.0, 10.0), 67)
        frames = torch.arange(3.0).view(1, 3, 1, 1).expand(3, 3, 1, 1).clone()
        padded = pad_lidar_sweeps(frames, 6)
        # Reflection pads [0, 1, 2] to [0, 1, 2, 2, 1], then one more reflected sweep.
        self.assertEqual(padded[0, :, 0, 0].tolist(), [0.0, 1.0, 2.0, 2.0, 1.0, 1.0])
        self.assertEqual(pad_lidar_sweeps(frames, 2).shape[1], 2)
        encoder = Cosmos3LidarEncoder(LIDAR_BLOCK)
        # Semantic-width input is circularly padded to the model width, ranges
        # normalized into [-1, 1], invalid rays filled with -1.
        clip = torch.zeros(3, 1, 128, 1800)
        clip[0, 0, 0, 0] = 52.5
        clip[1, 0, 0, 0] = 1.0
        clip[2, 0, 0, 0] = 1.0
        clip[0, 0, 1, 5] = 3.0  # below the 5 m minimum: invalid
        clip[2, 0, 1, 5] = 1.0
        prepared = encoder.prepare_input(clip)
        self.assertEqual(tuple(prepared.shape), (1, 3, 1, 128, 1808))
        self.assertAlmostEqual(prepared[0, 0, 0, 0, 4].item(), 0.0, places=5)
        self.assertEqual(prepared[0, 1, 0, 0, 4].item(), 1.0)
        self.assertEqual(prepared[0, 2, 0, 0, 4].item(), 1.0)
        self.assertEqual(prepared[0, 0, 0, 1, 9].item(), -1.0)
        self.assertEqual(prepared[0, 2, 0, 1, 9].item(), 0.0)
        # The circular pad wraps the last semantic columns in front of column 0.
        clip2 = torch.zeros(3, 1, 128, 1800)
        clip2[0, 0, 7, 1799] = 20.0
        clip2[2, 0, 7, 1799] = 1.0
        prepared2 = encoder.prepare_input(clip2)
        self.assertGreater(prepared2[0, 2, 0, 7, 3].item(), 0.5)


class TestLidarDecoder(unittest.TestCase):
    def test_network_to_metric_inverts_the_encoder_normalization(self):
        """Range is the inverse affine of the encoder's [5, 100] m -> [-1, 1] map,
        intensity the inverse of [0, 1] -> [-1, 1], and a ray whose mask logit
        falls below the 0.5 sigmoid cut reads zero range, zero intensity, validity 0."""
        torch.manual_seed(0)
        range_m = torch.rand(1, 1, 2, 4, 6) * 95.0 + 5.0
        intensity = torch.rand(1, 1, 2, 4, 6)
        valid = torch.rand(1, 1, 2, 4, 6) > 0.4
        network = torch.cat(
            (
                (range_m - 5.0) / 95.0 * 2.0 - 1.0,
                intensity * 2.0 - 1.0,
                torch.where(
                    valid, torch.full_like(range_m, 8.0), torch.full_like(range_m, -8.0)
                ),
            ),
            dim=1,
        )

        metric = lidar_network_to_metric(network, min_range_m=5.0, max_range_m=100.0)

        self.assertTrue(
            torch.allclose(
                metric[:, 0][valid[:, 0]], range_m[:, 0][valid[:, 0]], atol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                metric[:, 1][valid[:, 0]], intensity[:, 0][valid[:, 0]], atol=1e-6
            )
        )
        self.assertTrue(torch.equal(metric[:, 2], valid.float()[:, 0]))
        self.assertEqual(
            metric[:, :2][~valid.expand(-1, 2, -1, -1, -1)].abs().sum().item(), 0.0
        )
        probability = lidar_network_to_metric(
            network, min_range_m=5.0, max_range_m=100.0, apply_validity_mask=False
        )
        self.assertTrue(torch.allclose(probability[:, 2], torch.sigmoid(network[:, 2])))

    def test_pixel_shuffles_follow_the_reference_patch_order(self):
        """The decoder's expand and detokenizer lay a (p1, p2, c) channel group out as
        rows then columns, the inverse of the encoder's space-to-depth merge, so a
        checkpoint trained with einops' "(P1 P2 C)" ordering decodes on the same grid."""
        grouped = torch.arange(2 * 2 * 3 * 12, dtype=torch.float32).view(2, 2, 3, 12)

        self.assertTrue(torch.equal(_SpaceToDepth()(depth_to_space(grouped)), grouped))
        pixels = _DepthToPixels((2, 2), channels=3)(grouped)
        self.assertEqual(tuple(pixels.shape), (2, 3, 4, 6))
        # group index = p1 * 6 + p2 * 3 + c for the (row, column, channel) grouping
        for p1 in range(2):
            for p2 in range(2):
                for c in range(3):
                    self.assertEqual(
                        pixels[1, c, 1 * 2 + p1, 2 * 2 + p2].item(),
                        grouped[1, 1, 2, p1 * 6 + p2 * 3 + c].item(),
                    )
        self.assertTrue(
            torch.equal(_SpaceToDepth()(pixels.permute(0, 2, 3, 1)), grouped)
        )

    def test_crop_lidar_width_centers_the_azimuth_padding(self):
        clip = torch.arange(1808, dtype=torch.float32).view(1, 1, 1, 1, 1808)
        cropped = crop_lidar_width(clip, 1800)
        self.assertEqual(cropped.shape[-1], 1800)
        self.assertEqual(cropped[..., 0].item(), 4.0)
        with self.assertRaises(ValueError):
            crop_lidar_width(clip, 1801)

    def test_payload_and_preview_drop_invalid_rays(self):
        clip = torch.zeros(3, 2, 4, 8)
        clip[0] = 50.0
        clip[1] = 0.5
        clip[2] = 1.0
        clip[2, :, 0] = 0.0  # first beam dropped by the mask
        clip[0, :, 1] = 0.0  # second beam dropped by zero range
        frames = render_lidar_range_frames(clip, min_range_m=5.0, max_range_m=100.0)
        self.assertEqual(frames.shape, (2, 4, 8, 3))
        self.assertEqual(frames[:, :2].max(), 0)
        self.assertGreater(frames[:, 2:].max(), 0)
        payload = lidar_output_payload(
            clip,
            fps=10.0,
            min_range_m=5.0,
            max_range_m=100.0,
            files={"rangemap": "x"},
            include_arrays=True,
        )
        self.assertEqual(payload["sweeps"], 2)
        self.assertAlmostEqual(payload["valid_fraction"], 0.5)
        self.assertEqual(payload["range_m"][:, :2].max(), 0.0)
        self.assertEqual(
            set(lidar_payload_for_response(payload))
            & {"range_m", "intensity", "validity"},
            set(),
        )

    def test_video_api_accepts_an_empty_top_level_prompt(self):
        """Per-camera captions live in multiview.views[].prompt, so the multipart video
        endpoint must not reject the empty top-level prompt schema-2 requests send."""
        self.assertTrue(Cosmos3MultiviewSamplingParams.video_prompt_optional())

    def test_bev_places_rays_by_azimuth_and_range(self):
        """Column 0 of the range image is azimuth +180 (rear), the middle column is
        forward; a forward return at half the radius lands above the ego center and a
        rear one below it, and azimuth pooling keeps the nearest return of a group."""
        clip = torch.zeros(3, 1, 4, 8)
        clip[2] = 1.0
        clip[0, 0, :, 4] = 40.0  # forward (0 deg)
        clip[0, 0, :, 0] = 20.0  # rear (+180 deg)
        frames = render_lidar_bev_frames(
            clip, min_range_m=5.0, max_range_m=100.0, size_px=64, radius_m=80.0
        )
        self.assertEqual(frames.shape, (1, 64, 64, 3))
        center = 32
        self.assertTrue(frames[0, center - 16, center].any())  # forward, 40 of 80 m
        self.assertTrue(frames[0, center + 8, center].any())  # rear, 20 m
        self.assertFalse(frames[0, center, center].any())
        pooled = pool_lidar_azimuth(clip, 2)
        self.assertEqual(tuple(pooled.shape), (3, 1, 4, 4))
        self.assertEqual(pooled[0, 0, 0, 0].item(), 20.0)  # columns 0,1: nearest kept
        self.assertEqual(pooled[0, 0, 0, 2].item(), 40.0)  # columns 4,5
        self.assertEqual(pooled[0, 0, 0, 1].item(), 0.0)  # columns 2,3: no return

    def test_lidar_request_decode_flag(self):
        """LiDAR output is opt-in: a joint request returns only the camera video
        unless it sets decode true."""
        params = validate_lidar_request({"control_path": "/x/hdmap.safetensors"})
        self.assertFalse(params["decode"])
        params = validate_lidar_request(
            {"control_path": "/x/hdmap.safetensors", "decode": True}
        )
        self.assertTrue(params["decode"])
        with self.assertRaises(ValueError):
            validate_lidar_request(
                {"control_path": "/x/hdmap.safetensors", "decode": "no"}
            )
        with self.assertRaises(ValueError):
            validate_lidar_request(
                {"control_path": "/x/hdmap.safetensors", "outputs": []}
            )


def _maskless_token_table(layout):
    """Per GEN token: (axis, view, is_control, instant or None) straight from the items."""
    rows = []
    anchor = layout.items[0].seconds_per_frame
    for item in layout.items:
        frames = item.token_shape[0] // item.num_views
        spatial = item.token_shape[1] * item.token_shape[2]
        for view in range(item.num_views):
            for frame in range(frames):
                instant = int((frame + 0.5) * item.seconds_per_frame / anchor + 1e-6)
                for _ in range(spatial):
                    rows.append(
                        (1 if item.is_lidar else 0, view, item.is_control, instant)
                    )
    return rows


def _maskless_oracle(q, k, v, k_und, v_und, layout, *, count_once=False):
    """Softmax over the concatenated key multiset of the three passes.

    ``count_once`` collapses the multiset to a set, i.e. the single-mask
    attention of the masked exports; the folds deliberately do not compute it.
    """
    rows = _maskless_token_table(layout)
    gen, und = len(rows), k_und.shape[1]
    single_group = len({row[0] for row in rows}) == 1 and all(
        item.num_views == 1 for item in layout.items
    )
    fold_instants = layout.attention_scope == "decomposed" and not single_group
    counts = torch.zeros(gen, und + gen)
    starts = [0]
    for length in layout.caption_lengths:
        starts.append(starts[-1] + length)
    for qi, (q_axis, q_view, q_control, q_instant) in enumerate(rows):
        for ki, (k_axis, k_view, k_control, k_instant) in enumerate(rows):
            if (q_axis, q_view) == (k_axis, k_view) and (
                layout.control_attends_sensor or not q_control or k_control
            ):
                counts[qi, und + ki] += 1
            if (
                fold_instants
                and not q_control
                and not k_control
                and q_instant == k_instant
            ):
                counts[qi, und + ki] += 1
        if q_axis == 1 and not layout.lidar_attends_captions:
            continue
        if len(layout.caption_lengths) > 1 and q_axis == 0:
            counts[qi, starts[q_view] : starts[q_view + 1]] += 1
        else:
            counts[qi, :und] += 1
    if count_once:
        counts = counts.clamp(max=1)
    group = q.shape[2] // k.shape[2]
    keys = torch.cat([k_und, k], dim=1).float().repeat_interleave(group, dim=2)
    values = torch.cat([v_und, v], dim=1).float().repeat_interleave(group, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), keys) / (q.shape[-1] ** 0.5)
    weights = counts[None, None] * scores.exp()
    probs = weights / weights.sum(dim=-1, keepdim=True)
    return torch.einsum("bhqk,bkhd->bqhd", probs, values)


class TestMasklessAttention(unittest.TestCase):
    """Backend 'maskless': three unmasked folds merged by log-sum-exp (Sep-18 export)."""

    def _joint_layout(self, **overrides):
        camera = (4, 1, 2)  # 2 views x 2 frames at 0.4 s
        lidar = (8, 1, 1)  # 8 sweeps at 0.1 s
        items = (
            MaskItem(camera, 2, is_control=True, seconds_per_frame=0.4),
            MaskItem(camera, 2, seconds_per_frame=0.4),
            MaskItem(
                lidar,
                1,
                view_offset=2,
                is_control=True,
                seconds_per_frame=0.1,
                is_lidar=True,
            ),
            MaskItem(lidar, 1, view_offset=2, seconds_per_frame=0.1, is_lidar=True),
        )
        kwargs = dict(
            num_views=2,
            latent_frames=4,
            patch_height=1,
            patch_width=2,
            control_attends_sensor=True,
            seconds_per_frame=0.4,
            items=items,
            caption_lengths=(3, 2),
        )
        kwargs.update(overrides)
        return MultiviewLayout(**kwargs)

    def test_unavailable_reasons(self):
        ok = dict(
            attention_scope="decomposed",
            decomposed_temporal_window_seconds=None,
            control_attends_sensor=True,
        )
        self.assertIsNone(maskless_unavailable_reason(**ok))
        self.assertIn(
            "temporal window",
            maskless_unavailable_reason(
                **{**ok, "decomposed_temporal_window_seconds": 0.4}
            ),
        )
        self.assertIn(
            "all_views",
            maskless_unavailable_reason(**{**ok, "attention_scope": "all_views"}),
        )
        # control_attends_sensor is expressed by the split same-view pass, not refused.
        self.assertIsNone(
            maskless_unavailable_reason(**{**ok, "control_attends_sensor": False})
        )
        with self.assertRaisesRegex(ValueError, "temporal window"):
            build_multiview_maskless_plan(
                self._joint_layout(decomposed_temporal_window_seconds=0.4),
                und_tokens=5,
                batch_size=1,
                device=torch.device("cpu"),
            )

    def test_plan_partitions_joint_layout(self):
        layout = self._joint_layout()
        plan = build_multiview_maskless_plan(
            layout, und_tokens=5, batch_size=1, device=torch.device("cpu")
        )
        # Same view: control and target of one view share a group; LiDAR is its
        # own group. Camera view: 2 items x 2 frames x 2 spatial = 8 tokens.
        self.assertEqual(plan.same_view_offsets.tolist(), [0, 8, 16, 32])
        gather = plan.same_view_gather.tolist()
        self.assertEqual(gather[:8], [0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual(gather[16:], list(range(16, 32)))
        # Cross instant: sensor tokens only. Camera frame f covers sweeps
        # 4f..4f+3, so each instant holds 2 views x 2 spatial + 4 sweeps = 8.
        self.assertEqual(plan.cross_view_offsets.tolist(), [0, 8, 16])
        cross = plan.cross_view_gather.tolist()
        self.assertEqual(sorted(cross[:8]), [8, 9, 12, 13, 24, 25, 26, 27])
        self.assertEqual(sorted(cross[8:]), [10, 11, 14, 15, 28, 29, 30, 31])
        self.assertFalse(any(0 <= index < 8 or 16 <= index < 24 for index in cross))
        # Captions: view 0 reads caption 0 (3 tokens), view 1 caption 1 (2),
        # LiDAR all 5; the query side borrows the same-view partition.
        self.assertEqual(plan.caption_q_offsets.tolist(), [0, 8, 16, 32])
        self.assertEqual(plan.caption_kv_offsets.tolist(), [0, 3, 5, 10])
        self.assertEqual(
            plan.caption_kv_gather.tolist(), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        )

    def test_lidar_can_be_cut_off_from_captions(self):
        plan = build_multiview_maskless_plan(
            self._joint_layout(lidar_attends_captions=False),
            und_tokens=5,
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(plan.caption_q_offsets.tolist(), [0, 8, 16])
        self.assertEqual(plan.caption_kv_offsets.tolist(), [0, 3, 5])
        # Sample-level caption: LiDAR rows leave the per-sample pass.
        plan = build_multiview_maskless_plan(
            self._joint_layout(lidar_attends_captions=False, caption_lengths=()),
            und_tokens=5,
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(plan.caption_q_gather.tolist(), list(range(16)))
        self.assertIsNone(plan.caption_kv_gather)
        self.assertEqual(plan.caption_kv_offsets.tolist(), [0, 5])

    def test_control_attends_sensor_false_splits_the_same_view_pass(self):
        # Sensor queries keep the whole view group; control queries see only their
        # view's control tokens. One varlen call, two segments per group.
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=2,
            patch_height=1,
            patch_width=1,
            control_attends_sensor=False,
            items=(
                MaskItem((2, 1, 1), 2, is_control=True),
                MaskItem((2, 1, 1), 2),
            ),
        )
        plan = build_multiview_maskless_plan(
            layout, und_tokens=3, batch_size=1, device=torch.device("cpu")
        )
        # Packed [C0 C1 | W0 W1]: sensor segments (W0 -> {W0, C0}), (W1 -> {W1, C1}),
        # then control segments (C0 -> {C0}), (C1 -> {C1}).
        self.assertEqual(plan.same_view_q_gather.tolist(), [2, 3, 0, 1])
        self.assertEqual(plan.same_view_q_offsets.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(plan.same_view_kv_gather.tolist(), [2, 0, 3, 1, 0, 1])
        self.assertEqual(plan.same_view_kv_offsets.tolist(), [0, 2, 4, 5, 6])
        # With the flag on, the pass is keyed by the plain partition.
        on = build_multiview_maskless_plan(
            msgspec.structs.replace(layout, control_attends_sensor=True),
            und_tokens=3,
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertIsNone(on.same_view_kv_gather)

    def test_single_camera_without_lidar_skips_the_instant_fold(self):
        layout = MultiviewLayout(
            num_views=1,
            latent_frames=3,
            patch_height=1,
            patch_width=2,
            control_attends_sensor=True,
        )
        plan = build_multiview_maskless_plan(
            layout, und_tokens=4, batch_size=1, device=torch.device("cpu")
        )
        self.assertIsNone(plan.cross_view_offsets)
        self.assertIsNone(plan.same_view_gather)  # one group tiles the stream
        self.assertIsNone(plan.caption_q_gather)
        # same_view scope never folds instants, even with several views.
        plan = build_multiview_maskless_plan(
            self._joint_layout(attention_scope="same_view"),
            und_tokens=5,
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertIsNone(plan.cross_view_offsets)

    def test_batch_tiling_shifts_every_gather(self):
        plan = build_multiview_maskless_plan(
            self._joint_layout(), und_tokens=5, batch_size=2, device=torch.device("cpu")
        )
        self.assertEqual(plan.same_view_offsets.tolist(), [0, 8, 16, 32, 40, 48, 64])
        self.assertEqual(
            plan.same_view_gather[32:40].tolist(),
            [32 + i for i in [0, 1, 2, 3, 8, 9, 10, 11]],
        )
        self.assertEqual(
            plan.caption_kv_gather[10:].tolist(),
            [5 + i for i in [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
        )

    def test_merge_attentions_matches_concatenated_keys(self):
        torch.manual_seed(0)
        q = torch.randn(6, 2, 8)
        k1, v1 = torch.randn(5, 2, 8), torch.randn(5, 2, 8)
        k2, v2 = torch.randn(3, 2, 8), torch.randn(3, 2, 8)

        def attend(k, v):
            scores = torch.einsum("qhd,khd->hqk", q, k)
            return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v), torch.logsumexp(
                scores, -1
            ).transpose(0, 1)

        merged = merge_attentions(*zip(attend(k1, v1), attend(k2, v2)))
        expected, _ = attend(torch.cat([k1, k2]), torch.cat([v1, v2]))
        torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-5)

    def _run(self, layout, device, dtype, atol, rtol, batch_size=1, kernel=None):
        torch.manual_seed(0)
        heads, kv_heads, head_dim = 4, 2, 16
        und = sum(layout.caption_lengths) or 5
        gen = layout.gen_tokens
        q = torch.randn(batch_size, gen, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, gen, kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, gen, kv_heads, head_dim, device=device, dtype=dtype)
        k_und = torch.randn(
            batch_size, und, kv_heads, head_dim, device=device, dtype=dtype
        )
        v_und = torch.randn(
            batch_size, und, kv_heads, head_dim, device=device, dtype=dtype
        )
        context = MultiviewAttentionContext(layout, {})
        out = multiview_maskless_attention(
            q, k, v, k_und, v_und, context, kernel=kernel
        )
        self.assertEqual(tuple(out.shape), (batch_size, gen, heads, head_dim))
        expected = _maskless_oracle(
            q.cpu(), k.cpu(), v.cpu(), k_und.cpu(), v_und.cpu(), layout
        )
        torch.testing.assert_close(out.float().cpu(), expected, atol=atol, rtol=rtol)
        self.assertEqual(len(context.plan_cache), 1)

    def test_cpu_matches_multiset_oracle(self):
        cpu = torch.device("cpu")
        self._run(self._joint_layout(), cpu, torch.float32, 1e-5, 1e-5)
        self._run(self._joint_layout(), cpu, torch.float32, 1e-5, 1e-5, batch_size=2)
        self._run(
            self._joint_layout(lidar_attends_captions=False),
            cpu,
            torch.float32,
            1e-5,
            1e-5,
        )
        self._run(
            self._joint_layout(caption_lengths=()), cpu, torch.float32, 1e-5, 1e-5
        )
        self._run(
            self._joint_layout(control_attends_sensor=False),
            cpu,
            torch.float32,
            1e-5,
            1e-5,
        )
        self._run(
            self._joint_layout(control_attends_sensor=False),
            cpu,
            torch.float32,
            1e-5,
            1e-5,
            batch_size=2,
        )
        camera_only = MultiviewLayout(
            num_views=3,
            latent_frames=6,
            patch_height=1,
            patch_width=2,
            control_attends_sensor=True,
            seconds_per_frame=0.4,
        )
        self._run(camera_only, cpu, torch.float32, 1e-5, 1e-5)

    def test_own_cell_is_double_weighted_relative_to_a_single_mask(self):
        # The folds are a different attention from a single OR-mask on purpose:
        # the query's own (view, frame) keys appear in both sensor passes. Pin
        # that the output follows the multiset oracle, not the set oracle.
        layout = MultiviewLayout(
            num_views=2,
            latent_frames=4,
            patch_height=1,
            patch_width=2,
            control_attends_sensor=True,
            seconds_per_frame=0.4,
        )
        torch.manual_seed(0)
        gen, und = layout.gen_tokens, 5
        q, k, v = (torch.randn(1, gen, 2, 8) for _ in range(3))
        k_und, v_und = (torch.randn(1, und, 2, 8) for _ in range(2))
        out = multiview_maskless_attention(
            q, k, v, k_und, v_und, MultiviewAttentionContext(layout, {})
        )
        multiset = _maskless_oracle(q, k, v, k_und, v_und, layout)
        single = _maskless_oracle(q, k, v, k_und, v_und, layout, count_once=True)
        torch.testing.assert_close(out, multiset, atol=1e-5, rtol=1e-5)
        self.assertGreater((out - single).abs().mean().item(), 1e-3)

    def test_kernel_resolution(self):
        cpu = torch.device("cpu")
        available = {"fa2", "fa3", "fa4"}

        def fake_import(name):
            if name not in available:
                raise ImportError(f"{name} missing")
            return maskless_module._fa2_varlen

        with (
            mock.patch.object(maskless_module, "_import_kernel", fake_import),
            mock.patch.dict(maskless_module._resolved_kernels, clear=True),
            mock.patch.dict(os.environ, {MASKLESS_KERNEL_ENV_VAR: ""}),
        ):
            # auto: FA4 on Blackwell, FA3 on Hopper, torch FA2 below.
            self.assertEqual(resolve_maskless_kernel(cpu, capability_major=10), "fa4")
            self.assertEqual(resolve_maskless_kernel(cpu, capability_major=12), "fa4")
            self.assertEqual(resolve_maskless_kernel(cpu, capability_major=9), "fa3")
            self.assertEqual(resolve_maskless_kernel(cpu, capability_major=8), "fa2")
            # A missing package degrades auto to FA2 but makes an explicit request fail.
            available.discard("fa4")
            maskless_module._resolved_kernels.clear()
            self.assertEqual(resolve_maskless_kernel(cpu, capability_major=10), "fa2")
            with self.assertRaises(ImportError):
                resolve_maskless_kernel(cpu, "fa4", capability_major=10)
            available.add("fa4")
            # An explicit kernel below its architecture floor is refused up front.
            with self.assertRaisesRegex(ValueError, "compute capability"):
                resolve_maskless_kernel(cpu, "fa4", capability_major=8)
            with self.assertRaisesRegex(ValueError, MASKLESS_KERNEL_ENV_VAR):
                resolve_maskless_kernel(cpu, "flex", capability_major=10)
            # The env override wins over auto.
            with mock.patch.dict(os.environ, {MASKLESS_KERNEL_ENV_VAR: "fa2"}):
                maskless_module._resolved_kernels.clear()
                self.assertEqual(
                    resolve_maskless_kernel(cpu, capability_major=10), "fa2"
                )
            # Off-CUDA always takes the reference path's FA2 label.
            self.assertEqual(resolve_maskless_kernel(cpu), "fa2")

    def _available_cuda_kernels(self):
        kernels = ["fa2"]
        major = torch.cuda.get_device_capability()[0]
        for name, floor in (("fa3", 8), ("fa4", 9)):
            if major < floor:
                continue
            try:
                maskless_module._import_kernel(name)
            except ImportError:
                continue
            kernels.append(name)
        return kernels

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for the flash kernels")
    def test_cuda_flash_path_matches_oracle(self):
        # Every kernel the device offers computes the same three passes; the merge
        # then makes them agree with the multiset oracle to bf16 rounding.
        cuda = torch.device("cuda")
        for kernel in self._available_cuda_kernels():
            with self.subTest(kernel=kernel):
                self._run(
                    self._joint_layout(),
                    cuda,
                    torch.bfloat16,
                    3e-2,
                    3e-2,
                    kernel=kernel,
                )
                self._run(
                    self._joint_layout(),
                    cuda,
                    torch.bfloat16,
                    3e-2,
                    3e-2,
                    batch_size=2,
                    kernel=kernel,
                )
                self._run(
                    self._joint_layout(caption_lengths=()),
                    cuda,
                    torch.bfloat16,
                    3e-2,
                    3e-2,
                    kernel=kernel,
                )


class TestRegistry(unittest.TestCase):
    def test_pipeline_is_discoverable_and_paths_resolve(self):
        _discover_and_register_pipelines()
        self.assertIn("Cosmos3MultiviewPipeline", _PIPELINE_REGISTRY)
        for model_path in (
            "nvidia/Cosmos3-Nano-Transfer-Auto",
            "/mnt/models/Cosmos3-Nano-Transfer-Auto",
        ):
            with self.subTest(model_path=model_path):
                config_info = _get_config_info(model_path)
                self.assertIs(config_info.pipeline_config_cls, Cosmos3MultiviewConfig)
                self.assertIs(
                    config_info.sampling_param_cls, Cosmos3MultiviewSamplingParams
                )
        # The plain Nano release must keep resolving to the regular pipeline.
        self.assertIs(
            _get_config_info("nvidia/Cosmos3-Nano").pipeline_config_cls, Cosmos3Config
        )

    def test_class_name_detectors_stay_disjoint(self):
        expectations = {
            "Cosmos3MultiviewPipeline": Cosmos3MultiviewConfig,
            "Cosmos3OmniPipeline": Cosmos3Config,
        }
        for class_name, config_cls in expectations.items():
            with self.subTest(class_name=class_name):
                with mock.patch(
                    "sglang.multimodal_gen.registry.maybe_download_model_index",
                    return_value={"_class_name": class_name},
                ):
                    config_info = _get_config_info(f"acme/renamed-{class_name.lower()}")
                self.assertIs(config_info.pipeline_config_cls, config_cls)

    def test_transformer_class_is_resolvable(self):
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        model_cls, _ = ModelRegistry.resolve_model_cls("Cosmos3MultiviewTransformer")
        self.assertEqual(model_cls.__name__, "Cosmos3MultiviewTransformer")
        self.assertEqual(
            model_cls._cross_attention_cls.__name__, "Cosmos3MultiviewCrossAttention"
        )


if __name__ == "__main__":
    unittest.main()
