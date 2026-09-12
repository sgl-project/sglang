# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Cosmos-Dreams-Transfer port: control-video contract
parsing, chunk-partition alignment, prompt formatting, vision-only position
ids, and registry wiring."""

import copy
import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    parse_cosmos_dreams_manifest,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams_transfer import (
    CosmosDreamsTransferConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos_dreams_transfer import (
    CosmosDreamsTransferSamplingParams,
)
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
    build_cosmos_dreams_position_ids,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams import (
    append_kv_history,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams_transfer import (
    CONTROL_EMPHASIS_TEMPLATE,
    TRANSFER_SYSTEM_PROMPT,
    align_pixel_frames_to_chunks,
    format_transfer_prompt,
    read_video_fps,
    resize_center_crop_uint8,
    synthetic_control_frames,
    tokenize_transfer_prompt,
)

# ``transformer/config.json["cosmos_dreams"]`` of Cosmos3-Nano-Sim-Transfer
# (checkpoint causal_8b_sf_dmd_transfer_4modality_480p_ga_v2_midtrain_
# causal_control_with_rgb_history_text_dropout_0@iter_000001750).
TRANSFER_ARTIFACT = {
    "attention_mode": "three_way",
    "base_fps": 24.0,
    "checkpoint_hash": "f72aa96e1b639ddc3db45caac76ff23810ffd63c47fb6bb108cd7c46f0b8f936",
    "checkpoint_id": (
        "causal_8b_sf_dmd_transfer_4modality_480p_ga_v2_midtrain_causal_control_"
        "with_rgb_history_text_dropout_0@iter_000001750"
    ),
    "checkpoint_iteration": 1750,
    "chunk_size": 4,
    "conditioning": {
        "emphasize_control_in_prompt": True,
        "hints": ["edge", "blur", "depth", "seg"],
        "mode": "control_video",
        "no_eviction": True,
        "share_vision_temporal_positions": True,
        "system_prompt_id": "cosmos3_transfer_v1",
        "transfer_control_attention_mode": "causal_control_with_rgb_history",
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

MANIFEST = parse_cosmos_dreams_manifest(TRANSFER_ARTIFACT)


def _artifact() -> dict:
    return copy.deepcopy(TRANSFER_ARTIFACT)


class _FakeTokenizer:
    eos_token_id = 7

    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []
        self.template_ids = [1, 2, 3]

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        self.conversations.append(conversation)
        return list(self.template_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|vision_start|>": 9}[token]


class TestTransferManifest(unittest.TestCase):
    def test_transfer_defaults_describe_a_distilled_model(self):
        self.assertEqual(CosmosDreamsTransferSamplingParams().guidance_scale, 1.0)
        deployment = CosmosDreamsTransferConfig().get_model_deployment_config()
        self.assertFalse(deployment.supports_cfg_parallel)

    def test_control_video_contract_parses(self):
        self.assertEqual(MANIFEST.conditioning_mode, "control_video")
        self.assertEqual(
            MANIFEST.control_contract.hints, ("edge", "blur", "depth", "seg")
        )
        self.assertTrue(MANIFEST.control_contract.no_eviction)
        self.assertTrue(MANIFEST.control_contract.emphasize_control_in_prompt)
        self.assertEqual(MANIFEST.action_tokens_per_frame, 0)
        self.assertEqual(MANIFEST.chunk_size, 4)
        with self.assertRaises(ValueError):
            _ = MANIFEST.max_action_dim
        with self.assertRaises(ValueError):
            _ = MANIFEST.action_contract

    def test_rejects_contracts_the_runtime_cannot_honor(self):
        tamperings = {
            "unknown hint": lambda a: a["conditioning"].__setitem__(
                "hints", ["edge", "lidar"]
            ),
            "duplicate hint": lambda a: a["conditioning"].__setitem__(
                "hints", ["edge", "edge"]
            ),
            "attention mode": lambda a: a["conditioning"].__setitem__(
                "transfer_control_attention_mode", "global_control"
            ),
            "positions": lambda a: a["conditioning"].__setitem__(
                "share_vision_temporal_positions", False
            ),
            "system prompt": lambda a: a["conditioning"].__setitem__(
                "system_prompt_id", "cosmos3_video_v1"
            ),
            "finite window": lambda a: a["conditioning"].__setitem__(
                "no_eviction", False
            ),
            "unknown mode": lambda a: a["conditioning"].__setitem__("mode", "lidar"),
        }
        for name, tamper in tamperings.items():
            with self.subTest(name=name):
                artifact = _artifact()
                tamper(artifact)
                with self.assertRaises(ValueError):
                    parse_cosmos_dreams_manifest(artifact)


class TestTransferFramesAndHistory(unittest.TestCase):
    def test_alignment_to_chunk_partition(self):
        # 61 pixel frames -> 16 latents; the partition [1, 4, 4, 4] keeps 13 -> 49 frames.
        cases = {61: 49, 49: 49, 17: 17, 81: 81, 601: 593}
        for pixel_frames, expected in cases.items():
            with self.subTest(pixel_frames=pixel_frames):
                self.assertEqual(
                    align_pixel_frames_to_chunks(
                        pixel_frames, temporal_compression_factor=4, chunk_size=4
                    ),
                    expected,
                )
        with self.assertRaises(ValueError):
            align_pixel_frames_to_chunks(
                16, temporal_compression_factor=4, chunk_size=4
            )

    def test_unbounded_history_keeps_every_committed_block(self):
        history = None
        for block_frames in (1, 4, 1, 1):
            kv = [
                (
                    torch.ones(1, block_frames * 6, 2, 4),
                    torch.ones(1, block_frames * 6, 2, 4),
                )
            ]
            history = append_kv_history(
                history, kv, tokens_per_frame=6, sink_frames=0, window_frames=None
            )
        self.assertEqual(history[0][0].shape[1], 7 * 6)

    def test_resize_center_crop_matches_canvas(self):
        frames = torch.randint(0, 256, (3, 5, 300, 400), dtype=torch.uint8)
        out = resize_center_crop_uint8(frames, height=480, width=832)
        self.assertEqual(tuple(out.shape), (3, 5, 480, 832))
        self.assertEqual(out.dtype, torch.uint8)

    def test_missing_video_has_no_fps(self):
        self.assertIsNone(read_video_fps("/nonexistent/control.mp4"))

    def test_synthetic_control_matches_warmup_request(self):
        frames = synthetic_control_frames(num_frames=17, height=480, width=832)
        self.assertEqual(tuple(frames.shape), (3, 17, 480, 832))
        self.assertEqual(frames.dtype, torch.uint8)
        with self.assertRaises(ValueError):
            synthetic_control_frames(num_frames=0, height=480, width=832)

    def test_long_prompt_is_truncated_like_training(self):
        tokenizer = _FakeTokenizer()
        tokenizer.template_ids = list(range(5000))
        ids, _ = tokenize_transfer_prompt(
            tokenizer, "x" * 10, device=torch.device("cpu")
        )
        self.assertEqual(ids.shape[1], 4096 + 2)
        self.assertEqual(ids[0, -2:].tolist(), [7, 9])


class TestTransferPrompt(unittest.TestCase):
    def test_plain_prompt_gets_training_metadata_and_emphasis(self):
        prompt = format_transfer_prompt(
            "A red car drives through the rain.",
            hint="depth",
            fps=30,
            num_frames=61,
            height=480,
            width=832,
            emphasize_control=True,
        )
        expected = (
            "A red car drives through the rain. The video is 2.0 seconds long and is of 30 FPS. "
            "This video is of 480x832 resolution."
            + CONTROL_EMPHASIS_TEMPLATE.format(hint="depth")
        )
        self.assertEqual(prompt, expected)

    def test_plain_prompt_without_emphasis_and_empty_prompt(self):
        prompt = format_transfer_prompt(
            "A cat",
            hint="edge",
            fps=24,
            num_frames=49,
            height=480,
            width=832,
            emphasize_control=False,
        )
        self.assertEqual(
            prompt,
            "A cat. The video is 2.0 seconds long and is of 24 FPS. This video is of 480x832 resolution.",
        )
        empty = format_transfer_prompt(
            "",
            hint="edge",
            fps=24,
            num_frames=49,
            height=480,
            width=832,
            emphasize_control=False,
        )
        self.assertTrue(empty.startswith("The video is 2.0 seconds long"))

    def test_json_prompt_gets_metadata_fields(self):
        prompt = format_transfer_prompt(
            '{"actions": [{"description": "A cat sits."}]}',
            hint="seg",
            fps=30,
            num_frames=61,
            height=480,
            width=832,
            emphasize_control=False,
        )
        self.assertEqual(
            prompt,
            '{"actions": [{"description": "A cat sits."}], "duration": "2s", "fps": 30.0, '
            '"resolution": {"H": 480, "W": 832}, "aspect_ratio": "16,9"}',
        )
        with self.assertRaises(ValueError):
            format_transfer_prompt(
                "x",
                hint="edge",
                fps=0,
                num_frames=49,
                height=480,
                width=832,
                emphasize_control=False,
            )

    def test_tokenizer_uses_transfer_system_prompt(self):
        tokenizer = _FakeTokenizer()
        ids, mask = tokenize_transfer_prompt(
            tokenizer, "hello", device=torch.device("cpu")
        )
        conversation = tokenizer.conversations[0]
        self.assertEqual(
            conversation[0], {"role": "system", "content": TRANSFER_SYSTEM_PROMPT}
        )
        self.assertEqual(conversation[1], {"role": "user", "content": "hello"})
        self.assertEqual(ids.tolist(), [[1, 2, 3, 7, 9]])
        self.assertTrue(bool(mask.all()))


class TestTransferPositionIds(unittest.TestCase):
    def test_vision_only_ids_share_frame_time_and_reset_space(self):
        ids = build_cosmos_dreams_position_ids(
            frame_start=5,
            num_frames=2,
            grid_h=2,
            grid_w=3,
            text_length=10,
            temporal_modality_margin=100,
            fps=24.0,
            base_fps=24.0,
            temporal_compression_factor=4,
            action_tokens_per_frame=0,
            null_action_frames=(),
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(ids.shape), (3, 12))
        self.assertEqual(ids[0, :6].tolist(), [115.0] * 6)
        self.assertEqual(ids[0, 6:].tolist(), [116.0] * 6)
        self.assertEqual(ids[1, :6].tolist(), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(ids[2, :6].tolist(), [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            build_cosmos_dreams_position_ids(
                frame_start=0,
                num_frames=1,
                grid_h=1,
                grid_w=1,
                text_length=1,
                temporal_modality_margin=0,
                fps=24.0,
                base_fps=24.0,
                temporal_compression_factor=4,
                action_tokens_per_frame=0,
                null_action_frames=(0,),
                device=torch.device("cpu"),
            )


class TestTransferRegistryAndConfig(unittest.TestCase):
    def test_pipeline_is_discoverable_and_paths_resolve(self):
        _discover_and_register_pipelines()
        self.assertIn("CosmosDreamsTransferPipeline", _PIPELINE_REGISTRY)
        for model_path in (
            "nvidia/Cosmos3-Nano-Sim-Transfer",
            "/models/Cosmos3-Nano-Sim-Transfer",
        ):
            with self.subTest(model_path=model_path):
                config_info = _get_config_info(model_path)
                self.assertIs(
                    config_info.pipeline_config_cls, CosmosDreamsTransferConfig
                )
                self.assertIs(
                    config_info.sampling_param_cls, CosmosDreamsTransferSamplingParams
                )

    def test_class_name_detectors_stay_disjoint(self):
        expectations = {
            "CosmosDreamsTransferPipeline": CosmosDreamsTransferConfig,
            "CosmosDreamsPipeline": CosmosDreamsConfig,
        }
        for class_name, config_cls in expectations.items():
            with self.subTest(class_name=class_name):
                with mock.patch(
                    "sglang.multimodal_gen.registry.maybe_download_model_index",
                    return_value={"_class_name": class_name},
                ):
                    # Resolution is cached per path, so each class needs its own.
                    config_info = _get_config_info(f"acme/renamed-{class_name.lower()}")
                self.assertIs(config_info.pipeline_config_cls, config_cls)

    def test_config_flags_and_mode_check(self):
        config = CosmosDreamsTransferConfig()
        self.assertEqual(config.conditioning_mode, "control_video")
        self.assertEqual(config.transformer_class_override, "CosmosDreamsTransformer")
        self.assertTrue(config.use_system_prompt)
        self.assertEqual(config.canvas_tier, "480")
        self.assertFalse(config.supports_action_endpoint())
        with mock.patch(
            "sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams._transformer_config",
            return_value={"cosmos_dreams": _artifact()},
        ):
            with self.assertRaises(ValueError):
                CosmosDreamsConfig()._validate_checkpoint("/models/transfer")


class TestTransferSamplingParams(unittest.TestCase):
    def test_defaults_and_control_resolution(self):
        params = CosmosDreamsTransferSamplingParams(
            prompt="a street", control_path="edge.mp4", control_hint="edge"
        )
        self.assertEqual(params.resolved_control_path, "edge.mp4")
        self.assertEqual(params.resolved_control_hint, "edge")
        self.assertEqual(params.num_frames, 601)
        self.assertEqual(params.fps, 24)
        self.assertEqual(params.supported_resolutions[0], (832, 480))
        self.assertIsNone(params.emphasize_control_in_prompt)
        self.assertFalse(params.canvas_from_control)
        self.assertFalse(params.is_explicit("fps"))
        self.assertIn(
            "emphasize_control_in_prompt",
            CosmosDreamsTransferSamplingParams.video_request_extra_fields(),
        )

    def test_rejects_unsupported_requests(self):
        base = {
            "prompt": "a street",
            "control_path": "edge.mp4",
            "control_hint": "edge",
        }
        # Default construction must work (offload planning instantiates the class);
        # the control input becomes mandatory at request validation.
        defaults = CosmosDreamsTransferSamplingParams()
        with self.assertRaises(ValueError):
            _ = defaults.resolved_control_path
        rejected = (
            {"control_path": ["a.mp4", "b.mp4"], "control_hint": ["edge", "depth"]},
            {"control_hint": None},
            {"control_hint": "lidar"},
            {"image_path": "first.png"},
            {"video_path": "source.mp4"},
            {"action_mode": "forward_dynamics"},
            {"sound_duration": 1.0},
            {"num_first_chunk_conditional_frames": 1},
            {"control_guidance": 2.0},
            {"num_frames": 1},
            {"emphasize_control_in_prompt": "yes"},
        )
        for overrides in rejected:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    CosmosDreamsTransferSamplingParams(**{**base, **overrides})


if __name__ == "__main__":
    unittest.main()
