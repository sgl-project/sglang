# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cosmos3 config, weight mapping, and sampling params."""

import importlib.util
import json
import unittest
from unittest import mock

import torch
from fastapi import HTTPException
from sglang.multimodal_gen.configs.models.dits.cosmos3video import (
    _build_cosmos3_param_names_mapping,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _reject_unsupported_cosmos3_modes,
)
from sglang.multimodal_gen.runtime.layers.domain_aware_linear import (
    DomainAwareLinear,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    build_action_prompt,
    canonical_aspect_ratio,
    find_closest_target_size,
    get_domain_id,
    get_raw_action_dim,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_guardrails import (
    is_cosmos_guardrail_available,
)


def _apply(mapping_fn, key):
    """Return (target_key, merge_index, total_splits) for a diffusers weight key."""
    return mapping_fn(key)


class TestCosmos3ParamNamesMapping(unittest.TestCase):
    """Verify diffusers → sglang weight key translations."""

    @classmethod
    def setUpClass(cls):
        cls.fn = staticmethod(
            get_param_names_mapping(_build_cosmos3_param_names_mapping())
        )

    # --- skipped / dropped weights ---

    def test_lm_head_dropped(self):
        key, idx, total = _apply(self.fn, "lm_head.weight")
        self.assertEqual(key, "")

    def test_norm_dropped(self):
        key, idx, total = _apply(self.fn, "norm.weight")
        self.assertEqual(key, "")

    def test_audio_proj_in_loaded(self):
        # Omni heads are loaded via identity passthrough.
        key, *_ = _apply(self.fn, "audio_proj_in.weight")
        self.assertEqual(key, "audio_proj_in.weight")

    def test_action_proj_in_loaded(self):
        key, *_ = _apply(self.fn, "action_proj_in.fc.weight")
        self.assertEqual(key, "action_proj_in.fc.weight")

    # --- top-level pass-through ---

    def test_embed_tokens(self):
        key, *_ = _apply(self.fn, "embed_tokens.weight")
        self.assertEqual(key, "language_model.embed_tokens.weight")

    def test_norm_moe_gen(self):
        key, *_ = _apply(self.fn, "norm_moe_gen.weight")
        self.assertEqual(key, "norm_moe_gen.weight")

    # --- time embedder (pass-through: checkpoint already uses linear_1/2) ---

    def test_time_embedder_linear_1_passthrough(self):
        key, *_ = _apply(self.fn, "time_embedder.linear_1.weight")
        self.assertEqual(key, "time_embedder.linear_1.weight")

    def test_time_embedder_linear_2_passthrough(self):
        key, *_ = _apply(self.fn, "time_embedder.linear_2.bias")
        self.assertEqual(key, "time_embedder.linear_2.bias")

    # --- GEN pathway: Q/K/V merge (must not be claimed by UND catch-all) ---

    def test_gen_q_proj_key_and_merge_index(self):
        key, idx, total = _apply(self.fn, "layers.3.self_attn.add_q_proj.weight")
        self.assertEqual(key, "gen_layers.3.cross_attention.to_qkv.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 3)

    def test_gen_k_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.add_k_proj.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 3)

    def test_gen_v_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.add_v_proj.weight")
        self.assertEqual(idx, 2)
        self.assertEqual(total, 3)

    def test_gen_o_proj(self):
        key, idx, total = _apply(self.fn, "layers.5.self_attn.to_add_out.weight")
        self.assertEqual(key, "gen_layers.5.cross_attention.to_out.weight")
        self.assertIsNone(idx)

    def test_gen_norm_added_q(self):
        key, idx, _ = _apply(self.fn, "layers.2.self_attn.norm_added_q.weight")
        self.assertEqual(key, "gen_layers.2.cross_attention.norm_q.weight")
        self.assertIsNone(idx)

    def test_gen_norm_added_k(self):
        key, idx, _ = _apply(self.fn, "layers.2.self_attn.norm_added_k.weight")
        self.assertEqual(key, "gen_layers.2.cross_attention.norm_k.weight")
        self.assertIsNone(idx)

    def test_gen_mlp_gate_proj(self):
        key, idx, total = _apply(self.fn, "layers.2.mlp_moe_gen.gate_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.gate_up_proj.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 2)

    def test_gen_mlp_up_proj(self):
        key, idx, total = _apply(self.fn, "layers.2.mlp_moe_gen.up_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.gate_up_proj.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 2)

    def test_gen_mlp_down_proj_passthrough(self):
        key, idx, _ = _apply(self.fn, "layers.2.mlp_moe_gen.down_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.down_proj.weight")
        self.assertIsNone(idx)

    # --- UND pathway: Q/K/V merge ---

    def test_und_q_proj_key_and_merge_index(self):
        key, idx, total = _apply(self.fn, "layers.7.self_attn.to_q.weight")
        self.assertEqual(key, "language_model.layers.7.self_attn.to_qkv.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 3)

    def test_und_k_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.to_k.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 3)

    def test_und_v_proj_merge_index(self):
        _, idx, total = _apply(self.fn, "layers.0.self_attn.to_v.weight")
        self.assertEqual(idx, 2)
        self.assertEqual(total, 3)

    def test_und_mlp_gate_proj(self):
        key, idx, total = _apply(self.fn, "layers.1.mlp.gate_proj.weight")
        self.assertEqual(key, "language_model.layers.1.mlp.gate_up_proj.weight")
        self.assertEqual(idx, 0)
        self.assertEqual(total, 2)

    def test_und_mlp_up_proj(self):
        _, idx, total = _apply(self.fn, "layers.1.mlp.up_proj.weight")
        self.assertEqual(idx, 1)
        self.assertEqual(total, 2)

    def test_und_layernorm_catch_all(self):
        key, idx, _ = _apply(self.fn, "layers.0.input_layernorm.weight")
        self.assertEqual(key, "language_model.layers.0.input_layernorm.weight")
        self.assertIsNone(idx)

    # --- ordering: GEN patterns must not be swallowed by UND catch-all ---

    def test_gen_layernorm_not_mapped_to_und(self):
        key, *_ = _apply(self.fn, "layers.0.input_layernorm_moe_gen.weight")
        self.assertIn("gen_layers", key)
        self.assertNotIn("language_model", key)

    def test_gen_post_attention_layernorm_not_mapped_to_und(self):
        key, *_ = _apply(self.fn, "layers.4.post_attention_layernorm_moe_gen.weight")
        self.assertIn("gen_layers", key)
        self.assertNotIn("language_model", key)


class TestCosmos3AdjustNumFrames(unittest.TestCase):
    """Verify VAE-aligned frame rounding in Cosmos3Config."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Cosmos3Config()

    def test_single_frame_t2i_bypass(self):
        self.assertEqual(self.cfg.adjust_num_frames(1), 1)

    def test_already_aligned(self):
        # (81 - 1) = 80, 80 % 4 == 0
        self.assertEqual(self.cfg.adjust_num_frames(81), 81)

    def test_rounds_down_to_nearest_aligned(self):
        # (83 - 1) = 82 → floor(82/4)*4 + 1 = 81
        self.assertEqual(self.cfg.adjust_num_frames(83), 81)
        # (6 - 1) = 5 → floor(5/4)*4 + 1 = 5
        self.assertEqual(self.cfg.adjust_num_frames(6), 5)

    def test_minimum_video_frame_count(self):
        # 2 frames: (2-1)=1, 1//4=0, 0*4+1=1 → rounds to 1, but 1 is T2I — still valid
        self.assertEqual(self.cfg.adjust_num_frames(2), 1)


class TestCosmos3SamplingParamsDataType(unittest.TestCase):
    """Verify num_frames==1 flips data_type to IMAGE before file name derivation."""

    def test_single_frame_sets_image_data_type(self):
        params = Cosmos3SamplingParams(prompt="test", num_frames=1)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.IMAGE)
        self.assertTrue(
            params.output_file_name.endswith((".png", ".jpg", ".jpeg", ".webp")),
            f"Expected image extension, got: {params.output_file_name}",
        )

    def test_multi_frame_keeps_video_data_type(self):
        params = Cosmos3SamplingParams(prompt="test", num_frames=81)
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.VIDEO)

    def test_default_num_frames_is_video(self):
        params = Cosmos3SamplingParams(prompt="test")
        params._set_output_file_name()
        self.assertEqual(params.data_type, DataType.VIDEO)


class TestCosmos3ModelResolution(unittest.TestCase):
    """Verify Cosmos3 checkpoints resolve to the native SGLang pipeline."""

    def test_hf_checkpoint_uses_registered_native_pipeline_config(self):
        for model_path in (
            "nvidia/Cosmos3-Nano",
            "nvidia/Cosmos3-Super",
            "nvidia/Cosmos3-Super-Text2Image",
            "nvidia/Cosmos3-Super-Image2Video",
        ):
            with self.subTest(model_path=model_path):
                self.assertIsNone(get_non_diffusers_pipeline_name(model_path))
                config_info = _get_config_info(model_path)
                self.assertIsNotNone(config_info)
                self.assertIs(config_info.sampling_param_cls, Cosmos3SamplingParams)
                self.assertIs(config_info.pipeline_config_cls, Cosmos3Config)


class TestCosmos3OpenAIProtocol(unittest.TestCase):
    """Verify Cosmos3-only knobs stay out of the stable request schema."""

    def test_cosmos3_private_fields_are_extra_fields(self):
        for request_cls in (ImageGenerationsRequest, VideoGenerationsRequest):
            with self.subTest(request_cls=request_cls.__name__):
                self.assertIn("max_sequence_length", request_cls.model_fields)
                self.assertIn("flow_shift", request_cls.model_fields)
                self.assertNotIn("use_duration_template", request_cls.model_fields)
                self.assertNotIn("use_resolution_template", request_cls.model_fields)
                self.assertNotIn("use_system_prompt", request_cls.model_fields)
                self.assertNotIn("use_guardrails", request_cls.model_fields)

        self.assertNotIn("generate_sound", VideoGenerationsRequest.model_fields)
        self.assertNotIn("sound_duration", VideoGenerationsRequest.model_fields)

    def test_unsupported_cosmos3_modes_allow_falsy_extra_fields(self):
        req = VideoGenerationsRequest(
            prompt="test",
            generate_sound=False,
            action_mode="",
            condition_frame_indexes_vision=[],
            condition_video_keep={},
        )
        _reject_unsupported_cosmos3_modes(req, "nvidia/Cosmos3-Nano")

        req = VideoGenerationsRequest(prompt="test", generate_sound=True)
        with self.assertRaises(HTTPException):
            _reject_unsupported_cosmos3_modes(req, "nvidia/Cosmos3-Nano")


class TestCosmos3Guardrails(unittest.TestCase):
    """Verify optional guardrail dependency handling."""

    def setUp(self):
        is_cosmos_guardrail_available.cache_clear()

    def tearDown(self):
        is_cosmos_guardrail_available.cache_clear()

    def test_guardrail_availability_matches_package_spec(self):
        self.assertEqual(
            is_cosmos_guardrail_available(),
            importlib.util.find_spec("cosmos_guardrail") is not None,
        )

    @mock.patch("importlib.util.find_spec", return_value=None)
    def test_missing_guardrail_package_reports_unavailable(self, _):
        self.assertFalse(is_cosmos_guardrail_available())


class TestDomainAwareLinear(unittest.TestCase):
    """Per-embodiment action projection (Cosmos3 action heads)."""

    def test_weight_table_shapes(self):
        layer = DomainAwareLinear(input_size=64, output_size=8, num_domains=32)
        self.assertEqual(tuple(layer.fc.weight.shape), (32, 8 * 64))
        self.assertEqual(tuple(layer.bias.weight.shape), (32, 8))

    def test_forward_rank2(self):
        layer = DomainAwareLinear(input_size=6, output_size=4, num_domains=3)
        x = torch.randn(2, 6)
        out = layer(x, torch.tensor([0, 2]))
        self.assertEqual(tuple(out.shape), (2, 4))

    def test_forward_rank3(self):
        layer = DomainAwareLinear(input_size=6, output_size=4, num_domains=3)
        x = torch.randn(2, 5, 6)
        out = layer(x, torch.tensor([1, 0]))
        self.assertEqual(tuple(out.shape), (2, 5, 4))

    def test_per_domain_weights_differ(self):
        layer = DomainAwareLinear(input_size=4, output_size=4, num_domains=2)
        x = torch.randn(1, 4)
        out0 = layer(x, torch.tensor([0]))
        out1 = layer(x, torch.tensor([1]))
        self.assertFalse(torch.allclose(out0, out1))

    def test_rejects_out_of_range_domain(self):
        layer = DomainAwareLinear(input_size=4, output_size=4, num_domains=2)
        with self.assertRaises(ValueError):
            layer(torch.randn(1, 4), torch.tensor([5]))

    def test_rejects_batch_mismatch(self):
        layer = DomainAwareLinear(input_size=4, output_size=4, num_domains=4)
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 4), torch.tensor([0, 1, 2]))


class TestCosmos3OmniSamplingParams(unittest.TestCase):
    """Sound flag and the native T2I CFG-window default."""

    def test_generate_sound_default_false(self):
        self.assertFalse(Cosmos3SamplingParams().generate_sound)

    def test_generate_sound_opt_in(self):
        self.assertTrue(Cosmos3SamplingParams(generate_sound=True).generate_sound)

    def test_t2i_cfg_window_default(self):
        self.assertEqual(
            Cosmos3SamplingParams(num_frames=1).guidance_interval, (400.0, 1000.0)
        )

    def test_video_leaves_cfg_window_unset(self):
        self.assertIsNone(Cosmos3SamplingParams(num_frames=81).guidance_interval)


class TestCosmos3FlowUniPCOverride(unittest.TestCase):
    """Native FlowUniPC scheduler override (vs the shipped Karras config)."""

    def test_scheduler_override(self):
        self.assertEqual(
            Cosmos3Config().scheduler_class_override, "FlowUniPCMultistepScheduler"
        )


class TestCosmos3ActionUtils(unittest.TestCase):
    """Action resolution, aspect ratio, JSON prompt, and embodiment lookups."""

    def test_find_closest_target_size(self):
        self.assertEqual(find_closest_target_size(480, 640, "480"), (736, 544))
        self.assertEqual(find_closest_target_size(512, 512, "480"), (640, 640))
        with self.assertRaises(ValueError):
            find_closest_target_size(480, 640, "999")

    def test_canonical_aspect_ratio(self):
        self.assertEqual(canonical_aspect_ratio(736, 544), "4,3")
        self.assertEqual(canonical_aspect_ratio(100, 50), "2,1")  # gcd fallback

    def test_build_action_prompt_matches_reference_json(self):
        d = json.loads(build_action_prompt(
            "Put the pot to the left of the purple item.", "ego_view",
            num_frames=17, fps=5, height=544, width=736,
        ))
        self.assertEqual(
            d["cinematography"]["framing"],
            "This video is captured from a first-person perspective looking at the scene.",
        )
        self.assertEqual(d["actions"][0]["time"], "0:00-0:03")
        self.assertNotIn("idle_frame", d["actions"][0])
        self.assertEqual(d["duration"], "3s")
        self.assertEqual(d["fps"], 5.0)
        self.assertEqual(d["resolution"], {"H": 544, "W": 736})
        self.assertEqual(d["aspect_ratio"], "4,3")

    def test_build_action_prompt_adds_terminal_punctuation(self):
        d = json.loads(build_action_prompt("pick up the cube", "wrist_view", 17, 5, 544, 736))
        self.assertEqual(d["actions"][0]["description"], "pick up the cube.")

    def test_bridge_embodiment_raw_dim(self):
        self.assertEqual(get_raw_action_dim("bridge_orig_lerobot"), 10)
        self.assertIsInstance(get_domain_id("bridge_orig_lerobot"), int)


class TestCosmos3V2VParams(unittest.TestCase):
    def test_condition_frame_indexes_default_none(self):
        self.assertIsNone(Cosmos3SamplingParams().condition_frame_indexes_vision)


if __name__ == "__main__":
    unittest.main()
