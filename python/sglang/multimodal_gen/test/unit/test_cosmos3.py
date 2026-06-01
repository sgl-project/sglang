# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cosmos3 config, weight mapping, and sampling params."""

import importlib.util
import unittest
from unittest import mock

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
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
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

    def test_audio_proj_in_dropped(self):
        key, *_ = _apply(self.fn, "audio_proj_in.weight")
        self.assertEqual(key, "")

    def test_action_proj_in_dropped(self):
        key, *_ = _apply(self.fn, "action_proj_in.weight")
        self.assertEqual(key, "")

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


if __name__ == "__main__":
    unittest.main()
