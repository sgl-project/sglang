# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Cosmos3 config, weight mapping, and sampling params."""

import importlib.util
import json
import types
import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.configs.models.dits.cosmos3video import (
    _build_cosmos3_param_names_mapping,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import (
    COSMOS3_EDGE_SUPPORTED_RESOLUTIONS,
    Cosmos3SamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.registry import (
    _PIPELINE_REGISTRY,
    _discover_and_register_pipelines,
    _get_config_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_generation_response,
    action_metadata,
    build_action_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _cosmos3_sampling_param_kwargs,
    _multipart_video_extras,
    _resolve_sound_duration,
    _resolve_video_path,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import scheduler_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
    SchedulerLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3OmniTransformer,
    DomainAwareLinear,
    compute_mrope_position_ids_action,
    compute_mrope_position_ids_sound,
    compute_mrope_position_ids_vision,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3DenoisingStage,
    Cosmos3ImagePreprocessStage,
    Cosmos3LatentPreparationStage,
    Cosmos3TimestepPreparationStage,
    Cosmos3TokenizationStage,
    _inject_caption_metadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    EMBODIMENT_TO_DOMAIN_ID,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_guardrails import (
    is_cosmos_guardrail_available,
)


def _apply(mapping_fn, key):
    """Return (target_key, merge_index, total_splits) for a diffusers weight key."""
    return mapping_fn(key)


def _cosmos3_server_args(config=None):
    return types.SimpleNamespace(
        model_id=None,
        model_path="nvidia/Cosmos3-Nano",
        served_model_name="cosmos3-production",
        backend=None,
        pipeline_class_name=None,
        output_path=None,
        comfyui_mode=False,
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        pipeline_config=config or Cosmos3Config(),
    )


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

    # --- top-level pass-through ---

    def test_audio_proj_in_passthrough(self):
        key, *_ = _apply(self.fn, "audio_proj_in.weight")
        self.assertEqual(key, "audio_proj_in.weight")

    def test_action_proj_in_passthrough(self):
        key, *_ = _apply(self.fn, "action_proj_in.fc.weight")
        self.assertEqual(key, "action_proj_in.fc.weight")

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


class TestCosmos3DenseParamNamesMapping(unittest.TestCase):
    """Dense (squared-ReLU) checkpoints ship no gate_proj; up/down_proj must
    pass through unmerged."""

    @classmethod
    def setUpClass(cls):
        cls.fn = staticmethod(
            get_param_names_mapping(_build_cosmos3_param_names_mapping(gated_mlp=False))
        )

    def test_und_mlp_up_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.1.mlp.up_proj.weight")
        self.assertEqual(key, "language_model.layers.1.mlp.up_proj.weight")
        self.assertIsNone(idx)

    def test_und_mlp_down_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.1.mlp.down_proj.weight")
        self.assertEqual(key, "language_model.layers.1.mlp.down_proj.weight")
        self.assertIsNone(idx)

    def test_gen_mlp_up_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.2.mlp_moe_gen.up_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.up_proj.weight")
        self.assertIsNone(idx)

    def test_gen_mlp_down_proj_unmerged(self):
        key, idx, _ = _apply(self.fn, "layers.2.mlp_moe_gen.down_proj.weight")
        self.assertEqual(key, "gen_layers.2.mlp.down_proj.weight")
        self.assertIsNone(idx)

    def test_qkv_merge_still_applies(self):
        key, idx, total = _apply(self.fn, "layers.0.self_attn.to_q.weight")
        self.assertEqual(key, "language_model.layers.0.self_attn.to_qkv.weight")
        self.assertEqual((idx, total), (0, 3))


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


class TestCosmos3SchedulerConfig(unittest.TestCase):
    """Verify Cosmos3 scheduler class and flow-shift defaults."""

    def test_config_overrides_checkpoint_scheduler_class(self):
        cfg = Cosmos3Config()
        self.assertEqual(cfg.scheduler_class_override, "FlowUniPCMultistepScheduler")
        self.assertIsNone(cfg.flow_shift)

    def test_scheduler_loader_uses_configured_class_override(self):
        class FakeScheduler:
            def __init__(self, **config):
                self.config = config

        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(
                scheduler_class_override="FlowUniPCMultistepScheduler",
                flow_shift=None,
            )
        )
        with (
            mock.patch.object(
                scheduler_loader,
                "get_diffusers_component_config",
                return_value={"_class_name": "CheckpointScheduler", "foo": "bar"},
            ),
            mock.patch.object(
                scheduler_loader.ModelRegistry,
                "resolve_model_cls",
                return_value=(FakeScheduler, None),
            ) as resolve,
        ):
            scheduler = SchedulerLoader().load_customized("unused", server_args)

        resolve.assert_called_once_with("FlowUniPCMultistepScheduler")
        self.assertEqual(scheduler.config["foo"], "bar")

    @staticmethod
    def _stage():
        stage = Cosmos3TimestepPreparationStage.__new__(Cosmos3TimestepPreparationStage)
        stage.scheduler = types.SimpleNamespace(
            config=types.SimpleNamespace(flow_shift=1.0)
        )
        return stage

    @staticmethod
    def _batch(**kwargs):
        sp_kwargs = kwargs.pop("sp_kwargs", {})
        return types.SimpleNamespace(
            sampling_params=Cosmos3SamplingParams(prompt="t", **sp_kwargs),
            data_type=kwargs.pop("data_type", DataType.VIDEO),
            preprocessed_image=kwargs.pop("preprocessed_image", None),
            preprocessed_video=kwargs.pop("preprocessed_video", None),
        )

    def test_per_mode_flow_shift_defaults(self):
        stage = self._stage()
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(data_type=DataType.IMAGE), is_edge=False
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_image=torch.empty(1)), is_edge=False
            ),
            10.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_video=torch.empty(1)), is_edge=False
            ),
            10.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(self._batch(), is_edge=False), 10.0
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(sp_kwargs={"action_mode": "policy"}), is_edge=False
            ),
            10.0,
        )

    def test_edge_flow_shift_default(self):
        stage = self._stage()
        # Edge uses 3.0 for T2I and every video mode (T2V/I2V/V2V); action stays high.
        self.assertEqual(
            stage._default_flow_shift_for_mode(self._batch(), is_edge=True), 3.0
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(data_type=DataType.IMAGE), is_edge=True
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_image=torch.empty(1)), is_edge=True
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(preprocessed_video=torch.empty(1)), is_edge=True
            ),
            3.0,
        )
        self.assertEqual(
            stage._default_flow_shift_for_mode(
                self._batch(sp_kwargs={"action_mode": "policy"}), is_edge=True
            ),
            10.0,
        )


class TestCosmos3EdgeSamplingDefaults(unittest.TestCase):
    """Edge variant fills its own resolution/guidance defaults; base is untouched."""

    def test_edge_t2v_defaults(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=81)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual((sp.width, sp.height), (832, 480))
        self.assertEqual(sp.guidance_scale, 5.0)

    def test_edge_t2i_defaults(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=1)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual((sp.width, sp.height), (640, 640))
        self.assertEqual(sp.guidance_scale, 7.0)

    def test_edge_restricts_supported_resolutions(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=1)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual(sp.supported_resolutions, COSMOS3_EDGE_SUPPORTED_RESOLUTIONS)
        # Base high-res sizes are excluded so they trip the "unsupported" warning.
        self.assertNotIn((1280, 720), sp.supported_resolutions)
        self.assertNotIn((1024, 1024), sp.supported_resolutions)

    def test_non_edge_defers_resolution_to_base(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=81)
        sp._resolve_variant_defaults(is_edge=False)
        self.assertIsNone(sp.width)
        self.assertIsNone(sp.height)
        self.assertEqual(sp.guidance_scale, 4.0)

    def test_explicit_resolution_preserved_for_edge(self):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=81, width=1024, height=576)
        sp._resolve_variant_defaults(is_edge=True)
        self.assertEqual((sp.width, sp.height), (1024, 576))


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

    def test_policy_adjusts_to_action_output(self):
        params = Cosmos3SamplingParams(
            prompt="test",
            action_mode="policy",
            num_frames=17,
            image_path="observation.png",
        )

        params._adjust(_cosmos3_server_args())

        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertFalse(params.save_output)
        self.assertFalse(params.return_file_paths_only)
        self.assertIsNone(params.output_file_name)
        self.assertEqual(params.num_frames, 17)

    def test_forward_dynamics_remains_video_output(self):
        params = Cosmos3SamplingParams(
            prompt="test",
            action_mode="forward_dynamics",
            num_frames=17,
            image_path="observation.png",
        )

        params._adjust(_cosmos3_server_args())

        self.assertEqual(params.data_type, DataType.VIDEO)


class TestCosmos3ActionEndpoint(unittest.TestCase):
    def test_policy_request_builds_action_sampling_params(self):
        image = torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()
        payload = {
            "request_id": "cosmos-action-1",
            "input": {
                "task": "pick up the block",
                "observation": {
                    "image": {
                        "dtype": "uint8",
                        "shape": [8, 8, 3],
                        "values": image.tolist(),
                    }
                },
            },
            "parameters": {
                "action_mode": "policy",
                "action_horizon": 16,
                "domain_name": "droid_lerobot",
                "num_inference_steps": 30,
                "height": 480,
                "width": 832,
                "fps": 5,
                "seed": 7,
            },
        }

        params = build_action_sampling_params(payload, _cosmos3_server_args())

        self.assertIsInstance(params, Cosmos3SamplingParams)
        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertEqual(params.prompt, "pick up the block")
        self.assertEqual(params.action_mode, "policy")
        self.assertEqual(params.domain_name, "droid_lerobot")
        self.assertEqual(params.num_frames, 17)
        self.assertEqual(params.num_inference_steps, 30)
        self.assertEqual(params.seed, 7)
        self.assertEqual(params.image_path.size, (8, 8))

    def test_inverse_dynamics_maps_video_input(self):
        payload = {
            "input": {
                "task": "infer the robot motion",
                "observation": {"video": "observation.mp4"},
            },
            "parameters": {
                "action_mode": "inverse_dynamics",
                "num_frames": 61,
                "domain_name": "av",
            },
        }

        params = build_action_sampling_params(payload, _cosmos3_server_args())

        self.assertEqual(params.data_type, DataType.ACTION)
        self.assertEqual(params.video_path, "observation.mp4")
        self.assertEqual(params.num_frames, 61)

    def test_forward_dynamics_is_rejected_by_action_endpoint(self):
        payload = {
            "input": {
                "task": "predict the next frames",
                "observation": {"image": "observation.png"},
            },
            "parameters": {"action_mode": "forward_dynamics"},
        }

        with self.assertRaisesRegex(ValueError, "/v1/videos"):
            build_action_sampling_params(payload, _cosmos3_server_args())

    def test_metadata_describes_cosmos_action_contract(self):
        metadata = action_metadata(_cosmos3_server_args())

        self.assertEqual(metadata["model"], "cosmos3-production")
        self.assertEqual(metadata["policy_family"], "cosmos3")
        self.assertEqual(metadata["input"]["modalities"], ["image", "video"])
        self.assertEqual(metadata["output"]["action_horizon"], 16)
        self.assertEqual(metadata["output"]["padded_action_dim"], 64)
        self.assertFalse(metadata["capabilities"]["openpi_websocket"])

    def test_action_response_includes_cosmos_metadata(self):
        output = {
            "request_id": "cosmos-action-2",
            "actions": torch.zeros(16, 10).numpy(),
            "action_mode": "policy",
            "domain_id": 8,
            "raw_action_dim": 10,
            "parameters": {"num_inference_steps": 30},
        }

        response = action_generation_response(output, _cosmos3_server_args())
        action = response["data"][0]["action"]

        self.assertEqual(action["shape"], [16, 10])
        self.assertEqual(action["action_mode"], "policy")
        self.assertEqual(action["domain_id"], 8)
        self.assertEqual(action["raw_action_dim"], 10)
        self.assertEqual(response["usage"]["denoise_steps"], 30)

    def test_action_decode_skips_vae(self):
        class FailIfDecoded:
            def decode(self, _latents):
                raise AssertionError("VAE decode must not run for action output")

        stage = Cosmos3DecodingStage.__new__(Cosmos3DecodingStage)
        stage.vae = FailIfDecoded()
        stage.sound_tokenizer = None
        stage._guardrails = False
        stage.log_info = lambda *_args, **_kwargs: None
        batch = types.SimpleNamespace(
            data_type=DataType.ACTION,
            action_latents=torch.arange(24, dtype=torch.float32).reshape(1, 4, 6),
            extra={
                "raw_action_dim": 3,
                "action_domain_ids": torch.tensor([8]),
            },
            sampling_params=Cosmos3SamplingParams(
                prompt="test",
                action_mode="policy",
                domain_name="droid_lerobot",
            ),
            request_id="cosmos-action-3",
            num_inference_steps=30,
            num_frames=5,
            metrics=None,
        )

        output = stage.forward(batch, types.SimpleNamespace(vae_cpu_offload=False))

        self.assertEqual(output.output[0]["actions"].shape, (4, 3))
        self.assertEqual(output.output[0]["domain_id"], 8)
        self.assertEqual(output.action_pred.shape, (1, 4, 3))


class TestCosmos3ModelResolution(unittest.TestCase):
    """Verify Cosmos3 checkpoints resolve to the native SGLang pipeline."""

    def test_hf_checkpoint_uses_registered_native_pipeline_config(self):
        for model_path in (
            "nvidia/Cosmos3-Nano",
            "nvidia/Cosmos3-Nano-Policy-DROID",
            "nvidia/Cosmos3-Super",
            "nvidia/Cosmos3-Super-Text2Image",
            "nvidia/Cosmos3-Super-Image2Video",
            "nvidia/Cosmos3-Edge",
        ):
            with self.subTest(model_path=model_path):
                self.assertIsNone(get_non_diffusers_pipeline_name(model_path))
                config_info = _get_config_info(model_path)
                self.assertIsNotNone(config_info)
                self.assertIs(config_info.sampling_param_cls, Cosmos3SamplingParams)
                self.assertIs(config_info.pipeline_config_cls, Cosmos3Config)

    def test_class_name_detection_matches_legacy_and_new(self):
        """Unregistered checkpoints resolve via ``_class_name``: both the legacy
        ``Cosmos3OmniDiffusersPipeline`` and the current ``Cosmos3OmniPipeline``
        map to the native Cosmos3 config."""
        for idx, class_name in enumerate(
            ("Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline")
        ):
            model_path = f"acme/mystery-ckpt-{idx}"
            with self.subTest(class_name=class_name):
                with mock.patch(
                    "sglang.multimodal_gen.registry.maybe_download_model_index",
                    return_value={"_class_name": class_name},
                ):
                    config_info = _get_config_info(model_path)
                self.assertIsNotNone(config_info)
                self.assertIs(config_info.pipeline_config_cls, Cosmos3Config)

    def test_legacy_and_new_pipeline_names_both_registered(self):
        """Both ``_class_name`` spellings resolve to a native pipeline class so
        old (Nano/Super) and new (Edge) checkpoints load."""
        _discover_and_register_pipelines()
        for pipeline_name in ("Cosmos3OmniPipeline", "Cosmos3OmniDiffusersPipeline"):
            with self.subTest(pipeline_name=pipeline_name):
                self.assertIn(pipeline_name, _PIPELINE_REGISTRY)


class TestCosmos3OpenAIProtocol(unittest.TestCase):
    """Verify Cosmos3 modality knobs stay model-specific video extras."""

    def test_cosmos3_template_fields_remain_extra_fields(self):
        for request_cls in (ImageGenerationsRequest, VideoGenerationsRequest):
            with self.subTest(request_cls=request_cls.__name__):
                self.assertIn("max_sequence_length", request_cls.model_fields)
                self.assertIn("flow_shift", request_cls.model_fields)
                self.assertNotIn("use_duration_template", request_cls.model_fields)
                self.assertNotIn("use_resolution_template", request_cls.model_fields)
                self.assertNotIn("use_system_prompt", request_cls.model_fields)
                self.assertNotIn("use_guardrails", request_cls.model_fields)

    def test_cosmos3_modal_fields_are_model_specific_video_extras(self):
        for field_name in (
            "generate_sound",
            "sound_duration",
            "condition_frame_indexes",
            "condition_frame_indexes_vision",
            "condition_video_keep",
            "control_path",
            "control_hint",
            "control_guidance",
            "control_guidance_interval",
            "action_mode",
            "domain_id",
            "domain_name",
            "raw_action_dim",
            "action_fps",
            "action",
            "action_view_point",
            "action_normalization",
        ):
            with self.subTest(field_name=field_name):
                self.assertNotIn(field_name, VideoGenerationsRequest.model_fields)
                self.assertIn(
                    field_name, Cosmos3SamplingParams.video_request_extra_fields()
                )

        self.assertIn("video_path", VideoGenerationsRequest.model_fields)
        self.assertIn("video_url", VideoGenerationsRequest.model_fields)
        self.assertNotIn("action_stats_path", VideoGenerationsRequest.model_fields)
        self.assertNotIn(
            "action_stats_path", Cosmos3SamplingParams.video_request_extra_fields()
        )

    def test_cosmos3_http_aliases_map_to_sampling_params(self):
        req = VideoGenerationsRequest(
            prompt="test",
            video_url="https://example.com/input.mp4",
            generate_sound=True,
            condition_frame_indexes_vision=[0, 2],
            condition_video_keep="last",
            action_mode="policy",
            domain_name="umi",
            raw_action_dim=9,
            action_fps=30.0,
            action_view_point="ego_view",
            control_path=["edge.mp4", "depth.mp4"],
            control_hint=["edge", "depth"],
            control_guidance=1.5,
            control_guidance_interval=[0.0, 500.0],
        )

        self.assertEqual(_resolve_video_path(req), "https://example.com/input.mp4")

        kwargs = _cosmos3_sampling_param_kwargs(req, num_frames=48, fps=24)
        self.assertEqual(kwargs["sound_duration"], 2.0)
        self.assertEqual(kwargs["condition_frame_indexes"], [0, 2])
        self.assertEqual(kwargs["condition_video_keep"], "last")
        self.assertEqual(kwargs["action_mode"], "policy")
        self.assertEqual(kwargs["domain_name"], "umi")
        self.assertEqual(kwargs["raw_action_dim"], 9)
        self.assertEqual(kwargs["action_fps"], 30.0)
        self.assertEqual(kwargs["action_view_point"], "ego_view")
        self.assertEqual(kwargs["control_path"], ["edge.mp4", "depth.mp4"])
        self.assertEqual(kwargs["control_hint"], ["edge", "depth"])
        self.assertEqual(kwargs["control_guidance"], 1.5)
        self.assertEqual(kwargs["control_guidance_interval"], (0.0, 500.0))

    def test_cosmos3_multipart_extras_are_model_specific(self):
        raw_form = {
            "generate_sound": "true",
            "control_path": '["edge.mp4", "depth.mp4"]',
            "control_hint": '["edge", "depth"]',
            "action_mode": "policy",
            "action_stats_path": "/tmp/action_stats.json",
        }

        generic = _multipart_video_extras(
            raw_form,
            extra_body=None,
            extra_params=None,
            sampling_params_cls=SamplingParams,
        )
        self.assertNotIn("generate_sound", generic)
        self.assertNotIn("control_path", generic)
        self.assertNotIn("action_mode", generic)

        cosmos = _multipart_video_extras(
            raw_form,
            extra_body=None,
            extra_params=None,
            sampling_params_cls=Cosmos3SamplingParams,
        )
        self.assertIs(cosmos["generate_sound"], True)
        self.assertEqual(cosmos["control_path"], ["edge.mp4", "depth.mp4"])
        self.assertEqual(cosmos["control_hint"], ["edge", "depth"])
        self.assertEqual(cosmos["action_mode"], "policy")
        self.assertNotIn("action_stats_path", cosmos)

    def test_generate_sound_false_disables_sound_duration(self):
        req = VideoGenerationsRequest(
            prompt="test", generate_sound=False, sound_duration=3.0
        )
        self.assertEqual(
            _resolve_sound_duration(req, num_frames=48, fps=24),
            0.0,
        )


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


class TestCosmos3MRoPE(unittest.TestCase):
    """mRoPE position-ID computation for vision / sound / action token grids."""

    DEVICE = torch.device("cpu")

    def test_vision_default_args_unchanged(self):
        # With base_temporal_compression_factor=None and start_frame_offset=0
        # the token and base rates cancel, so t-index == frame index.
        pos, _ = compute_mrope_position_ids_vision(
            grid_t=21,
            grid_h=1,
            grid_w=1,
            temporal_offset=0,
            device=self.DEVICE,
            fps=24.0,
            base_fps=24.0,
            temporal_compression_factor=4,
        )
        self.assertEqual(tuple(pos.shape), (3, 21))
        self.assertAlmostEqual(float(pos[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(pos[0, 20]), 20.0, places=5)

    def test_sound_grid_shape_and_scaling(self):
        pos, _ = compute_mrope_position_ids_sound(
            grid_t=10,
            temporal_offset=0,
            sound_latent_fps=25.0,
            device=self.DEVICE,
            base_fps=24.0,
            temporal_compression_factor_sound=1,
        )
        # (T, 1, 1) grid -> spatial axes are all zero.
        self.assertEqual(tuple(pos.shape), (3, 10))
        self.assertTrue(torch.all(pos[1] == 0))
        self.assertTrue(torch.all(pos[2] == 0))
        # t-index = i / sound_fps * base_fps = i / 25 * 24.
        self.assertAlmostEqual(float(pos[0, 5]), 5 / 25 * 24, places=4)

    def test_action_uses_video_base_compression_and_offset(self):
        # Action runs at frame rate (tcf=1) but is scaled by the video's
        # base_temporal_compression_factor=4, and shifted by start_frame_offset.
        pos, _ = compute_mrope_position_ids_action(
            grid_t=16,
            temporal_offset=0,
            action_fps=10.0,
            device=self.DEVICE,
            base_fps=24.0,
            base_temporal_compression_factor=4,
            start_frame_offset=1,
        )
        self.assertEqual(tuple(pos.shape), (3, 16))
        self.assertTrue(torch.all(pos[1] == 0))
        self.assertTrue(torch.all(pos[2] == 0))
        # t-index[i] = (i + start_frame_offset) / action_fps * (base_fps / base_tcf)
        #            = (i + 1) / 10 * (24 / 4) = (i + 1) * 0.6
        self.assertAlmostEqual(float(pos[0, 0]), 0.6, places=4)
        self.assertAlmostEqual(float(pos[0, 15]), 16 * 0.6, places=4)

    def test_action_offset_zero(self):
        pos, _ = compute_mrope_position_ids_action(
            grid_t=8,
            temporal_offset=0,
            action_fps=10.0,
            device=self.DEVICE,
            base_fps=24.0,
            base_temporal_compression_factor=4,
            start_frame_offset=0,
        )
        self.assertAlmostEqual(float(pos[0, 0]), 0.0, places=5)

    def test_action_aligns_with_video_positions(self):
        # Action frames at frame rate should share the video's temporal frame:
        # every 4th action token lands on the next video latent-frame position.
        media_offset = 100
        vid, _ = compute_mrope_position_ids_vision(
            grid_t=5,
            grid_h=1,
            grid_w=1,
            temporal_offset=media_offset,
            device=self.DEVICE,
            fps=24.0,
            base_fps=24.0,
            temporal_compression_factor=4,
        )
        act, _ = compute_mrope_position_ids_action(
            grid_t=16,
            temporal_offset=media_offset,
            action_fps=24.0,
            device=self.DEVICE,
            base_fps=24.0,
            base_temporal_compression_factor=4,
            start_frame_offset=0,
        )
        # video latent frame 1 sits at media_offset+1; action frame 4 (4 frames
        # per latent at tcf=4) lands at the same temporal position.
        self.assertAlmostEqual(float(vid[0, 1]), float(act[0, 4]), places=4)


class TestCosmos3Transfer(unittest.TestCase):
    """Transfer (control-video) conditioning: rope packing, control-CFG, defaults."""

    DEVICE = torch.device("cpu")

    @staticmethod
    def _transformer_self() -> Cosmos3OmniTransformer:
        model = Cosmos3OmniTransformer.__new__(Cosmos3OmniTransformer)
        model.temporal_margin = 15000
        model.base_fps = 24.0
        model.temporal_compression_factor = 4
        model.sound_latent_fps = 25.0
        model.temporal_compression_factor_sound = 1
        return model

    def test_single_control_shares_video_positions(self):
        model = self._transformer_self()
        text_mask = torch.ones(1, 4)
        T, Hp, Wp = 3, 2, 2
        tpc = T * Hp * Wp

        _, base = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=0
        )
        _, with_ctrl = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=T
        )
        self.assertEqual(tuple(with_ctrl.shape), (3, 1, 2 * tpc))
        # control prefix == video block, and video block is unchanged.
        self.assertTrue(torch.equal(with_ctrl[:, :, :tpc], with_ctrl[:, :, tpc:]))
        self.assertTrue(torch.equal(with_ctrl[:, :, tpc:], base))

    def test_multi_control_prepended_in_order(self):
        model = self._transformer_self()
        text_mask = torch.ones(1, 4)
        T, Hp, Wp = 3, 2, 2
        tpc = T * Hp * Wp

        _, base = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=0
        )
        _, multi = model._compute_rope_position_ids(
            text_mask, T, Hp, Wp, fps=None, device=self.DEVICE, control_frames=[T, T]
        )
        self.assertEqual(tuple(multi.shape), (3, 1, 3 * tpc))
        c0 = multi[:, :, :tpc]
        c1 = multi[:, :, tpc : 2 * tpc]
        vid = multi[:, :, 2 * tpc :]
        self.assertTrue(torch.equal(c0, vid))
        self.assertTrue(torch.equal(c1, vid))
        self.assertTrue(torch.equal(vid, base))

    def test_control_cfg_blend_math(self):
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)

        # Per-branch forward values (each run un-batched, bs=1): cond_full
        # (control in, cond text) -> 20; cond_nc (control dropped) -> 10; uncond
        # (control in, uncond text) -> 2. The unified executor reduces the
        # coefficient-weighted branch sum built by _control_cfg_branches.
        def fake_run(**kw):
            bs = kw["latents"].shape[0]
            if kw["control_latents"] is None:
                return torch.full((bs,), 10.0)  # cond_nc, control dropped
            if kw["cache_key"] == "uncond":
                return torch.full((bs,), 2.0)  # uncond, control in
            return torch.full((bs,), 20.0)  # cond_full, control in

        stage._run_transformer = fake_run
        cond_text_ids = torch.zeros(1)
        cond_text_mask = torch.ones(1)
        uncond_text_ids = torch.zeros(1)
        uncond_text_mask = torch.ones(1)
        control_latents = [torch.zeros(1)]

        def _run(text_g, control_g):
            branches = stage._control_cfg_branches(
                cond_text_ids,
                cond_text_mask,
                uncond_text_ids,
                uncond_text_mask,
                cond_text_seq_len=None,
                uncond_text_seq_len=None,
                control_latents=control_latents,
                text_guidance_scale=text_g,
                control_guidance_scale=control_g,
            )
            return stage._predict_noise_cfg(
                branches,
                latents=torch.zeros(1),
                timestep=torch.zeros(1),
                video_shape=(1, 1, 1),
                fps=24.0,
                cfg_rank=0,
                cfg_world_size=1,
            )

        # control-CFG only (g=1): cg*cond_full + (1-cg)*cond_nc
        #   = 2*20 + (-1)*10 = 30
        out = _run(text_g=1.0, control_g=2.0)
        self.assertTrue(torch.allclose(out, torch.full((1,), 30.0)))

        # control-CFG + text CFG (g=3): g*cg*cond_full + g*(1-cg)*cond_nc
        #   + (1-g)*uncond = 6*20 + (-3)*10 + (-2)*2 = 86
        out2 = _run(text_g=3.0, control_g=2.0)
        self.assertTrue(torch.allclose(out2, torch.full((1,), 86.0)))

    def test_text_cfg_batched_single_gpu(self):
        """Single-GPU text CFG batches both branches into one bs=2 forward."""
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)

        def fake_run(**kw):
            self.assertIsNone(kw["control_latents"])
            self.assertEqual(kw["latents"].shape[0], 2)  # batched [uncond, cond]
            return torch.tensor([2.0, 20.0])

        stage._run_transformer = fake_run
        out = stage._predict_noise_cfg_batched(
            latents=torch.zeros(1),
            timestep=torch.zeros(1),
            cond_text_ids=torch.zeros(1),
            cond_text_mask=torch.ones(1),
            uncond_text_ids=torch.zeros(1),
            uncond_text_mask=torch.ones(1),
            video_shape=(1, 1, 1),
            fps=24.0,
            guidance_scale=3.0,
        )
        # uncond + g*(cond - uncond) = 2 + 3*(20-2) = 56
        self.assertTrue(torch.allclose(out, torch.full((1,), 56.0)))

    def test_control_cfg_parallel_distribution(self):
        """Round-robin branch distribution + all-reduce equals the serial blend."""
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)
        seen = {}

        def make_run(rank):
            seen[rank] = []

            def fake_run(**kw):
                seen[rank].append(kw["cache_key"])
                bs = kw["latents"].shape[0]
                if kw["control_latents"] is None:
                    return torch.full((bs,), 10.0)  # cond_nc
                if kw["cache_key"] == "uncond":
                    return torch.full((bs,), 2.0)  # uncond
                return torch.full((bs,), 20.0)  # cond_full

            return fake_run

        def run(rank, world):
            stage._run_transformer = make_run(rank)
            branches = stage._control_cfg_branches(
                torch.zeros(1),
                torch.ones(1),
                torch.zeros(1),
                torch.ones(1),
                cond_text_seq_len=None,
                uncond_text_seq_len=None,
                control_latents=[torch.zeros(1)],
                text_guidance_scale=3.0,
                control_guidance_scale=2.0,
            )
            return stage._predict_noise_cfg(
                branches,
                latents=torch.zeros(1),
                timestep=torch.zeros(1),
                video_shape=(1, 1, 1),
                fps=24.0,
                cfg_rank=rank,
                cfg_world_size=world,
            )

        target = (
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.cosmos3.cfg_model_parallel_all_reduce"
        )
        # Stand in for the all-reduce with identity so each rank returns its own
        # partial; summing them mimics the cross-rank reduction.
        with mock.patch(target, side_effect=lambda x: x):
            # 2 ranks: rank 0 runs the two control-in forwards (cond_full +
            # uncond), rank 1 runs only the control-dropped cond_nc.
            r0 = run(0, 2)
            r1 = run(1, 2)
            self.assertEqual(seen[0], ["cond", "uncond"])
            self.assertEqual(seen[1], ["cond_nc"])
            self.assertTrue(torch.allclose(r0 + r1, torch.full((1,), 86.0)))

            seen.clear()
            # 4 ranks for 3 branches: rank 3 is idle and contributes zeros.
            parts = [run(r, 4) for r in range(4)]
            self.assertEqual(seen[0], ["cond"])
            self.assertEqual(seen[1], ["cond_nc"])
            self.assertEqual(seen[2], ["uncond"])
            self.assertEqual(seen[3], [])
            self.assertTrue(torch.allclose(parts[3], torch.zeros(1)))
            self.assertTrue(torch.allclose(sum(parts), torch.full((1,), 86.0)))

    def test_single_hint_defaults_applied(self):
        sp = Cosmos3SamplingParams(
            prompt="t", control_path="edge.mp4", control_hint="edge"
        )
        sp._explicit_fields = {"prompt", "control_path", "control_hint"}
        sp._apply_transfer_hint_defaults()
        self.assertEqual(sp.control_guidance, 1.5)
        self.assertEqual(sp.guidance_scale, 3.0)
        self.assertEqual(sp.flow_shift, 10.0)

    def test_explicit_value_overrides_hint_default(self):
        sp = Cosmos3SamplingParams(
            prompt="t",
            control_path="seg.mp4",
            control_hint="seg",
            control_guidance=4.2,
        )
        sp._explicit_fields = {"control_path", "control_hint", "control_guidance"}
        sp._apply_transfer_hint_defaults()
        self.assertEqual(sp.control_guidance, 4.2)

    def test_multi_hint_defaults_not_applied(self):
        sp = Cosmos3SamplingParams(
            prompt="t",
            control_path=["edge.mp4", "depth.mp4"],
            control_hint=["edge", "depth"],
        )
        base_guidance = sp.guidance_scale
        sp._explicit_fields = {"control_path", "control_hint"}
        sp._apply_transfer_hint_defaults()
        self.assertEqual(sp.control_guidance, 1.0)
        self.assertEqual(sp.guidance_scale, base_guidance)

    def test_unknown_hint_rejected(self):
        sp = Cosmos3SamplingParams(
            prompt="t", control_path="x.mp4", control_hint="bogus"
        )
        with self.assertRaises(ValueError):
            sp._resolve_control_hints()


class TestCosmos3DomainAwareLinear(unittest.TestCase):
    """Per-domain action projection."""

    def test_rank3_and_rank2_shapes(self):
        layer = DomainAwareLinear(input_size=7, output_size=64, num_domains=32)
        x3 = torch.randn(2, 16, 7)
        out3 = layer(x3, torch.tensor([1, 5]))
        self.assertEqual(tuple(out3.shape), (2, 16, 64))
        x2 = torch.randn(3, 7)
        out2 = layer(x2, torch.tensor([0, 1, 2]))
        self.assertEqual(tuple(out2.shape), (3, 64))

    def test_distinct_domains_give_distinct_outputs(self):
        torch.manual_seed(0)
        layer = DomainAwareLinear(input_size=4, output_size=8, num_domains=4)
        x = torch.randn(1, 3, 4)
        out_a = layer(x, torch.tensor([0]))
        out_b = layer(x, torch.tensor([2]))
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_scalar_domain_id_promoted(self):
        layer = DomainAwareLinear(input_size=4, output_size=8, num_domains=4)
        out = layer(torch.randn(1, 2, 4), torch.tensor(3))
        self.assertEqual(tuple(out.shape), (1, 2, 8))


class TestCosmos3ConditionIndexes(unittest.TestCase):
    """Vision condition-frame resolution across V2V and action modes."""

    @staticmethod
    def _batch(num_frames=61, **sp_kwargs):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=num_frames, **sp_kwargs)
        return types.SimpleNamespace(sampling_params=sp, num_frames=num_frames)

    def test_v2v_default(self):
        idx = Cosmos3ImagePreprocessStage._resolve_condition_indexes(self._batch())
        self.assertEqual(idx, [0, 1])

    def test_v2v_explicit_sorted_unique(self):
        idx = Cosmos3ImagePreprocessStage._resolve_condition_indexes(
            self._batch(condition_frame_indexes=[2, 0, 2])
        )
        self.assertEqual(idx, [0, 2])

    def test_inverse_dynamics_conditions_all_latent_frames(self):
        # 61 frames -> (61-1)//4 + 1 = 16 latent frames, all locked.
        idx = Cosmos3ImagePreprocessStage._resolve_condition_indexes(
            self._batch(num_frames=61, action_mode="inverse_dynamics")
        )
        self.assertEqual(idx, list(range(16)))


class TestCosmos3DomainResolution(unittest.TestCase):
    """Embodiment domain-id resolution for action generation."""

    @staticmethod
    def _batch(**sp_kwargs):
        sp = Cosmos3SamplingParams(prompt="t", **sp_kwargs)
        return types.SimpleNamespace(sampling_params=sp)

    def test_explicit_domain_id(self):
        self.assertEqual(
            Cosmos3LatentPreparationStage._resolve_domain_id(self._batch(domain_id=7)),
            7,
        )

    def test_domain_name_lookup(self):
        self.assertEqual(
            Cosmos3LatentPreparationStage._resolve_domain_id(
                self._batch(domain_name="av")
            ),
            EMBODIMENT_TO_DOMAIN_ID["av"],
        )
        self.assertEqual(
            Cosmos3LatentPreparationStage._resolve_domain_id(
                self._batch(domain_name="umi")
            ),
            EMBODIMENT_TO_DOMAIN_ID["umi"],
        )

    def test_missing_domain_raises(self):
        with self.assertRaises(ValueError):
            Cosmos3LatentPreparationStage._resolve_domain_id(self._batch())

    def test_unknown_domain_name_raises(self):
        with self.assertRaises(ValueError):
            Cosmos3LatentPreparationStage._resolve_domain_id(
                self._batch(domain_name="not_a_robot")
            )


class TestCosmos3ActionLatentPrep(unittest.TestCase):
    """Action latent / mask preparation per action mode."""

    @classmethod
    def setUpClass(cls):
        # Bypass PipelineStage.__init__ (needs global server args); only the
        # transformer's action_dim and log_info are used by _prepare_action_latents.
        cls.stage = Cosmos3LatentPreparationStage.__new__(Cosmos3LatentPreparationStage)
        cls.stage.transformer = types.SimpleNamespace(action_dim=64)
        cls.stage.log_info = lambda *a, **k: None
        cls.device = torch.device("cpu")
        cls.dtype = torch.float32

    def _run(self, num_frames=17, **sp_kwargs):
        sp = Cosmos3SamplingParams(prompt="t", num_frames=num_frames, **sp_kwargs)
        batch = types.SimpleNamespace(
            sampling_params=sp, num_frames=num_frames, extra={}
        )
        gen = torch.Generator(device=self.device).manual_seed(0)
        self.stage._prepare_action_latents(batch, gen, self.device, self.dtype)
        return batch

    def test_forward_dynamics_clean_conditioning(self):
        batch = self._run(
            action_mode="forward_dynamics",
            domain_name="agibotworld",
            action=[[0.1] * 29 for _ in range(16)],
        )
        # action_chunk_size = num_frames - 1 = 16; padded to action_dim 64.
        self.assertEqual(tuple(batch.action_latents.shape), (1, 16, 64))
        # raw_action_dim is derived from the embodiment (agibotworld -> 29).
        self.assertEqual(batch.extra["raw_action_dim"], 29)
        self.assertEqual(batch.extra["action_start_frame_offset"], 1)
        self.assertEqual(
            int(batch.extra["action_domain_ids"][0]),
            EMBODIMENT_TO_DOMAIN_ID["agibotworld"],
        )
        # forward_dynamics: action is clean conditioning -> velocity mask all zero.
        self.assertTrue(torch.all(batch.extra["action_velocity_mask"] == 0))

    def test_policy_denoises_from_noise(self):
        batch = self._run(
            action_mode="policy", domain_name="droid_lerobot", raw_action_dim=10
        )
        self.assertEqual(tuple(batch.action_latents.shape), (1, 16, 64))
        self.assertEqual(batch.extra["raw_action_dim"], 10)
        # policy: action fully denoised -> velocity mask all one.
        self.assertTrue(torch.all(batch.extra["action_velocity_mask"] == 1))
        # padding dims beyond raw_action_dim start at zero.
        self.assertTrue(torch.all(batch.action_latents[:, :, 10:] == 0))

    def test_inverse_dynamics_denoises_from_noise(self):
        batch = self._run(
            num_frames=61,
            action_mode="inverse_dynamics",
            domain_name="av",
            raw_action_dim=9,
        )
        self.assertEqual(tuple(batch.action_latents.shape), (1, 60, 64))
        self.assertTrue(torch.all(batch.extra["action_velocity_mask"] == 1))

    def test_forward_dynamics_requires_action(self):
        with self.assertRaises(ValueError):
            self._run(action_mode="forward_dynamics", domain_name="agibotworld")

    def test_policy_requires_raw_action_dim(self):
        # policy has no input action to infer from, so it needs raw_action_dim
        # from either the embodiment or an explicit value. With only a numeric
        # domain_id (no embodiment name) and no raw_action_dim, it must raise.
        with self.assertRaises(ValueError):
            self._run(action_mode="policy", domain_id=0)

    def test_policy_raw_action_dim_from_embodiment(self):
        # droid_lerobot -> 10, so policy no longer needs an explicit raw dim.
        batch = self._run(action_mode="policy", domain_name="droid_lerobot")
        self.assertEqual(batch.extra["raw_action_dim"], 10)

    def test_unknown_action_mode_raises(self):
        with self.assertRaises(ValueError):
            self._run(action_mode="teleport", domain_id=0)


class TestCosmos3ModalitySamplingParams(unittest.TestCase):
    """Sound / V2V / action sampling-param fields and defaults."""

    def test_sound_duration_default_and_set(self):
        self.assertEqual(Cosmos3SamplingParams(prompt="t").sound_duration, 0.0)
        self.assertEqual(
            Cosmos3SamplingParams(prompt="t", sound_duration=3.0).sound_duration, 3.0
        )

    def test_v2v_fields(self):
        sp = Cosmos3SamplingParams(
            prompt="t", video_path="in.mp4", condition_frame_indexes=[0, 1]
        )
        self.assertEqual(sp.video_path, "in.mp4")
        self.assertEqual(sp.condition_frame_indexes, [0, 1])
        self.assertEqual(sp.condition_video_keep, "first")

    def test_action_fields_default_none(self):
        sp = Cosmos3SamplingParams(prompt="t")
        for field in (
            "action_mode",
            "domain_id",
            "domain_name",
            "raw_action_dim",
            "action_fps",
            "action",
        ):
            self.assertIsNone(getattr(sp, field))


class TestCosmos3CaptionMetadata(unittest.TestCase):
    """Structured captions get generation metadata; prose prompts opt out."""

    def test_video_caption_gets_resolution_duration_and_fps(self):
        prompt = json.dumps({"temporal_caption": "a cone melts"})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=189, fps=24.0, height=480, width=832
            )
        )
        self.assertEqual(caption["resolution"], {"H": 480, "W": 832})
        # Whole seconds, matching the caption format the model was trained on.
        self.assertEqual(caption["duration"], "7s")
        self.assertEqual(caption["fps"], 24.0)
        self.assertEqual(caption["temporal_caption"], "a cone melts")

    def test_image_caption_drops_temporal_fields(self):
        prompt = json.dumps({"subjects": ["hands"], "duration": "5s", "fps": 24.0})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=1, fps=24.0, height=768, width=768
            )
        )
        self.assertEqual(caption["resolution"], {"H": 768, "W": 768})
        self.assertNotIn("duration", caption)
        self.assertNotIn("fps", caption)
        self.assertEqual(caption["subjects"], ["hands"])

    def test_request_resolution_overrides_caption(self):
        prompt = json.dumps({"resolution": {"H": 111, "W": 222}})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=1, fps=24.0, height=640, width=640
            )
        )
        self.assertEqual(caption["resolution"], {"H": 640, "W": 640})

    def test_unrelated_caption_fields_are_preserved(self):
        prompt = json.dumps({"aspect_ratio": "1,1", "subjects": ["a vase"]})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=1, fps=24.0, height=640, width=640
            )
        )
        self.assertEqual(caption["aspect_ratio"], "1,1")
        self.assertEqual(caption["subjects"], ["a vase"])

    def test_zero_fps_does_not_divide_by_zero(self):
        prompt = json.dumps({"temporal_caption": "a cone melts"})
        caption = json.loads(
            _inject_caption_metadata(
                prompt, num_frames=81, fps=0.0, height=480, width=832
            )
        )
        self.assertEqual(caption["duration"], "0s")

    def test_non_structured_prompts_are_declined(self):
        for prompt in (
            "A curious raccoon in a field of sunflowers.",
            json.dumps(["not", "an", "object"]),
            json.dumps("a bare string"),
            "",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(
                    _inject_caption_metadata(
                        prompt, num_frames=81, fps=24.0, height=480, width=832
                    )
                )


class TestCosmos3DurationTemplateSuppression(unittest.TestCase):
    """A structured caption must not also get the prose duration suffix."""

    def _run_prompt_stage(self, prompt: str, use_duration_template: bool) -> str:
        """Drive Cosmos3TokenizationStage.forward far enough to see the prompt."""
        seen = {}

        def fake_tokenize(prompt_text, *args, **kwargs):
            seen.setdefault("prompt", prompt_text)
            return (
                torch.zeros(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
                4,
            )

        stage = Cosmos3TokenizationStage.__new__(Cosmos3TokenizationStage)
        batch = types.SimpleNamespace(
            prompt=prompt,
            negative_prompt="bad",
            max_sequence_length=512,
            use_duration_template=use_duration_template,
            use_system_prompt=False,
            fps=24.0,
            num_frames=189,
            height=480,
            width=832,
            data_type=DataType.VIDEO,
            sampling_params=types.SimpleNamespace(action_mode=None),
            extra={},
        )
        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(
                use_duration_template=True, use_system_prompt=False
            )
        )
        with (
            mock.patch.object(stage, "_tokenize_prompt", fake_tokenize),
            mock.patch.object(stage, "log_info"),
        ):
            stage.forward(batch, server_args)
        return seen["prompt"]

    def test_structured_caption_stays_valid_json(self):
        prompt = json.dumps({"temporal_caption": "a cone melts"})
        final = self._run_prompt_stage(prompt, use_duration_template=True)
        caption = json.loads(final)  # would raise if the suffix were appended
        self.assertEqual(caption["duration"], "7s")
        self.assertNotIn("seconds long", final)

    def test_prose_prompt_still_gets_the_duration_suffix(self):
        final = self._run_prompt_stage("A curious raccoon.", use_duration_template=True)
        self.assertIn("seconds long", final)


if __name__ == "__main__":
    unittest.main()
