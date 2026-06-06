import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from sglang.multimodal_gen.configs.models.dits.ideogram import Ideogram4DiTConfig
from sglang.multimodal_gen.configs.models.encoders.ideogram import (
    Ideogram4TextEncoderConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ideogram import (
    Ideogram4PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.ideogram import (
    IDEOGRAM4_PRESETS,
    Ideogram4SamplingParams,
)
from sglang.multimodal_gen.registry import _get_config_info, get_model_info
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType, get_module_role
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.bitsandbytes import (
    _maybe_shard_bitsandbytes_4bit_quant_state,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
    FP8_WEIGHT_DTYPE,
    WeightOnlyFP8ColumnParallelLinear,
    WeightOnlyFP8Linear,
    dequantize_rowwise_fp8_weight,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    Qwen3VLTextRotaryEmbedding,
    qwen3_apply_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
    _server_args_for_transformer_component,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.ideogram import (
    Ideogram4Transformer2DModel,
)
from sglang.multimodal_gen.runtime.models.encoders.ideogram import (
    IdeogramQwen3VLTextEncoder,
)
from sglang.multimodal_gen.runtime.pipelines.ideogram import (
    _resolve_ideogram4_unconditional_transformer_weights_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ideogram import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4DecodingStage,
    Ideogram4DenoisingStage,
    Ideogram4TextEncodingStage,
    make_step_intervals,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


def _reference_qwen3_mrope(position_ids, head_dim, rope_theta, mrope_section):
    batch_size = position_ids.shape[0]
    pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    inv_freq = inv_freq[None, None, :, None].expand(3, batch_size, -1, 1)
    freqs = inv_freq @ pos.unsqueeze(2)
    freqs = freqs.transpose(2, 3)
    freqs_t = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = mrope_section[axis] * 3
        idx = torch.arange(offset, length, 3, device=freqs_t.device)
        freqs_t[..., idx] = freqs[axis][..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    return emb.cos(), emb.sin()


class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return messages[0]["content"][0]["text"]

    def __call__(self, text, return_tensors, add_special_tokens):
        values = [int(x) for x in text.split()]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}


class FakeIdeogramTransformer(torch.nn.Module):
    def forward(self, *, x, **kwargs):
        return torch.zeros_like(x)


class FakeIdeogramVAE(torch.nn.Module):
    def decode(self, z):
        return z[:, :3]


class FakeIdeogramPipeline:
    def __init__(self, transformer, unconditional_transformer):
        self.modules = {
            "transformer": transformer,
            "unconditional_transformer": unconditional_transformer,
        }


class FakeBnbQuantState:
    def __init__(
        self,
        absmax,
        shape=None,
        code=None,
        blocksize=None,
        quant_type=None,
        dtype=None,
        offset=None,
        state2=None,
    ):
        self.absmax = absmax
        self.shape = shape
        self.code = code
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.dtype = dtype
        self.offset = offset
        self.state2 = state2
        self.nested = state2 is not None


def _fake_server_args(cfg=None):
    return SimpleNamespace(
        pipeline_config=cfg or Ideogram4PipelineConfig(),
        comfyui_mode=False,
        enable_torch_compile=False,
        attention_backend="torch_sdpa",
        enable_layerwise_nvtx_marker=False,
        model_loaded={"transformer": True},
        model_paths={},
        disable_autocast=False,
        enable_cfg_parallel=False,
        attention_backend_config=None,
    )


def _fake_ideogram_pipeline(transformer, unconditional_transformer):
    return FakeIdeogramPipeline(transformer, unconditional_transformer)


class TestIdeogram4(unittest.TestCase):
    def test_registry_resolves_model_index_class_name(self):
        get_model_info.cache_clear()
        _get_config_info.cache_clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/model_index.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"_class_name": "Ideogram4Pipeline", "_diffusers_version": "0.0.0"},
                    f,
                )
            for subdir in (
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "unconditional_transformer",
                "vae",
            ):
                os.mkdir(f"{tmpdir}/{subdir}")
            info = get_model_info(tmpdir, backend="sglang")
        self.assertEqual(info.pipeline_cls.__name__, "Ideogram4Pipeline")
        self.assertIs(info.pipeline_config_cls, Ideogram4PipelineConfig)
        self.assertIs(info.sampling_param_cls, Ideogram4SamplingParams)

    def test_registry_resolves_comfy_nvfp4_repo_to_native_pipeline(self):
        get_model_info.cache_clear()
        _get_config_info.cache_clear()

        info = get_model_info("Comfy-Org/Ideogram-4", backend="sglang")

        self.assertEqual(info.pipeline_cls.__name__, "Ideogram4Nvfp4Pipeline")
        self.assertIs(info.pipeline_config_cls, Ideogram4PipelineConfig)
        self.assertIs(info.sampling_param_cls, Ideogram4SamplingParams)

    def test_registry_resolves_official_nf4_repo_to_native_pipeline(self):
        get_model_info.cache_clear()
        _get_config_info.cache_clear()

        with patch(
            "sglang.multimodal_gen.registry.maybe_download_model_index",
            return_value={
                "_class_name": "Ideogram4Pipeline",
                "_diffusers_version": "0.0.0",
            },
        ):
            info = get_model_info("ideogram-ai/ideogram-4-nf4", backend="sglang")

        self.assertEqual(info.pipeline_cls.__name__, "Ideogram4Pipeline")
        self.assertIs(info.pipeline_config_cls, Ideogram4PipelineConfig)
        self.assertIs(info.sampling_param_cls, Ideogram4SamplingParams)

    def test_rowwise_fp8_dequant_uses_output_channel_scale(self):
        weight = torch.tensor(
            [[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=FP8_WEIGHT_DTYPE
        )
        scale = torch.tensor([0.5, 2.0], dtype=torch.float32)
        actual = dequantize_rowwise_fp8_weight(weight, scale, torch.float32)
        expected = weight.to(torch.float32) * scale[:, None]
        torch.testing.assert_close(actual, expected)

    def test_shared_qwen3_mrope_matches_ideogram_reference_layout(self):
        position_ids = torch.tensor(
            [
                [[0, 0, 0], [1, 1, 1], [65536, 65536, 65536]],
                [[0, 0, 0], [0, 2, 3], [65536, 65537, 65538]],
            ],
            dtype=torch.long,
        )
        head_dim = 8
        rope_theta = 5_000_000.0
        mrope_section = (2, 1, 1)
        rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )

        cos, sin = rotary_emb(torch.empty((), dtype=torch.float32), position_ids)
        ref_cos, ref_sin = _reference_qwen3_mrope(
            position_ids, head_dim, rope_theta, mrope_section
        )

        torch.testing.assert_close(cos.float(), ref_cos)
        torch.testing.assert_close(sin.float(), ref_sin)

    def test_usp_attention_key_mask_matches_segment_mask_for_valid_tokens(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
            )
            torch.manual_seed(0)
            batch_size, seq_len, num_heads, head_dim = 2, 5, 2, 8
            q = torch.randn(batch_size, seq_len, num_heads, head_dim)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim)
            segment_ids = torch.tensor(
                [[-1, -1, 1, 1, 1], [-1, 1, 1, 1, 1]], dtype=torch.long
            )
            position_ids = torch.stack(
                [
                    torch.arange(seq_len).expand(batch_size, -1),
                    torch.arange(seq_len).expand(batch_size, -1) + 1,
                    torch.arange(seq_len).expand(batch_size, -1) + 2,
                ],
                dim=-1,
            )
            rotary_emb = Qwen3VLTextRotaryEmbedding(
                head_dim=head_dim, mrope_section=(2, 1, 1)
            )
            cos, sin = rotary_emb(q, position_ids)
            q, k = qwen3_apply_rotary_pos_emb(q, k, cos.unsqueeze(2), sin.unsqueeze(2))

            full_mask = (
                segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
            ).unsqueeze(1)
            expected = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=full_mask,
            ).transpose(1, 2)

            with (
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                    return_value=1,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_sequence_parallel_world_size",
                    return_value=1,
                ),
            ):
                attn = USPAttention(
                    num_heads=num_heads,
                    head_size=head_dim,
                    supported_attention_backends={AttentionBackendEnum.TORCH_SDPA},
                )
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    actual = attn(q, k, v, attn_mask=segment_ids > 0)

            valid = segment_ids > 0
            torch.testing.assert_close(actual[valid], expected[valid])
        finally:
            set_global_server_args(prev_args)

    def test_ideogram_preset_guidance_order(self):
        turbo = IDEOGRAM4_PRESETS["V4_TURBO_12"]
        default = IDEOGRAM4_PRESETS["V4_DEFAULT_20"]
        self.assertEqual(turbo["num_steps"], 12)
        self.assertEqual(default["num_steps"], 20)
        self.assertEqual(turbo["guidance_schedule"][0], 3.0)
        self.assertEqual(turbo["guidance_schedule"][-1], 7.0)
        self.assertEqual(tuple(make_step_intervals(2).tolist()), (0.0, 0.5, 1.0))

    def test_ideogram_sampling_params_sync_steps_with_preset(self):
        params = Ideogram4SamplingParams(preset="V4_TURBO_12")
        self.assertEqual(params.num_inference_steps, 12)
        self.assertEqual(params.guidance_scale, 7.0)
        same_steps = Ideogram4SamplingParams(
            preset="V4_TURBO_12", num_inference_steps=12
        )
        self.assertEqual(same_steps.num_inference_steps, 12)
        with self.assertRaisesRegex(ValueError, "derives num_inference_steps"):
            Ideogram4SamplingParams(preset="V4_TURBO_12", num_inference_steps=20)
        same_guidance = Ideogram4SamplingParams(
            preset="V4_TURBO_12", guidance_scale=7.0
        )
        self.assertEqual(same_guidance.guidance_scale, 7.0)
        with self.assertRaisesRegex(ValueError, "guidance_scale cannot be set"):
            Ideogram4SamplingParams(preset="V4_TURBO_12", guidance_scale=6.0)
        with self.assertRaisesRegex(ValueError, "Unknown Ideogram 4 preset"):
            Ideogram4SamplingParams(preset="V4_FAST")

    def test_ideogram_sampling_params_merge_recomputes_preset_fields(self):
        target = Ideogram4SamplingParams()
        user = Ideogram4SamplingParams(
            preset="V4_TURBO_12",
            height=256,
            width=256,
        )

        target._merge_with_user_params(
            user, explicit_fields={"preset", "height", "width"}
        )

        self.assertEqual(target.preset, "V4_TURBO_12")
        self.assertEqual(target.num_inference_steps, 12)
        self.assertEqual(target.guidance_scale, 7.0)
        self.assertEqual(target.height, 256)
        self.assertEqual(target.width, 256)

    def test_unconditional_transformer_uses_denoiser_loader_path(self):
        self.assertIn("unconditional_transformer", TransformerLoader.component_names)
        self.assertEqual(
            get_module_role("unconditional_transformer"), RoleType.DENOISER
        )

        server_args = SimpleNamespace(
            transformer_weights_path="/unused/override.safetensors",
            nunchaku_config={"enabled": True},
            component_transformer_weights_paths={},
        )
        component_args = _server_args_for_transformer_component(
            server_args, "unconditional_transformer"
        )
        self.assertIsNot(component_args, server_args)
        self.assertIsNone(component_args.transformer_weights_path)
        self.assertIsNone(component_args.nunchaku_config)

    def test_transformer_component_uses_per_component_weights_override(self):
        server_args = SimpleNamespace(
            transformer_weights_path=(
                "/ckpt/diffusion_models/ideogram4_nvfp4_mixed.safetensors"
            ),
            nunchaku_config={"enabled": True},
            component_transformer_weights_paths={
                "unconditional_transformer": (
                    "/ckpt/diffusion_models/"
                    "ideogram4_unconditional_nvfp4_mixed.safetensors"
                )
            },
        )

        component_args = _server_args_for_transformer_component(
            server_args,
            "unconditional_transformer",
        )

        self.assertIsNot(component_args, server_args)
        self.assertEqual(
            component_args.transformer_weights_path,
            "/ckpt/diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors",
        )
        self.assertIsNone(component_args.nunchaku_config)

    def test_ideogram_nvfp4_unconditional_transformer_path_uses_sibling_file(self):
        self.assertEqual(
            _resolve_ideogram4_unconditional_transformer_weights_path(
                "/ckpt/diffusion_models/ideogram4_nvfp4_mixed.safetensors"
            ),
            "/ckpt/diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors",
        )
        self.assertIsNone(
            _resolve_ideogram4_unconditional_transformer_weights_path(
                "/ckpt/custom_transformer.safetensors"
            )
        )

    def test_ideogram_denoiser_does_not_request_dtype_cast(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(_fake_server_args())
            transformer = FakeIdeogramTransformer()
            unconditional_transformer = FakeIdeogramTransformer()
            stage = Ideogram4DenoisingStage(
                transformer=transformer,
                unconditional_transformer=unconditional_transformer,
                pipeline=_fake_ideogram_pipeline(
                    transformer, unconditional_transformer
                ),
            )
            uses = stage.component_uses(_fake_server_args(), "stage")
        finally:
            set_global_server_args(prev_args)
        self.assertEqual(
            [use.component_name for use in uses],
            [
                "transformer",
                "unconditional_transformer",
            ],
        )
        self.assertTrue(all(use.target_dtype is None for use in uses))

    def test_ideogram_stages_inherit_common_stage_bases(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(_fake_server_args())
            text_stage = Ideogram4TextEncodingStage(
                text_encoder=None, tokenizer=DummyTokenizer()
            )
            denoising_stage = Ideogram4DenoisingStage(
                transformer=FakeIdeogramTransformer(),
                unconditional_transformer=FakeIdeogramTransformer(),
            )
        finally:
            set_global_server_args(prev_args)
        self.assertIsInstance(text_stage, TextEncodingStage)
        self.assertIsInstance(denoising_stage, DenoisingStage)

    def test_ideogram_text_encoding_dedup_fingerprint_and_extra_copy(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        cfg = Ideogram4PipelineConfig()
        args = _fake_server_args(cfg)
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(args)
            stage = Ideogram4TextEncodingStage(
                text_encoder=None, tokenizer=DummyTokenizer()
            )
        finally:
            set_global_server_args(prev_args)
        base = Req(
            sampling_params=Ideogram4SamplingParams(
                prompt="11 12",
                height=256,
                width=256,
                num_outputs_per_prompt=1,
            )
        )
        same = Req(
            sampling_params=Ideogram4SamplingParams(
                prompt="11 12",
                height=256,
                width=256,
                num_outputs_per_prompt=1,
            )
        )
        different_height = Req(
            sampling_params=Ideogram4SamplingParams(
                prompt="11 12",
                height=512,
                width=256,
                num_outputs_per_prompt=1,
            )
        )
        different_width = Req(
            sampling_params=Ideogram4SamplingParams(
                prompt="11 12",
                height=256,
                width=512,
                num_outputs_per_prompt=1,
            )
        )
        different_outputs = Req(
            sampling_params=Ideogram4SamplingParams(
                prompt="11 12",
                height=256,
                width=256,
                num_outputs_per_prompt=2,
            )
        )

        base_fingerprint = stage.build_dedup_fingerprint(base, args)
        self.assertEqual(base_fingerprint, stage.build_dedup_fingerprint(same, args))
        self.assertNotEqual(
            base_fingerprint, stage.build_dedup_fingerprint(different_height, args)
        )
        self.assertNotEqual(
            base_fingerprint, stage.build_dedup_fingerprint(different_width, args)
        )
        self.assertNotEqual(
            base_fingerprint, stage.build_dedup_fingerprint(different_outputs, args)
        )

        base.prompt_embeds = [torch.tensor([1.0])]
        base.prompt_embeds_mask = [torch.tensor([True])]
        base.extra["ideogram4"] = {
            "position_ids": torch.tensor([[1]]),
            "metadata": {"grid_h": 16},
        }
        stage.copy_deduplicated_outputs(base, same)

        self.assertIn("ideogram4", same.extra)
        self.assertTrue(
            torch.equal(
                same.extra["ideogram4"]["position_ids"],
                base.extra["ideogram4"]["position_ids"],
            )
        )
        self.assertIsNot(
            same.extra["ideogram4"]["position_ids"],
            base.extra["ideogram4"]["position_ids"],
        )

    def test_ideogram_text_encoding_verifies_custom_outputs(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        args = _fake_server_args()
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(args)
            stage = Ideogram4TextEncodingStage(
                text_encoder=None, tokenizer=DummyTokenizer()
            )
        finally:
            set_global_server_args(prev_args)

        batch = Req(
            sampling_params=Ideogram4SamplingParams(
                prompt="11 12",
                height=256,
                width=256,
                num_outputs_per_prompt=1,
            )
        )
        self.assertTrue(stage.verify_input(batch, args).is_valid())

        batch.do_classifier_free_guidance = True
        batch.negative_prompt = []
        batch.negative_prompt_embeds = []
        batch.prompt_embeds = [torch.zeros(1, 4, 8)]
        batch.prompt_embeds_mask = [torch.ones(1, 4, dtype=torch.bool)]
        batch.extra["ideogram4"] = {"position_ids": torch.zeros(1, 4, 3)}

        self.assertTrue(stage.verify_output(batch, args).is_valid())

    def test_ideogram_denoising_component_names_from_pipeline_modules(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        transformer = FakeIdeogramTransformer()
        unconditional_transformer = FakeIdeogramTransformer()
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(_fake_server_args())
            stage = Ideogram4DenoisingStage(
                transformer=transformer,
                unconditional_transformer=unconditional_transformer,
                pipeline=_fake_ideogram_pipeline(
                    transformer, unconditional_transformer
                ),
            )
            uses = stage.component_uses(_fake_server_args(), "stage")
        finally:
            set_global_server_args(prev_args)

        self.assertEqual(
            [use.component_name for use in uses],
            ["transformer", "unconditional_transformer"],
        )

    def test_ideogram_attention_backend_is_passed_from_config(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        config = Ideogram4DiTConfig()
        self.assertEqual(
            config.arch_config._supported_attention_backends,
            {AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA},
        )
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(_fake_server_args())
            with patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ):
                with torch.device("meta"):
                    model = Ideogram4Transformer2DModel(config, {})
        finally:
            set_global_server_args(prev_args)

        self.assertEqual(
            model.supported_attention_backends,
            config.arch_config._supported_attention_backends,
        )
        self.assertEqual(
            model.layers[0].attention.attn.backend,
            AttentionBackendEnum.TORCH_SDPA,
        )

    def test_ideogram_dit_meta_state_dict_matches_checkpoint_shapes(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
            )
            with patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ):
                with torch.device("meta"):
                    model = Ideogram4Transformer2DModel(Ideogram4DiTConfig(), {})
        finally:
            set_global_server_args(prev_args)
        state = model.state_dict()
        self.assertEqual(len(state), 669)
        self.assertEqual(tuple(state["input_proj.weight"].shape), (4608, 128))
        self.assertEqual(tuple(state["input_proj.weight_scale"].shape), (4608,))
        self.assertEqual(
            tuple(state["layers.0.attention.qkv.weight"].shape), (13824, 4608)
        )
        self.assertEqual(state["layers.0.attention.qkv.weight"].dtype, FP8_WEIGHT_DTYPE)

    def test_ideogram_dit_uses_tp_fp8_linears_when_tp_is_initialized(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        fake_tp_group = SimpleNamespace(world_size=2, rank_in_group=1)
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
            )
            with (
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.ideogram.model_parallel_is_initialized",
                    return_value=True,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.ideogram.get_tp_world_size",
                    return_value=2,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                    return_value=fake_tp_group,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8.get_tp_group",
                    return_value=fake_tp_group,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                    return_value=1,
                ),
            ):
                with torch.device("meta"):
                    model = Ideogram4Transformer2DModel(Ideogram4DiTConfig(), {})
        finally:
            set_global_server_args(prev_args)

        self.assertIsInstance(model.input_proj, WeightOnlyFP8ColumnParallelLinear)
        self.assertEqual(tuple(model.input_proj.weight.shape), (2304, 128))
        self.assertEqual(
            tuple(model.layers[0].attention.qkv.weight.shape), (6912, 4608)
        )

    def test_ideogram_dit_nvfp4_quant_config_uses_native_fp4_linears(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "input_proj",
                "llm_cond_proj",
                "t_embedding.*",
                "adaln_proj",
                "layers.*.adaln_modulation",
                "final_layer.*",
            ],
        )
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
            )
            with patch(
                "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                return_value=1,
            ):
                with torch.device("meta"):
                    model = Ideogram4Transformer2DModel(
                        Ideogram4DiTConfig(),
                        {},
                        quant_config=quant_config,
                    )
        finally:
            set_global_server_args(prev_args)

        self.assertEqual(model.layers[0].attention.qkv.prefix, "layers.0.attention.qkv")
        self.assertIsInstance(
            model.layers[0].attention.qkv.quant_method,
            ModelOptFp4LinearMethod,
        )
        self.assertIsInstance(model.input_proj.quant_method, UnquantizedLinearMethod)

        state = model.state_dict()
        self.assertEqual(
            tuple(state["layers.0.attention.qkv.weight"].shape),
            (13824, 2304),
        )
        self.assertEqual(state["layers.0.attention.qkv.weight"].dtype, torch.uint8)
        self.assertEqual(
            tuple(state["layers.0.attention.qkv.weight_scale"].shape),
            (13824, 288),
        )
        self.assertEqual(
            state["layers.0.attention.qkv.weight_scale"].dtype,
            FP8_WEIGHT_DTYPE,
        )
        self.assertEqual(
            tuple(state["layers.0.attention.qkv.weight_scale_2"].shape),
            (1,),
        )
        self.assertEqual(
            tuple(state["layers.0.attention.qkv.input_scale"].shape),
            (1,),
        )

    def test_ideogram_dit_tp_nvfp4_uses_column_parallel_quant_linears(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        fake_tp_group = SimpleNamespace(world_size=2, rank_in_group=1)
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
            )
            with (
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.ideogram.model_parallel_is_initialized",
                    return_value=True,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.models.dits.ideogram.get_tp_world_size",
                    return_value=2,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                    return_value=fake_tp_group,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
                    return_value=1,
                ),
            ):
                with torch.device("meta"):
                    model = Ideogram4Transformer2DModel(
                        Ideogram4DiTConfig(),
                        {},
                        quant_config=quant_config,
                    )
        finally:
            set_global_server_args(prev_args)

        self.assertTrue(model.layers[0].attention.qkv.gather_output)
        self.assertEqual(
            tuple(model.layers[0].attention.qkv.weight.shape), (6912, 2304)
        )
        self.assertIsInstance(
            model.layers[0].attention.qkv.quant_method,
            ModelOptFp4LinearMethod,
        )

    def test_bitsandbytes_tp_quant_state_uses_local_output_shard(self):
        param = torch.nn.Parameter(
            torch.empty(8, 1, dtype=torch.uint8), requires_grad=False
        )
        param.bnb_full_shape = (4, 8)
        param.bnb_local_shape = (2, 8)
        param.bnb_output_shard_start = 2
        param.bnb_input_shard_start = 0
        quant_state = FakeBnbQuantState(
            absmax=torch.arange(8, dtype=torch.float32),
            shape=torch.Size((4, 8)),
            code=torch.ones(16, dtype=torch.float32),
            blocksize=4,
            quant_type="nf4",
            dtype=torch.bfloat16,
        )

        sharded = _maybe_shard_bitsandbytes_4bit_quant_state(param, quant_state)

        self.assertEqual(sharded.shape, torch.Size((2, 8)))
        torch.testing.assert_close(sharded.absmax, torch.tensor([4.0, 5.0, 6.0, 7.0]))

    def test_assign_load_preserves_bitsandbytes_tp_attrs(self):
        class TinyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.empty(8, 1, dtype=torch.uint8), requires_grad=False
                )
                self.weight.bnb_full_shape = (4, 8)
                self.weight.bnb_local_shape = (2, 8)
                self.weight.bnb_output_shard_start = 2
                self.weight.bnb_input_shard_start = 0

        model = TinyModule()
        load_model_from_full_model_state_dict(
            model,
            iter([("weight", torch.ones(8, 1, dtype=torch.uint8))]),
            torch.device("cpu"),
            param_dtype=None,
            strict=True,
            param_names_mapping=lambda name: (name, None, None),
        )

        self.assertEqual(model.weight.bnb_full_shape, (4, 8))
        self.assertEqual(model.weight.bnb_local_shape, (2, 8))
        self.assertEqual(model.weight.bnb_output_shard_start, 2)
        self.assertEqual(model.weight.bnb_input_shard_start, 0)

    def test_missing_weight_only_fp8_scale_is_fatal(self):
        with torch.device("meta"):
            model = WeightOnlyFP8Linear(3, 2, bias=False)
        weights = iter(
            [
                (
                    "weight",
                    torch.tensor(
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        dtype=FP8_WEIGHT_DTYPE,
                    ),
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "Required checkpoint parameter"):
            load_model_from_full_model_state_dict(
                model,
                weights,
                torch.device("cpu"),
                param_dtype=None,
                strict=False,
                param_names_mapping=lambda name: (name, None, None),
            )

    def test_weight_only_fp8_load_accepts_explicit_scale(self):
        with torch.device("meta"):
            model = WeightOnlyFP8Linear(3, 2, bias=False)
        weights = iter(
            [
                (
                    "weight",
                    torch.tensor(
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        dtype=FP8_WEIGHT_DTYPE,
                    ),
                ),
                ("weight_scale", torch.tensor([0.5, 2.0], dtype=torch.float32)),
            ]
        )
        load_model_from_full_model_state_dict(
            model,
            weights,
            torch.device("cpu"),
            param_dtype=None,
            strict=False,
            param_names_mapping=lambda name: (name, None, None),
        )
        self.assertEqual(model.weight.dtype, FP8_WEIGHT_DTYPE)
        self.assertEqual(model.weight_scale.dtype, torch.float32)

    def test_ideogram_text_encoder_post_config_hook_preserves_local_arch(self):
        config = Ideogram4TextEncoderConfig()
        config.arch_config.architectures = ["RemoteQwen3VLTextModel"]
        config.arch_config.ideogram_fp8_weight_only = False
        config.post_diffusers_config_update()
        self.assertEqual(
            config.arch_config.architectures, ["IdeogramQwen3VLTextEncoder"]
        )
        self.assertTrue(config.arch_config.ideogram_fp8_weight_only)
        self.assertFalse(config.arch_config.ideogram_bnb_4bit_weight_only)
        self.assertFalse(config.arch_config.requires_gpu_resident_text_encoder)

    def test_ideogram_text_encoder_post_config_hook_uses_bnb_for_nf4(self):
        config = Ideogram4TextEncoderConfig()
        config.update_model_arch(
            {
                "quantization_config": {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                }
            }
        )

        self.assertEqual(
            config.arch_config.architectures, ["IdeogramQwen3VLTextEncoder"]
        )
        self.assertTrue(config.arch_config.ideogram_bnb_4bit_weight_only)
        self.assertFalse(config.arch_config.ideogram_fp8_weight_only)
        self.assertTrue(config.arch_config.requires_gpu_resident_text_encoder)

    def test_ideogram_text_encoder_swaps_linears_to_weight_only_fp8(self):
        config = Ideogram4TextEncoderConfig()
        config.post_diffusers_config_update()
        config.arch_config.text_config = Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            pad_token_id=0,
        )
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
            )
            with torch.device("meta"):
                encoder = IdeogramQwen3VLTextEncoder(config)
        finally:
            set_global_server_args(prev_args)
        self.assertTrue(
            any(isinstance(module, WeightOnlyFP8Linear) for module in encoder.modules())
        )
        self.assertFalse(
            any(isinstance(module, torch.nn.Linear) for module in encoder.modules())
        )

    def test_ideogram_text_encoder_tp_fp8_uses_column_parallel_linears(self):
        config = Ideogram4TextEncoderConfig()
        config.post_diffusers_config_update()
        config.arch_config.text_config = Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            max_position_embeddings=64,
            pad_token_id=0,
        )
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        fake_tp_group = SimpleNamespace(world_size=2, rank_in_group=1)
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(
                SimpleNamespace(attention_backend="torch_sdpa", comfyui_mode=False)
            )
            with (
                patch(
                    "sglang.multimodal_gen.runtime.models.encoders.qwen3vl.model_parallel_is_initialized",
                    return_value=True,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.models.encoders.qwen3vl.get_tp_world_size",
                    return_value=2,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8.get_tp_group",
                    return_value=fake_tp_group,
                ),
            ):
                with torch.device("meta"):
                    encoder = IdeogramQwen3VLTextEncoder(config)
        finally:
            set_global_server_args(prev_args)

        layer = encoder.language_model.layers[0]
        self.assertEqual(layer.self_attn.num_heads, 2)
        self.assertEqual(layer.self_attn.num_key_value_heads, 2)
        self.assertIsInstance(layer.self_attn.q_proj, WeightOnlyFP8ColumnParallelLinear)
        self.assertFalse(layer.self_attn.q_proj.gather_output)
        self.assertTrue(layer.self_attn.o_proj.gather_output)
        self.assertIsInstance(layer.mlp.gate_proj, WeightOnlyFP8ColumnParallelLinear)
        self.assertFalse(layer.mlp.gate_proj.gather_output)
        self.assertTrue(layer.mlp.down_proj.gather_output)

    def test_denoise_and_decode_shape_smoke(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        cfg = Ideogram4PipelineConfig()
        args = _fake_server_args(cfg)
        device = get_local_torch_device()
        prev_args = server_args_module._global_server_args
        try:
            set_global_server_args(args)
            transformer = FakeIdeogramTransformer()
            unconditional_transformer = FakeIdeogramTransformer()
            denoise_stage = Ideogram4DenoisingStage(
                transformer=transformer,
                unconditional_transformer=unconditional_transformer,
                pipeline=_fake_ideogram_pipeline(
                    transformer, unconditional_transformer
                ),
            )
            decode_stage = Ideogram4DecodingStage(vae=FakeIdeogramVAE())
            batch = Req(
                sampling_params=Ideogram4SamplingParams(
                    prompt="11 12",
                    height=256,
                    width=256,
                    preset="V4_TURBO_12",
                    suppress_logs=True,
                )
            )
            batch.latents = torch.zeros(1, 1, 128, device=device)
            batch.raw_latent_shape = batch.latents.shape
            batch.prompt_embeds = [torch.zeros(1, 2, 8, device=device)]
            batch.extra["ideogram4"] = {
                "max_text_tokens": 1,
                "num_image_tokens": 1,
                "position_ids": torch.zeros(1, 2, 3, dtype=torch.long, device=device),
                "segment_ids": torch.ones(1, 2, dtype=torch.long, device=device),
                "indicator": torch.tensor(
                    [[LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR]],
                    dtype=torch.long,
                    device=device,
                ),
                "grid_h": 1,
                "grid_w": 1,
            }
            denoised = denoise_stage.forward(batch, args)
            self.assertEqual(tuple(denoised.latents.shape), (1, 1, 128))
            decoded = decode_stage.forward(denoised, args)
        finally:
            set_global_server_args(prev_args)

        self.assertEqual(tuple(decoded.output.shape), (1, 3, 2, 2))

    def test_text_input_builder_matches_official_layout(self):
        prev_args = None
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            cfg = Ideogram4PipelineConfig()
            args = SimpleNamespace(pipeline_config=cfg, comfyui_mode=False)
            set_global_server_args(args)
            stage = Ideogram4TextEncodingStage(
                text_encoder=None, tokenizer=DummyTokenizer()
            )
            inputs = stage._build_inputs(["11 12 13", "21"], 256, 256, args)
        finally:
            set_global_server_args(prev_args)

        self.assertEqual(inputs["grid_h"], 16)
        self.assertEqual(inputs["grid_w"], 16)
        self.assertEqual(inputs["num_image_tokens"], 256)
        self.assertEqual(inputs["max_text_tokens"], 3)
        self.assertEqual(inputs["token_ids"][0, :3].tolist(), [11, 12, 13])
        self.assertEqual(inputs["token_ids"][1, :2].tolist(), [0, 0])
        self.assertTrue(
            torch.all(inputs["indicator"][0, :3] == LLM_TOKEN_INDICATOR).item()
        )
        self.assertTrue(
            torch.all(inputs["indicator"][0, 3:] == OUTPUT_IMAGE_INDICATOR).item()
        )
        self.assertEqual(inputs["position_ids"][0, 3, 0].item(), IMAGE_POSITION_OFFSET)

    def test_text_input_builder_rejects_unsupported_resolution(self):
        import sglang.multimodal_gen.runtime.server_args as server_args_module

        prev_args = server_args_module._global_server_args
        try:
            cfg = Ideogram4PipelineConfig()
            args = SimpleNamespace(pipeline_config=cfg, comfyui_mode=False)
            set_global_server_args(args)
            stage = Ideogram4TextEncodingStage(
                text_encoder=None, tokenizer=DummyTokenizer()
            )
            with self.assertRaisesRegex(ValueError, "between 256 and 2048"):
                stage._build_inputs(["11"], 128, 256, args)
        finally:
            set_global_server_args(prev_args)


if __name__ == "__main__":
    unittest.main()
