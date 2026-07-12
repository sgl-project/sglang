import types

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
    DreamZeroPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.dreamzero import DreamZeroSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
)
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.entrypoints.vla.protocol import (
    action_generation_response,
    action_metadata,
    build_action_sampling_params,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_checkpoint_utils import (
    DreamZeroCheckpointLoadReport,
    load_matching_tensors,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_encoder_loader import (
    expected_dreamzero_text_state_keys,
    remap_dreamzero_image_model_key,
    remap_dreamzero_text_model_key,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_vae_loader import (
    remap_dreamzero_vae_model_key,
)
from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    BRANCH_COND,
    BRANCH_UNCOND,
    DreamZeroCachePoolManager,
    apply_request_lifecycle_resets,
    normalize_batched_session_fields,
    resolve_request_cache,
)
from sglang.multimodal_gen.runtime.pipelines.dreamzero_pipeline import DreamZeroPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero.denoising import (
    DreamZeroCausalDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero.text_encoding import (
    DreamZeroTextEncodingStage,
)


def test_dreamzero_registry_detects_non_diffusers_model():
    model_info = get_model_info("nvidia/DreamZero-DROID", backend="sglang")

    assert get_non_diffusers_pipeline_name("nvidia/DreamZero-DROID") == (
        "DreamZeroPipeline"
    )
    assert model_info is not None
    assert model_info.pipeline_cls is DreamZeroPipeline
    assert model_info.sampling_param_cls is DreamZeroSamplingParams
    assert model_info.pipeline_config_cls is DreamZeroPipelineConfig


def test_dreamzero_config_and_sampling_defaults_are_action_typed():
    config = DreamZeroPipelineConfig()
    params = DreamZeroSamplingParams()

    assert config.task_type is ModelTaskType.VLA_ACTION
    assert config.task_type.data_type() is DataType.ACTION
    assert params.data_type is DataType.ACTION
    extra = params.build_request_extra()
    assert "dreamzero_action_horizon" not in extra
    assert "dreamzero_relative_action_per_horizon" not in extra
    assert "dreamzero_embodiment_tag" not in extra


def test_dreamzero_action_response_uses_common_action_contract():
    server_args = types.SimpleNamespace(
        model_id=None,
        model_path="dreamzero-test",
        pipeline_config=DreamZeroPipelineConfig(),
    )
    output = {
        "actions": torch.zeros(1, 24, 7).numpy(),
        "parameters": {"num_inference_steps": 16},
    }

    response = action_generation_response(output, server_args)

    assert response["object"] == "action.generation"
    assert response["data"][0]["action"]["shape"] == [1, 24, 7]
    assert response["usage"]["denoise_steps"] == 16

    response_without_parameters = action_generation_response(
        {"actions": torch.zeros(1, 24, 7).numpy()},
        server_args,
    )

    assert response_without_parameters["usage"]["denoise_steps"] == 16


def test_dreamzero_action_request_builder_uses_fixed_sampling_params():
    server_args = types.SimpleNamespace(
        model_path="nvidia/DreamZero-DROID",
        model_id=None,
        backend="sglang",
        pipeline_class_name=None,
        output_path=None,
        comfyui_mode=False,
        pipeline_config=DreamZeroPipelineConfig(),
    )
    payload = {
        "id": "req-1",
        "input": {
            "prompt": ["pick the cube"],
            "observation": {
                "state": {"values": [1.0, 2.0], "dtype": "float32", "shape": [2]},
            },
        },
        "parameters": {
            "session_ids": ["session-a"],
            "reset_mask": [True],
            "negative_prompts": [""],
            "embodiment_tag": "libero_sim",
            "action_horizon": 24,
            "relative_action_per_horizon": False,
            "guidance_scale": 9.0,
            "seed": 7,
            "num_inference_steps": 99,
        },
    }

    params = build_action_sampling_params(payload, server_args)
    extra = params.build_request_extra()

    assert params.request_id == "req-1"
    assert params.prompt == "pick the cube"
    assert params.num_inference_steps == 4
    assert params.guidance_scale == DreamZeroSamplingParams().guidance_scale
    assert extra["dreamzero_session_ids"] == ["session-a"]
    assert extra["dreamzero_reset_mask"] == [True]
    assert extra["dreamzero_prompts"] == ["pick the cube"]
    assert extra["dreamzero_normalized_input"]["state"].shape == (2,)
    assert "dreamzero_action_horizon" not in extra
    assert "dreamzero_relative_action_per_horizon" not in extra
    assert "dreamzero_embodiment_tag" not in extra


def test_dreamzero_action_metadata_is_wam_specific():
    server_args = types.SimpleNamespace(
        model_id=None,
        model_path="dreamzero-test",
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        pipeline_config=DreamZeroPipelineConfig(),
    )

    metadata = action_metadata(server_args)

    assert metadata["object"] == "action.metadata"
    assert metadata["policy_family"] == "dreamzero"
    assert metadata["defaults"]["num_inference_steps"] == 16


def test_vla_action_additions_do_not_change_visual_defaults():
    assert ModelTaskType.T2V.data_type() is DataType.VIDEO
    assert ModelTaskType.I2V.data_type() is DataType.VIDEO
    assert ModelTaskType.T2I.data_type() is DataType.IMAGE
    assert ModelTaskType.I2M.data_type() is DataType.MESH
    assert ModelTaskType.T2V.requires_image_input() is False
    assert ModelTaskType.I2V.requires_image_input() is True
    assert ModelTaskType.VLA_ACTION.requires_image_input() is False
    assert ModelTaskType.VLA_ACTION.accepts_image_input() is True


def test_dreamzero_single_request_session_fields_still_use_batched_contract():
    session_ids, reset_mask = normalize_batched_session_fields(
        session_ids=["session-a"],
        reset_mask=[True],
        batch_size=1,
    )

    assert session_ids == ["session-a"]
    assert reset_mask == [True]

    try:
        normalize_batched_session_fields(
            session_ids="session-a",
            reset_mask=True,
            batch_size=1,
        )
    except TypeError as exc:
        assert "dreamzero_session_ids must be a list" in str(exc)
    else:
        raise AssertionError("DreamZero session ids must use explicit batched lists")


def test_dreamzero_denoising_skip_schedule_reuses_previous_predictions():
    skip_state = {"countdown": 0}
    predictions = [
        (
            torch.zeros(1, 1),
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(1, 1),
        ),
        (
            torch.zeros(1, 1),
            torch.tensor([[0.99, 0.01]]),
            torch.zeros(1, 1),
        ),
    ]

    should_run = DreamZeroCausalDenoisingStage._should_run_model(
        step_index=2,
        current_timestep=torch.tensor(2),
        prev_predictions=predictions,
        dit_step_mask=None,
        dynamic_cache_schedule=True,
        skip_state=skip_state,
    )

    assert should_run is False
    assert skip_state["countdown"] == 4
    assert (
        DreamZeroCausalDenoisingStage._should_run_model(
            step_index=3,
            current_timestep=torch.tensor(1),
            prev_predictions=predictions,
            dit_step_mask=None,
            dynamic_cache_schedule=True,
            skip_state=skip_state,
        )
        is False
    )
    assert skip_state["countdown"] == 3


def test_dreamzero_scheduler_step_supports_optional_step_index():
    class SchedulerWithStepIndex:
        def step(self, *, model_output, timestep, sample, step_index, return_dict):
            assert step_index == 7
            assert return_dict is False
            del timestep
            return (sample - model_output,)

    class SchedulerWithoutStepIndex:
        def step(self, *, model_output, timestep, sample, return_dict):
            assert return_dict is False
            del timestep
            return (sample + model_output,)

    sample = torch.tensor([3.0])
    model_output = torch.tensor([1.0])
    timestep = torch.tensor(4)

    assert torch.equal(
        DreamZeroCausalDenoisingStage._scheduler_step(
            SchedulerWithStepIndex(),
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            step_index=7,
        ),
        torch.tensor([2.0]),
    )
    assert torch.equal(
        DreamZeroCausalDenoisingStage._scheduler_step(
            SchedulerWithoutStepIndex(),
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            step_index=7,
        ),
        torch.tensor([4.0]),
    )


def test_dreamzero_session_cache_allocates_reuses_and_resets_slots():
    manager = DreamZeroCachePoolManager(max_sessions=2)
    batch = types.SimpleNamespace(
        extra={
            "dreamzero_session_ids": ["session-a", "session-b"],
            "dreamzero_reset_mask": [False, False],
            "dreamzero_prompts": ["pick", "place"],
            "dreamzero_negative_prompts": ["", ""],
        },
        dreamzero_inputs={},
    )

    request_cache = resolve_request_cache(
        batch,
        manager,
        local_attn_size=4,
        batch_size=2,
    )

    assert request_cache.slot_indices == [0, 1]
    assert request_cache.cache_hit == [False, False]

    prompt_values = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    manager.pool.scatter_prompt(
        BRANCH_COND,
        request_cache.slot_indices,
        prompt_values,
        request_cache.prompt_hashes,
    )
    manager.pool.scatter_prompt(
        BRANCH_UNCOND,
        request_cache.slot_indices,
        prompt_values + 100,
        request_cache.neg_prompt_hashes,
    )
    manager.pool.scatter_visual(
        request_cache.slot_indices,
        clip_feas=torch.ones(2, 1),
        ys=torch.ones(2, 1, 1),
        latent_video=torch.ones(2, 1, 1, 1, 1),
    )

    second_batch = types.SimpleNamespace(
        extra=batch.extra,
        dreamzero_inputs={},
    )
    second_cache = resolve_request_cache(
        second_batch,
        manager,
        local_attn_size=4,
        batch_size=2,
    )

    assert second_cache.slot_indices == [0, 1]
    assert second_cache.cache_hit == [True, True]
    assert second_cache.prompt_reusable == [True, True]
    assert second_cache.neg_prompt_reusable == [True, True]

    reset_batch = types.SimpleNamespace(
        extra={
            "dreamzero_session_ids": ["session-a", "session-b"],
            "dreamzero_reset_mask": [True, False],
            "dreamzero_prompts": ["pick", "place"],
            "dreamzero_negative_prompts": ["", ""],
        },
        dreamzero_inputs={},
    )
    reset_cache = resolve_request_cache(
        reset_batch,
        manager,
        local_attn_size=4,
        batch_size=2,
    )
    apply_request_lifecycle_resets(reset_batch, manager, reset_cache)

    assert reset_cache.cache_hit == [False, True]
    assert manager.pool.prompt_valid[BRANCH_COND][0] is False
    assert manager.pool.prompt_valid[BRANCH_COND][1] is True
    assert manager.pool.visual_valid == [False, True]


def test_dreamzero_tensor_prompt_cache_requires_explicit_hash():
    manager = DreamZeroCachePoolManager(max_sessions=1)
    batch = types.SimpleNamespace(
        extra={
            "dreamzero_session_ids": ["session-a"],
            "dreamzero_reset_mask": [False],
            "dreamzero_prompts": [None],
            "dreamzero_negative_prompts": [None],
        },
        dreamzero_inputs={
            "text": torch.ones(1, 2, 3),
            "text_negative": torch.ones(1, 2, 3),
        },
    )

    request_cache = resolve_request_cache(
        batch,
        manager,
        local_attn_size=4,
        batch_size=1,
    )
    manager.pool.scatter_prompt(
        BRANCH_COND,
        request_cache.slot_indices,
        torch.ones(1, 2, 3),
        request_cache.prompt_hashes,
    )
    second_cache = resolve_request_cache(
        batch,
        manager,
        local_attn_size=4,
        batch_size=1,
    )

    assert request_cache.prompt_hashes == [None]
    assert second_cache.prompt_reusable == [False]

    keyed_batch = types.SimpleNamespace(
        extra={
            **batch.extra,
            "dreamzero_prompt_hashes": ["prompt-key"],
        },
        dreamzero_inputs=batch.dreamzero_inputs,
    )
    keyed_cache = resolve_request_cache(
        keyed_batch,
        manager,
        local_attn_size=4,
        batch_size=1,
    )
    manager.pool.scatter_prompt(
        BRANCH_COND,
        keyed_cache.slot_indices,
        torch.ones(1, 2, 3),
        keyed_cache.prompt_hashes,
    )
    reused_cache = resolve_request_cache(
        keyed_batch,
        manager,
        local_attn_size=4,
        batch_size=1,
    )

    assert keyed_cache.prompt_hashes == ["key:prompt-key"]
    assert reused_cache.prompt_reusable == [True]


def test_dreamzero_text_encoding_masks_padding_without_python_seq_len_sync():
    class FakeEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            values = torch.arange(
                input_ids.shape[0] * input_ids.shape[1] * 2,
                dtype=torch.float32,
                device=input_ids.device,
            )
            return types.SimpleNamespace(
                last_hidden_state=values.reshape(
                    input_ids.shape[0], input_ids.shape[1], 2
                )
            )

    stage = DreamZeroTextEncodingStage()
    output = stage._encode_prompt(
        FakeEncoder(),
        input_ids=torch.ones(2, 4, dtype=torch.long),
        attention_mask=torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 0]],
            dtype=torch.long,
        ),
        text_len=4,
    )

    assert output.dtype == torch.bfloat16
    assert torch.equal(output[0, 2:], torch.zeros_like(output[0, 2:]))
    assert torch.equal(output[1, 3:], torch.zeros_like(output[1, 3:]))
    assert torch.equal(
        output[1, :3],
        torch.tensor([[8, 9], [10, 11], [12, 13]], dtype=torch.bfloat16),
    )


def test_dreamzero_checkpoint_helper_loads_parameters_buffers_and_reports_errors():
    class SmallModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.mismatch = torch.nn.Parameter(torch.zeros(2))
            self.log_scale = torch.nn.Parameter(torch.zeros(()))
            self.register_buffer("scale", torch.zeros(2))

    model = SmallModule()
    report = load_matching_tensors(
        model,
        [
            ("prefix.weight", torch.ones(2, 2)),
            ("prefix.scale", torch.ones(2) * 2),
            ("prefix.log_scale", torch.ones(1) * 3),
            ("prefix.mismatch", torch.ones(3)),
            ("prefix.unused", torch.ones(1)),
        ],
        device=torch.device("cpu"),
        key_mapper=lambda key: key.removeprefix("prefix."),
        report_cls=DreamZeroCheckpointLoadReport,
    )

    assert torch.equal(model.weight, torch.ones(2, 2))
    assert torch.equal(model.scale, torch.ones(2) * 2)
    assert torch.equal(model.log_scale, torch.tensor(3.0))
    assert report.loaded_keys == ["weight", "scale", "log_scale"]
    assert report.unexpected_keys == ["unused"]
    assert report.shape_mismatches == {"mismatch": ((2,), (3,))}


def test_dreamzero_vae_loader_remaps_original_wan_keys_to_sglang_wan_vae():
    cases = {
        "encoder.conv1.weight": "encoder.conv_in.weight",
        "encoder.downsamples.3.residual.2.weight": (
            "encoder.down_blocks.3.conv1.weight"
        ),
        "encoder.downsamples.3.shortcut.bias": (
            "encoder.down_blocks.3.conv_shortcut.bias"
        ),
        "encoder.middle.1.to_qkv.weight": (
            "encoder.mid_block.attentions.0.to_qkv.weight"
        ),
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "conv1.weight": "quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
        "decoder.conv1.weight": "decoder.conv_in.weight",
        "decoder.upsamples.7.time_conv.weight": (
            "decoder.up_blocks.1.upsamplers.0.time_conv.weight"
        ),
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
    }

    for old_key, new_key in cases.items():
        assert remap_dreamzero_vae_model_key(old_key) == new_key


def test_dreamzero_vae_loader_remaps_original_wan38_keys_to_sglang_wan_vae():
    cases = {
        "encoder.downsamples.1.downsamples.0.shortcut.weight": (
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight"
        ),
        "encoder.downsamples.2.downsamples.2.time_conv.bias": (
            "encoder.down_blocks.2.downsampler.time_conv.bias"
        ),
        "decoder.upsamples.0.upsamples.3.time_conv.weight": (
            "decoder.up_blocks.0.upsampler.time_conv.weight"
        ),
        "decoder.upsamples.3.upsamples.0.shortcut.bias": (
            "decoder.up_blocks.3.resnets.0.conv_shortcut.bias"
        ),
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    for old_key, new_key in cases.items():
        assert remap_dreamzero_vae_model_key(old_key) == new_key


def test_dreamzero_text_encoder_key_remap():
    cases = {
        "token_embedding.weight": "shared.weight",
        "norm.weight": "encoder.final_layer_norm.weight",
        "blocks.3.norm1.weight": "encoder.block.3.layer.0.layer_norm.weight",
        "blocks.3.attn.q.weight": ("encoder.block.3.layer.0.SelfAttention.q.weight"),
        "blocks.3.attn.k.weight": ("encoder.block.3.layer.0.SelfAttention.k.weight"),
        "blocks.3.attn.v.weight": ("encoder.block.3.layer.0.SelfAttention.v.weight"),
        "blocks.3.attn.o.weight": ("encoder.block.3.layer.0.SelfAttention.o.weight"),
        "blocks.3.pos_embedding.embedding.weight": (
            "encoder.block.3.layer.0.SelfAttention.relative_attention_bias.weight"
        ),
        "blocks.3.norm2.weight": "encoder.block.3.layer.1.layer_norm.weight",
        "blocks.3.ffn.gate.0.weight": (
            "encoder.block.3.layer.1.DenseReluDense.wi_0.weight"
        ),
        "blocks.3.ffn.fc1.weight": (
            "encoder.block.3.layer.1.DenseReluDense.wi_1.weight"
        ),
        "blocks.3.ffn.fc2.weight": ("encoder.block.3.layer.1.DenseReluDense.wo.weight"),
    }
    for old_key, new_key in cases.items():
        assert remap_dreamzero_text_model_key(old_key) == new_key


def test_dreamzero_image_encoder_key_remap():
    cases = {
        "model.visual.cls_embedding": "vision_model.embeddings.class_embedding",
        "model.visual.patch_embedding.weight": (
            "vision_model.embeddings.patch_embedding.weight"
        ),
        "model.visual.pos_embedding": (
            "vision_model.embeddings.position_embedding.weight"
        ),
        "model.visual.pre_norm.weight": "vision_model.pre_layrnorm.weight",
        "model.visual.transformer.3.norm1.bias": (
            "vision_model.encoder.layers.3.layer_norm1.bias"
        ),
        "model.visual.transformer.3.attn.to_qkv.weight": (
            "vision_model.encoder.layers.3.self_attn.qkv_proj.weight"
        ),
        "model.visual.transformer.3.attn.proj.bias": (
            "vision_model.encoder.layers.3.self_attn.out_proj.bias"
        ),
        "model.visual.transformer.3.mlp.0.weight": (
            "vision_model.encoder.layers.3.mlp.fc1.weight"
        ),
        "model.visual.transformer.3.mlp.2.bias": (
            "vision_model.encoder.layers.3.mlp.fc2.bias"
        ),
    }
    for old_key, new_key in cases.items():
        assert remap_dreamzero_image_model_key(old_key) == new_key

    assert remap_dreamzero_image_model_key("model.log_scale") is None
    assert remap_dreamzero_image_model_key("model.visual.head") is None
    assert remap_dreamzero_image_model_key("model.visual.post_norm.weight") is None
    assert (
        remap_dreamzero_image_model_key(
            "model.visual.transformer.31.attn.to_qkv.weight"
        )
        is None
    )


def test_dreamzero_text_expected_keys_ignore_tied_embedding_alias():
    state_keys = {
        "shared.weight",
        "encoder.embed_tokens.weight",
        "encoder.final_layer_norm.weight",
    }

    expected = expected_dreamzero_text_state_keys(
        state_keys, {"shared.weight", "encoder.final_layer_norm.weight"}
    )

    assert expected == {"shared.weight", "encoder.final_layer_norm.weight"}
