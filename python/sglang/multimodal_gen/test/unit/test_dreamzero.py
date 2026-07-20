import types

import pytest
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
from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    BRANCH_COND,
    BRANCH_UNCOND,
    DreamZeroCachePoolManager,
    apply_request_lifecycle_resets,
    normalize_batched_session_fields,
    resolve_request_cache,
)
from sglang.multimodal_gen.runtime.models.dits.dreamzero_causal import (
    DreamZeroCausalWanModel,
)
from sglang.multimodal_gen.runtime.pipelines.dreamzero_pipeline import DreamZeroPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero.denoising import (
    DreamZeroCausalDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero.text_encoding import (
    DreamZeroTextEncodingStage,
)


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        del attention_mask
        self.calls += 1
        return types.SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float())


def _make_text_server_args(*, enable_cfg_parallel: bool = False) -> types.SimpleNamespace:
    server_args = types.SimpleNamespace(
        enable_cfg_parallel=enable_cfg_parallel,
        pipeline_config=DreamZeroPipelineConfig(),
    )
    server_args.pipeline_config.text_encoder_precisions = ("bf16",)
    return server_args


def _make_text_stage(
    *,
    manager: DreamZeroCachePoolManager,
    encoder: _FakeTextEncoder | None = None,
    enable_cfg_parallel: bool = False,
) -> tuple[DreamZeroTextEncodingStage, types.SimpleNamespace, _FakeTextEncoder]:
    if encoder is None:
        encoder = _FakeTextEncoder()
    stage = DreamZeroTextEncodingStage(text_encoder=encoder, cache_manager=manager)
    server_args = _make_text_server_args(enable_cfg_parallel=enable_cfg_parallel)
    stage.server_args = server_args
    return stage, server_args, encoder


def _tokenized_text_batch(
    *,
    session_id: str,
    prompt: str,
    negative_prompt: str | None = None,
    reset: bool = False,
    text_token: int = 1,
) -> types.SimpleNamespace:
    inputs = {
        "text": torch.full((1, 2), text_token, dtype=torch.long),
        "text_attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    if negative_prompt is not None:
        inputs["text_negative"] = torch.zeros(1, 2, dtype=torch.long)
    return types.SimpleNamespace(
        extra={
            "dreamzero_session_ids": [session_id],
            "dreamzero_reset_mask": [reset],
            "dreamzero_prompts": [prompt],
            "dreamzero_negative_prompts": [negative_prompt],
        },
        dreamzero_inputs=inputs,
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


def test_dreamzero_dit_rope_lengths_are_configurable():
    model = DreamZeroCausalWanModel(
        model_type="i2v",
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=0,
        frame_seqlen=8,
        text_dim=32,
        hidden_size=16,
        rope_video_max_positions=(7, 8, 9),
        rope_action_max_positions=10,
        rope_state_max_positions=11,
    )

    assert model.freqs[0].shape[0] == 7
    assert model.freqs[1].shape[0] == 8
    assert model.freqs[2].shape[0] == 9
    assert model.freqs_action.shape[0] == 10
    assert model.freqs_state.shape[0] == 11


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
    assert params.num_inference_steps == 99
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

    with pytest.raises(TypeError, match="dreamzero_session_ids must be a list"):
        normalize_batched_session_fields(
            session_ids="session-a",
            reset_mask=True,
            batch_size=1,
        )


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


def test_dreamzero_single_prompt_embedding_does_not_duplicate_cfg_branch():
    stage = object.__new__(DreamZeroCausalDenoisingStage)
    prompt_emb = torch.ones(1, 2, 3)
    batch = types.SimpleNamespace(dreamzero_prompt_embs=[prompt_emb])
    server_args = types.SimpleNamespace(enable_cfg_parallel=False)

    branch_ctx = stage._prepare_cfg_branches(
        batch,
        server_args,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert branch_ctx.local_branch_indices == [0]
    assert len(branch_ctx.local_prompt_embs) == 1
    assert torch.equal(branch_ctx.local_prompt_embs[0], prompt_emb)


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


def test_dreamzero_text_stage_does_not_gather_non_reusable_new_slots():
    manager = DreamZeroCachePoolManager(max_sessions=2)
    stage, server_args, _ = _make_text_stage(manager=manager)

    first_batch = _tokenized_text_batch(
        session_id="session-a",
        prompt="pick",
    )
    stage.forward(first_batch, server_args)
    assert manager.pool.cached_prompt_embs[BRANCH_COND].shape[0] == 1

    second_batch = _tokenized_text_batch(
        session_id="session-b",
        prompt="place",
        text_token=2,
    )
    stage.forward(second_batch, server_args)

    assert second_batch.dreamzero_prompt_embs[0].shape[0] == 1
    assert manager.pool.cached_prompt_embs[BRANCH_COND].shape[0] == 2


def test_dreamzero_cfg_text_stage_keeps_encoder_collectives_aligned():
    manager = DreamZeroCachePoolManager(max_sessions=1)
    stage, server_args, encoder = _make_text_stage(
        manager=manager,
        enable_cfg_parallel=True,
    )

    cached_neg_batch = _tokenized_text_batch(
        session_id="session-a",
        prompt="old prompt",
        negative_prompt="",
    )
    cached_neg = resolve_request_cache(
        cached_neg_batch,
        manager,
        local_attn_size=4,
        batch_size=1,
    )
    stage._forward_cache_manager(
        cached_neg_batch,
        server_args,
        cached_neg,
        cfg_parallel=True,
        cfg_rank=1,
    )
    assert manager.pool.prompt_hashes[BRANCH_COND][0] == cached_neg.prompt_hashes[0]

    changed_cond_batch = _tokenized_text_batch(
        session_id="session-a",
        prompt="new prompt",
        negative_prompt="",
    )
    changed_cond = resolve_request_cache(
        changed_cond_batch,
        manager,
        local_attn_size=4,
        batch_size=1,
    )
    assert changed_cond.prompt_reusable == [False]
    assert changed_cond.neg_prompt_reusable == [True]

    stage._forward_cache_manager(
        changed_cond_batch,
        server_args,
        changed_cond,
        cfg_parallel=True,
        cfg_rank=1,
    )

    assert changed_cond_batch.dreamzero_lifecycle_reset_mask == [True]
    assert encoder.calls == 2

    stable_cond_batch = _tokenized_text_batch(
        session_id="session-a",
        prompt="new prompt",
        negative_prompt="",
    )
    stable_cond = resolve_request_cache(
        stable_cond_batch,
        manager,
        local_attn_size=4,
        batch_size=1,
    )
    assert stable_cond.prompt_reusable == [False]
    assert stable_cond.neg_prompt_reusable == [True]

    stage._forward_cache_manager(
        stable_cond_batch,
        server_args,
        stable_cond,
        cfg_parallel=True,
        cfg_rank=1,
    )

    assert encoder.calls == 2


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
