import types

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.dreamzero_causal import (
    DreamZeroCausalWanArchConfig,
    DreamZeroCausalWanConfig,
)
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
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.models.dits import dreamzero_causal as dreamzero_dit
from sglang.multimodal_gen.runtime.models.dits.dreamzero_causal import (
    DreamZeroCausalWanModel,
    DreamZeroCausalWanSelfAttention,
)
from sglang.multimodal_gen.runtime.pipelines.dreamzero_pipeline import (
    DreamZeroPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.denoising import (
    DreamZeroActionOutputStage,
    DreamZeroCausalDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.image_encoding import (
    DreamZeroVisualEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.session_cache import (
    BRANCH_COND,
    BRANCH_UNCOND,
    DreamZeroCachePoolManager,
    apply_request_lifecycle_resets,
    normalize_batched_session_fields,
    resolve_request_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.text_encoding import (
    DreamZeroTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        del output_hidden_states
        del attention_mask
        self.calls += 1
        return types.SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float())


class _FakeTokenizedText(dict):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)

    def to(self, device):
        return _FakeTokenizedText(
            self["input_ids"].to(device),
            self["attention_mask"].to(device),
        )


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, **kwargs):
        texts = list(texts)
        self.calls.append(texts)
        max_length = int(kwargs.get("max_length", 4))
        input_ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            length = min(max(len(str(text).split()), 1), max_length)
            token = (sum(ord(ch) for ch in str(text)) % 97) + 1
            input_ids[row, :length] = token
            attention_mask[row, :length] = 1
        return _FakeTokenizedText(input_ids, attention_mask)


def _make_text_server_args(
    *, enable_cfg_parallel: bool = False
) -> types.SimpleNamespace:
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
    stage = DreamZeroTextEncodingStage(
        text_encoder=encoder,
        tokenizer=_FakeTokenizer(),
        cache_manager=manager,
    )
    server_args = _make_text_server_args(enable_cfg_parallel=enable_cfg_parallel)
    stage.server_args = server_args
    return stage, server_args, encoder


def _patch_uninitialized_usp_ring_group(monkeypatch) -> None:
    monkeypatch.setattr(attention_layer, "get_ring_parallel_world_size", lambda: 1)


def _prompt_text_batch(
    *,
    session_id: str,
    prompt: str,
    negative_prompt: str | None = None,
    reset: bool = False,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        prompt=prompt,
        negative_prompt="" if negative_prompt is None else negative_prompt,
        extra={
            "dreamzero_session_ids": [session_id],
            "dreamzero_reset_mask": [reset],
            "dreamzero_prompts": [prompt],
            "dreamzero_negative_prompts": [
                "" if negative_prompt is None else negative_prompt
            ],
        },
        dreamzero_inputs={"state": torch.zeros(1, 1)},
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
    assert "tokenizer" in DreamZeroPipeline._required_config_modules
    assert "image_encoder" in DreamZeroPipeline._required_config_modules
    assert config.image_encoder_config.prefix == "dreamzero_image_encoder"
    assert config.image_encoder_config.num_hidden_layers_override == 31
    assert config.action_horizon == config.dit_config.arch_config.num_action_per_block
    assert config.action_dim == config.dit_config.arch_config.action_dim
    assert config.output_action_dim == config.dit_config.arch_config.action_dim
    assert params.data_type is DataType.ACTION
    assert params.num_inference_steps == config.default_num_inference_steps
    assert not hasattr(params, "guidance_scale")
    extra = params.build_request_extra()
    assert "dreamzero_action_horizon" not in extra
    assert "dreamzero_relative_action_per_horizon" not in extra
    assert "dreamzero_embodiment_tag" not in extra


def test_dreamzero_dit_rope_lengths_are_configurable():
    config = DreamZeroCausalWanConfig(
        arch_config=DreamZeroCausalWanArchConfig(
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
    )
    model = DreamZeroCausalWanModel(
        config=config,
    )

    assert model.rope_video_max_positions == (7, 8, 9)
    assert model.rope_action_max_positions == 10
    assert model.rope_state_max_positions == 11
    assert model.rotary_emb.rope_dim_list == [8, 4, 4]
    assert model.action_rotary_emb.rope_dim_list == [16]
    assert model.state_rotary_emb.rope_dim_list == [16]


def test_dreamzero_dit_keeps_cross_attention_norm_for_native_loading(monkeypatch):
    _patch_uninitialized_usp_ring_group(monkeypatch)
    config = DreamZeroCausalWanConfig(
        arch_config=DreamZeroCausalWanArchConfig(
            model_type="ti2v",
            dim=64,
            ffn_dim=128,
            num_heads=4,
            num_layers=1,
            frame_seqlen=8,
            text_dim=32,
            hidden_size=16,
        )
    )
    model = DreamZeroCausalWanModel(config=config)

    assert not isinstance(model.blocks[0].norm3, torch.nn.Identity)


def test_dreamzero_dit_uses_native_usp_attention_modules(monkeypatch):
    _patch_uninitialized_usp_ring_group(monkeypatch)
    config = DreamZeroCausalWanConfig(
        arch_config=DreamZeroCausalWanArchConfig(
            model_type="ti2v",
            dim=64,
            ffn_dim=128,
            num_heads=4,
            num_layers=1,
            frame_seqlen=8,
            text_dim=32,
            hidden_size=16,
        )
    )
    model = DreamZeroCausalWanModel(config=config)
    block = model.blocks[0]

    assert isinstance(block.self_attn.attn, USPAttention)
    assert isinstance(block.self_attn.sequence_parallel_attn, USPAttention)
    assert isinstance(block.cross_attn.attn, USPAttention)
    assert block.self_attn.attn.skip_sequence_parallel is True
    assert block.self_attn.sequence_parallel_attn.skip_sequence_parallel is False
    assert block.cross_attn.attn.skip_sequence_parallel is True


def test_dreamzero_cached_self_attention_attends_prefix_and_current_tokens(monkeypatch):
    _patch_uninitialized_usp_ring_group(monkeypatch)

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> torch.Tensor:
            self.calls.append((q.shape, k.shape, v.shape))
            return torch.zeros_like(q)

    stage = DreamZeroCausalWanSelfAttention(dim=8, num_heads=2, frame_seqlen=2)
    fake_attn = FakeAttention()
    stage.attn = fake_attn

    x = torch.randn(1, 4, 8)
    freqs_cis = (torch.ones(4, 2), torch.zeros(4, 2))
    kv_cache = torch.zeros(2, 1, 0, 2, 4)

    output, updated_cache = stage(
        x=x,
        freqs_cis=freqs_cis,
        action_register_length=None,
        kv_cache=kv_cache,
    )

    assert output.shape == x.shape
    assert updated_cache.shape == (2, 1, 4, 2, 4)
    assert fake_attn.calls == [
        (
            torch.Size([1, 4, 2, 4]),
            torch.Size([1, 4, 2, 4]),
            torch.Size([1, 4, 2, 4]),
        )
    ]


def test_dreamzero_global_sequence_shard_includes_action_state(monkeypatch):
    seqs = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    freqs = (torch.ones(5, 4), torch.zeros(5, 4))

    monkeypatch.setattr(dreamzero_dit, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(dreamzero_dit, "get_ulysses_parallel_rank", lambda: 0)
    local0, freqs0, seq_lens = dreamzero_dit._sp_shard_sequence(seqs, freqs)

    monkeypatch.setattr(dreamzero_dit, "get_ulysses_parallel_rank", lambda: 1)
    local1, freqs1, _ = dreamzero_dit._sp_shard_sequence(seqs, freqs)

    assert seq_lens == [3, 2]
    assert torch.equal(local0, seqs[:, :3])
    assert torch.equal(local1, seqs[:, 3:])
    assert freqs0[0].shape[0] == 3
    assert freqs1[0].shape[0] == 2


def test_dreamzero_text_stage_inherits_standard_text_encoding_and_preserves_folding():
    config = DreamZeroPipelineConfig()
    config.text_encoder_configs[0].parallel_folding_mode = "world"
    server_args = types.SimpleNamespace(
        enable_cfg_parallel=True,
        pipeline_config=config,
    )
    stage = DreamZeroTextEncodingStage(
        text_encoder=_FakeTextEncoder(),
        tokenizer=_FakeTokenizer(),
        cache_manager=DreamZeroCachePoolManager(max_sessions=1),
    )
    stage.server_args = server_args

    assert isinstance(stage, TextEncodingStage)
    assert stage.parallelism_type is StageParallelismType.REPLICATED
    assert config.text_encoder_configs[0].parallel_folding_mode == "world"


def test_dreamzero_visual_stage_inherits_standard_image_encoding_without_dedup():
    stage = DreamZeroVisualEncodingStage(
        image_encoder=torch.nn.Identity(),
        vae=torch.nn.Identity(),
        cache_manager=DreamZeroCachePoolManager(max_sessions=1),
    )

    assert isinstance(stage, ImageEncodingStage)
    assert stage.deduplicated_output_fields == ()


def test_dreamzero_text_stage_requires_string_prompts():
    assert DreamZeroTextEncodingStage._batched_texts("pick", 2, "prompt") == [
        "pick",
        "pick",
    ]
    assert DreamZeroTextEncodingStage._batched_texts(
        None, 2, "negative_prompt", default=""
    ) == ["", ""]

    with pytest.raises(ValueError, match="prompt is required"):
        DreamZeroTextEncodingStage._batched_texts(None, 1, "prompt")
    with pytest.raises(TypeError, match="prompt must be a string or list of strings"):
        DreamZeroTextEncodingStage._batched_texts(["pick", None], 2, "prompt")
    with pytest.raises(ValueError, match="prompt batch size mismatch"):
        DreamZeroTextEncodingStage._batched_texts(["pick"], 2, "prompt")


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


def test_dreamzero_action_request_builder_uses_runtime_steps_and_server_stable_scale():
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
    assert not hasattr(params, "guidance_scale")
    assert extra["dreamzero_session_ids"] == ["session-a"]
    assert extra["dreamzero_reset_mask"] == [True]
    assert extra["dreamzero_prompts"] == ["pick the cube"]
    assert extra["dreamzero_normalized_input"]["state"].shape == (2,)
    assert "dreamzero_action_horizon" not in extra
    assert "dreamzero_relative_action_per_horizon" not in extra
    assert "dreamzero_embodiment_tag" not in extra


def test_dreamzero_rollout_schedulers_use_request_num_inference_steps():
    class FakeScheduler:
        def __init__(self) -> None:
            self.calls = []
            self.timesteps = []

        def set_timesteps(self, steps, *, device, shift):
            self.calls.append((steps, device, shift))
            self.timesteps = torch.arange(steps, device=device)

    created_schedulers = [FakeScheduler(), FakeScheduler()]
    pending_schedulers = list(created_schedulers)
    stage = object.__new__(DreamZeroCausalDenoisingStage)
    stage._new_unipc_scheduler = lambda: pending_schedulers.pop(0)
    stage._prepare_rollout_state = lambda ctx: torch.zeros(1, 1)
    stage._rollout_step_prediction = lambda **_: (
        torch.zeros(1, 1, 1),
        torch.zeros(1, 2),
    )
    stage._scheduler_step = (
        lambda scheduler, model_output, timestep, sample, step_index: sample
    )

    batch = types.SimpleNamespace(num_inference_steps=7)
    server_args = types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(
            default_num_inference_steps=16,
            flow_shift=5.0,
        )
    )
    ctx = types.SimpleNamespace(
        inputs=types.SimpleNamespace(device=torch.device("cpu")),
        noise=types.SimpleNamespace(
            noise_obs=torch.zeros(1, 1, 1),
            noise_action=torch.zeros(1, 2),
        ),
    )

    stage._run_action_rollout(batch, server_args, ctx)

    assert batch.dreamzero_action_pred.shape == (1, 2)
    assert [scheduler.calls[0][0] for scheduler in created_schedulers] == [7, 7]


def test_dreamzero_action_output_reports_request_num_inference_steps():
    stage = DreamZeroActionOutputStage()
    batch = types.SimpleNamespace(
        request_id="req-steps",
        dreamzero_action_pred=torch.zeros(1, 24, 7),
        num_inference_steps=7,
        metrics=None,
    )
    server_args = types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(default_num_inference_steps=16)
    )

    output = stage.forward(batch, server_args)

    assert output.output[0]["parameters"]["num_inference_steps"] == 7


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
    assert metadata["output"]["action_horizon"] == 24
    assert metadata["output"]["action_dim"] == 32
    assert metadata["output"]["padded_action_dim"] == 32
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
    step_with_index = DreamZeroCausalDenoisingStage._build_scheduler_step(
        SchedulerWithStepIndex()
    )
    step_without_index = DreamZeroCausalDenoisingStage._build_scheduler_step(
        SchedulerWithoutStepIndex()
    )

    assert torch.equal(
        step_with_index(
            SchedulerWithStepIndex(),
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            step_index=7,
        ),
        torch.tensor([2.0]),
    )
    assert torch.equal(
        step_without_index(
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


def test_dreamzero_prompt_cache_hashes_prompt_strings_and_explicit_hashes():
    manager = DreamZeroCachePoolManager(max_sessions=2)
    batch = types.SimpleNamespace(
        extra={
            "dreamzero_session_ids": ["session-a"],
            "dreamzero_reset_mask": [False],
            "dreamzero_prompts": ["pick"],
            "dreamzero_negative_prompts": [""],
        },
        dreamzero_inputs={},
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

    assert request_cache.prompt_hashes[0].startswith("str:")
    assert second_cache.prompt_reusable == [True]

    keyed_batch = types.SimpleNamespace(
        extra={
            "dreamzero_session_ids": ["session-b"],
            "dreamzero_reset_mask": [False],
            "dreamzero_prompts": [None],
            "dreamzero_negative_prompts": [None],
            "dreamzero_prompt_hashes": ["prompt-key"],
        },
        dreamzero_inputs={},
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

    first_batch = _prompt_text_batch(
        session_id="session-a",
        prompt="pick",
    )
    stage.forward(first_batch, server_args)
    assert manager.pool.cached_prompt_embs[BRANCH_COND].shape[0] == 1

    second_batch = _prompt_text_batch(
        session_id="session-b",
        prompt="place",
    )
    stage.forward(second_batch, server_args)

    assert second_batch.dreamzero_prompt_embs[0].shape[0] == 1
    assert manager.pool.cached_prompt_embs[BRANCH_COND].shape[0] == 2


def test_dreamzero_cfg_text_stage_is_replicated_and_encodes_both_branches():
    manager = DreamZeroCachePoolManager(max_sessions=1)
    stage, server_args, encoder = _make_text_stage(
        manager=manager,
        enable_cfg_parallel=True,
    )

    changed_cond_batch = _prompt_text_batch(
        session_id="session-a",
        prompt="new prompt",
        negative_prompt="",
    )
    stage.forward(changed_cond_batch, server_args)

    assert changed_cond_batch.dreamzero_cfg_branch_index is None
    assert len(changed_cond_batch.dreamzero_prompt_embs) == 2
    assert torch.equal(
        changed_cond_batch.negative_prompt_embeds,
        changed_cond_batch.dreamzero_prompt_embs[1],
    )
    assert encoder.calls == 2

    stable_cond_batch = _prompt_text_batch(
        session_id="session-a",
        prompt="new prompt",
        negative_prompt="",
    )
    stage.forward(stable_cond_batch, server_args)

    assert encoder.calls == 2


def test_dreamzero_text_encoding_masks_padding_without_python_seq_len_sync():
    stage = DreamZeroTextEncodingStage()
    prompt_emb = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    output = stage._mask_prompt_padding(
        prompt_emb,
        torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long),
    )

    assert torch.equal(output[0, 2:], torch.zeros_like(output[0, 2:]))
    assert torch.equal(output[1, 3:], torch.zeros_like(output[1, 3:]))
    assert torch.equal(
        output[1, :3],
        torch.tensor([[8, 9], [10, 11], [12, 13]], dtype=torch.float32),
    )
