import types

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
    DreamZeroPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.dreamzero import DreamZeroSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import DataType, SamplingParams
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    BRANCH_COND,
    BRANCH_UNCOND,
    DreamZeroCachePoolManager,
    apply_request_lifecycle_resets,
    resolve_request_cache,
)
from sglang.multimodal_gen.runtime.models.encoders.dreamzero_image import (
    WanImageEncoderStateDictConverter,
    VisionAttentionBlock,
    VisionSelfAttention,
    XLMRobertaAttentionBlock,
    XLMRobertaSelfAttention,
)
from sglang.multimodal_gen.runtime.models.encoders.dreamzero_text import (
    WanTextEncoder,
    WanTextEncoderStateDictConverter,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_checkpoint_utils import (
    DreamZeroCheckpointLoadReport,
    load_matching_tensors,
)
from sglang.multimodal_gen.runtime.pipelines.dreamzero_pipeline import DreamZeroPipeline
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

    assert config.task_type is ModelTaskType.ACTION
    assert config.task_type.data_type() is DataType.ACTION
    assert params.data_type is DataType.ACTION
    assert params.build_request_extra()["dreamzero_action_horizon"] == 24


def test_action_api_additions_do_not_change_non_dreamzero_defaults():
    req = Req(sampling_params=SamplingParams())

    assert req.sampling_params.data_type is DataType.VIDEO
    assert ModelTaskType.T2V.data_type() is DataType.VIDEO
    assert ModelTaskType.I2V.data_type() is DataType.VIDEO
    assert ModelTaskType.T2I.data_type() is DataType.IMAGE
    assert ModelTaskType.I2M.data_type() is DataType.MESH
    assert ModelTaskType.T2V.requires_image_input() is False
    assert ModelTaskType.I2V.requires_image_input() is True
    assert ModelTaskType.ACTION.requires_image_input() is False
    assert ModelTaskType.ACTION.accepts_image_input() is False
    assert req.session_id is None
    assert req.reset_session is False


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
            return values.reshape(input_ids.shape[0], input_ids.shape[1], 2)

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
    class Report(DreamZeroCheckpointLoadReport):
        include_fallback_impl = True

    class SmallModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.mismatch = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("scale", torch.zeros(2))

    model = SmallModule()
    report = load_matching_tensors(
        model,
        [
            ("prefix.weight", torch.ones(2, 2)),
            ("prefix.scale", torch.ones(2) * 2),
            ("prefix.mismatch", torch.ones(3)),
            ("prefix.unused", torch.ones(1)),
        ],
        device=torch.device("cpu"),
        key_mapper=lambda key: key.removeprefix("prefix."),
        report_cls=Report,
        fallback_impl="test.Loader",
    )

    assert torch.equal(model.weight, torch.ones(2, 2))
    assert torch.equal(model.scale, torch.ones(2) * 2)
    assert report.loaded_keys == ["weight", "scale"]
    assert report.unexpected_keys == ["unused"]
    assert report.shape_mismatches == {"mismatch": ((2,), (3,))}
    assert report.as_dict()["fallback_impl"] == "test.Loader"


def test_dreamzero_encoder_converters_and_lightweight_text_forward():
    assert XLMRobertaAttentionBlock(dim=4, num_heads=2, post_norm=True).attn.__class__ is (
        XLMRobertaSelfAttention
    )
    assert VisionAttentionBlock(dim=4, mlp_ratio=2, num_heads=2).attn.__class__ is (
        VisionSelfAttention
    )

    image_converter = WanImageEncoderStateDictConverter()
    converted = image_converter.from_civitai(
        {
            "visual.weight": torch.tensor([1.0]),
            "textual.weight": torch.tensor([2.0]),
        }
    )
    assert list(converted) == ["model.visual.weight"]

    text_converter = WanTextEncoderStateDictConverter()
    text_state = {"token_embedding.weight": torch.ones(4, 4)}
    assert text_converter.from_diffusers(text_state) is text_state
    assert text_converter.from_civitai(text_state) is text_state

    encoder = WanTextEncoder(
        vocab=8,
        dim=4,
        dim_attn=4,
        dim_ffn=8,
        num_heads=2,
        num_layers=1,
        num_buckets=4,
        dropout=0.0,
    )
    output = encoder(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert output.shape == (1, 3, 4)
