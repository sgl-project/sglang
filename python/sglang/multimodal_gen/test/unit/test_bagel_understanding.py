# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from fastapi import HTTPException
from PIL import Image

from sglang.multimodal_gen.configs.models.dits.bagel import (
    BagelDiTArchConfig,
    BagelDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    BagelUnderstandingPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.bagel import (
    BagelUnderstandingSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.openai.chat_api import (
    _resolve_max_new_tokens,
    _validate_request_features,
    chat_completions,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.models.dits.bagel_transformer import (
    BagelKVCache,
    BagelPrefixContext,
    BagelTransformer,
)
from sglang.multimodal_gen.runtime.pipelines.bagel_pipeline import (
    BagelUnderstandingPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
    TextGenerationOutput,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel import (
    VLM_THINK_SYSTEM_PROMPT,
    BagelUnderstandingInputValidationStage,
    BagelUnderstandingStage,
)
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest


class _Tokenizer:
    token_ids = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
    }
    unk_token_id = 0
    unk_token = "<unk>"

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text in self.token_ids:
            return [self.token_ids[text]]
        self.encoded_texts.append(text)
        if text == VLM_THINK_SYSTEM_PROMPT:
            return [29, 31]
        return [17, 23]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        assert not skip_special_tokens
        assert token_ids[0] == self.token_ids["<|im_start|>"]
        return "<|im_start|>A blue square.<|im_end|>ignored"


class _ImageEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.image_sizes: list[tuple[int, int]] = []

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        self.image_sizes.append(image.size)
        return torch.full((3, 8), 0.25)


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.prefix_call: dict[str, object] = {}
        self.generation_call: dict[str, object] = {}

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def build_understanding_prefix(
        self,
        vision_embeddings: torch.Tensor,
        user_input_ids: torch.Tensor,
        **kwargs,
    ) -> SimpleNamespace:
        self.prefix_call = {
            "vision_embeddings": vision_embeddings.clone(),
            "user_input_ids": user_input_ids.clone(),
            **kwargs,
        }
        return SimpleNamespace(
            kv_cache=SimpleNamespace(sequence_length=11),
            rope_offset=6,
        )

    def generate_text(
        self, prefix: SimpleNamespace, **kwargs
    ) -> tuple[torch.Tensor, str]:
        self.generation_call = {"prefix": prefix, **kwargs}
        return torch.tensor([151644, 41, 42]), "length"


def _runtime_args(config: BagelUnderstandingPipelineConfig) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=config,
        enable_cfg_parallel=False,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        use_fsdp_inference=False,
        enable_torch_compile=False,
        dit_layerwise_offload=False,
        layerwise_offload_components=None,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        cache_dit_config=None,
        quantization=None,
        lora_path=None,
        comfyui_mode=False,
        revision=None,
        output_path=None,
        model_paths={},
        model_loaded={
            "transformer": False,
            "image_encoder": False,
            "tokenizer": False,
        },
    )


def _understanding_batch(*, enable_thinking: bool = True) -> Req:
    params = BagelUnderstandingSamplingParams(
        prompt="What color is the square?",
        image_path=["input.png"],
        seed=9,
        max_new_tokens=7,
        enable_thinking=enable_thinking,
        save_output=False,
    )
    batch = Req(sampling_params=params)
    batch.condition_image = [Image.new("RGB", (64, 32), color="blue")]
    return batch


@pytest.fixture
def tiny_understanding_transformer() -> BagelTransformer:
    arch = BagelDiTArchConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_head_dim=4,
        vocab_size=32,
        max_position_embeddings=32,
        latent_patch_size=2,
        max_latent_size=2,
        latent_channel=1,
        latent_downsample=2,
        timestep_frequency_embedding_size=4,
        latent_position_embedding_rows=4,
        start_of_image_token_id=3,
        end_of_image_token_id=4,
    )
    torch.manual_seed(0)
    return BagelTransformer(
        BagelDiTConfig(
            arch_config=arch,
            load_lm_head=True,
            load_generation_expert=False,
        ),
        attention_backend="torch_sdpa",
    ).eval()


def test_understanding_params_and_config_are_text_only() -> None:
    config = BagelUnderstandingPipelineConfig()
    params = BagelUnderstandingSamplingParams(
        prompt="describe the image", image_path=["input.png"]
    )

    assert config.task_type is ModelTaskType.I2T
    assert config.task_type.data_type() is DataType.TEXT
    assert config.task_type.requires_image_input()
    assert config.dit_config.load_lm_head
    assert not config.dit_config.load_generation_expert
    assert config.get_model_deployment_config().keep_resident_components == (
        "dit",
        "image_encoder",
    )
    assert params.max_new_tokens == 512
    assert not params.do_sample
    assert params.temperature == 0.3
    assert not params.enable_thinking
    assert not params.save_output
    assert not params.return_file_paths_only

    params._adjust(
        SimpleNamespace(
            pipeline_config=config,
            output_path=None,
            comfyui_mode=False,
        )
    )
    assert params.data_type is DataType.TEXT
    assert params.num_frames == 1
    assert not params.adjust_frames
    assert params.output_file_name.endswith(".txt")


def test_explicit_understanding_pipeline_resolves_slim_local_checkpoint(
    tmp_path,
) -> None:
    for file_name in ("config.json", "llm_config.json", "ema.safetensors"):
        (tmp_path / file_name).touch()

    config = PipelineConfig.from_kwargs(
        {
            "model_path": str(tmp_path),
            "pipeline_class_name": "BagelUnderstandingPipeline",
        }
    )

    assert isinstance(config, BagelUnderstandingPipelineConfig)
    assert not (tmp_path / "ae.safetensors").exists()


def test_unrelated_pipeline_cannot_bypass_native_checkpoint_detection(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not contain model_index.json"):
        PipelineConfig.from_kwargs(
            {
                "model_path": str(tmp_path),
                "pipeline_class_name": "MOVAPipeline",
            }
        )


def test_understanding_controls_are_available_to_offline_cli() -> None:
    parser = argparse.ArgumentParser()
    SamplingParams.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--max-new-tokens",
            "64",
            "--do-sample",
            "--temperature",
            "0.8",
            "--enable-thinking",
        ]
    )

    cli_args = BagelUnderstandingSamplingParams.get_cli_args(args)

    assert cli_args["max_new_tokens"] == 64
    assert cli_args["do_sample"] is True
    assert cli_args["temperature"] == 0.8
    assert cli_args["enable_thinking"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"do_sample": "yes"}, "do_sample"),
        ({"enable_thinking": 1}, "enable_thinking"),
        ({"do_sample": True, "temperature": 0.0}, "temperature"),
        ({"do_sample": True, "temperature": float("nan")}, "temperature"),
    ],
)
def test_understanding_params_reject_invalid_decode_controls(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BagelUnderstandingSamplingParams(prompt="describe", **kwargs)


def test_understanding_input_validation_applies_outer_resize_and_white_alpha(
    tmp_path,
) -> None:
    image_path = tmp_path / "transparent.png"
    Image.new("RGBA", (400, 800), color=(10, 20, 30, 0)).save(image_path)
    config = BagelUnderstandingPipelineConfig()
    params = BagelUnderstandingSamplingParams(
        prompt="describe",
        image_path=[str(image_path)],
        seed=7,
        save_output=False,
    )
    batch = Req(sampling_params=params)
    args = SimpleNamespace(pipeline_config=config, enable_cfg_parallel=False)

    output = BagelUnderstandingInputValidationStage().forward(batch, args)

    assert output.original_condition_image_size == (400, 800)
    assert (output.width, output.height) == (512, 1024)
    assert len(output.condition_image) == 1
    assert output.condition_image[0].mode == "RGB"
    assert output.condition_image[0].size == (512, 1024)
    assert output.condition_image[0].getpixel((0, 0)) == (255, 255, 255)
    assert output.generator[0].device.type == "cpu"


def test_understanding_prefix_is_system_then_image_then_user(
    tiny_understanding_transformer: BagelTransformer,
) -> None:
    model = tiny_understanding_transformer
    system_ids = torch.tensor([1, 2])
    user_ids = torch.tensor([5, 6])
    vision_embeddings = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24

    actual = model.build_understanding_prefix(
        vision_embeddings,
        user_ids,
        system_input_ids=system_ids,
        start_of_image_token_id=3,
        end_of_image_token_id=4,
    )
    expected = BagelPrefixContext(
        BagelKVCache.empty(model.num_layers),
        torch.zeros(1, dtype=torch.int32),
        0,
    )
    expected = model._append_text_prefix(expected, system_ids)
    expected = model._append_image_prefix(
        expected,
        vision_embeddings,
        mode="und",
        start_of_image_token_id=3,
        end_of_image_token_id=4,
    )
    expected = model._append_text_prefix(expected, user_ids)

    assert actual.kv_cache.sequence_length == 9
    assert actual.kv_lens.tolist() == [9]
    assert actual.rope_offset == 5
    for actual_key, expected_key in zip(
        actual.kv_cache.key_cache, expected.kv_cache.key_cache, strict=True
    ):
        assert actual_key is not None and expected_key is not None
        torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
    for actual_value, expected_value in zip(
        actual.kv_cache.value_cache, expected.kv_cache.value_cache, strict=True
    ):
        assert actual_value is not None and expected_value is not None
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_generate_text_reports_stop_and_length(
    tiny_understanding_transformer: BagelTransformer,
) -> None:
    model = tiny_understanding_transformer
    assert model.lm_head is not None
    model.lm_head.weight.data.zero_()
    prefix = model.prefill_context(torch.tensor([1, 2]))

    stopped_ids, stopped_reason = model.generate_text(
        prefix,
        bos_token_id=5,
        eos_token_id=0,
        max_length=4,
        return_finish_reason=True,
    )
    capped_ids, capped_reason = model.generate_text(
        prefix,
        bos_token_id=5,
        eos_token_id=31,
        max_length=2,
        return_finish_reason=True,
    )

    assert stopped_ids.tolist() == [5]
    assert stopped_reason == "stop"
    assert capped_ids.tolist() == [5, 0]
    assert capped_reason == "length"
    assert prefix.kv_cache.sequence_length == 2


def test_understanding_terminal_stage_returns_pure_text_with_metadata() -> None:
    config = BagelUnderstandingPipelineConfig()
    args = SimpleNamespace(pipeline_config=config)
    transformer = _Transformer()
    image_encoder = _ImageEncoder()
    tokenizer = _Tokenizer()
    stage = BagelUnderstandingStage(transformer, image_encoder, tokenizer)
    stage.server_args = args
    stage._registered_stage_name = "bagel_understanding_stage"
    batch = _understanding_batch(enable_thinking=True)

    output = stage.forward(batch, args)

    assert output.output is None
    assert output.output_file_paths is None
    assert output.revised_prompts is None
    assert output.text_outputs is not None
    assert output.text_outputs == [
        TextGenerationOutput(
            text="A blue square.",
            prompt_tokens=11,
            completion_tokens=2,
            finish_reason="length",
        )
    ]
    assert output.metrics is batch.metrics
    assert image_encoder.image_sizes == [(64, 32)]
    assert transformer.prefix_call["user_input_ids"].tolist() == [
        151644,
        17,
        23,
        151645,
    ]
    assert transformer.prefix_call["system_input_ids"].tolist() == [
        151644,
        29,
        31,
        151645,
    ]
    assert transformer.prefix_call["start_of_image_token_id"] == 151652
    assert transformer.prefix_call["end_of_image_token_id"] == 151653
    assert transformer.generation_call["max_length"] == 7
    assert transformer.generation_call["do_sample"] is False
    assert transformer.generation_call["return_finish_reason"] is True
    assert VLM_THINK_SYSTEM_PROMPT == (
        "You should first think about the reasoning process in the mind and then "
        "provide the user with the answer.\n"
        "The reasoning process is enclosed within <think> </think> tags, i.e. "
        "<think> reasoning process here </think> answer here"
    )


@pytest.mark.parametrize(
    "control",
    [
        "save_output",
        "return_file_paths_only",
        "return_frames",
        "return_raw_frames",
    ],
)
def test_understanding_rejects_media_and_file_output_controls(control: str) -> None:
    batch = _understanding_batch(enable_thinking=False)
    setattr(batch, control, True)

    with pytest.raises(ValueError, match="returns text directly"):
        BagelUnderstandingStage._validate_request(batch)


def test_understanding_loader_uses_only_injected_text_components() -> None:
    config = BagelUnderstandingPipelineConfig()
    args = _runtime_args(config)
    pipeline = BagelUnderstandingPipeline.__new__(BagelUnderstandingPipeline)
    pipeline.model_path = "must-not-be-resolved"
    pipeline.memory_usages = {}
    modules = {
        "transformer": torch.nn.Identity(),
        "image_encoder": torch.nn.Identity(),
        "tokenizer": _Tokenizer(),
    }

    with patch.object(
        BagelUnderstandingPipeline,
        "_resolve_checkpoint",
        side_effect=AssertionError("snapshot resolution must not run"),
    ):
        loaded = pipeline.load_modules(args, modules)

    assert loaded == modules
    assert set(pipeline.memory_usages) == {
        "transformer",
        "image_encoder",
        "tokenizer",
    }
    assert "vae" not in loaded
    assert "scheduler" not in loaded
    assert all(args.model_loaded[name] for name in loaded)


def test_offline_generator_transports_text_without_media_materialization() -> None:
    params = BagelUnderstandingSamplingParams(
        prompt="describe",
        image_path=["input.png"],
        seed=5,
        save_output=False,
    )
    params.data_type = DataType.TEXT
    generator = object.__new__(DiffGenerator)
    generator.server_args = SimpleNamespace(
        model_path="unused",
        prompt_file_path=None,
        batching_max_size=1,
        warmup=False,
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
        enable_trace=False,
    )
    scheduler_output = OutputBatch(
        text_outputs=[
            TextGenerationOutput(
                text="A blue square.",
                prompt_tokens=17,
                completion_tokens=4,
                finish_reason="stop",
            )
        ]
    )

    with (
        patch.object(
            SamplingParams,
            "from_user_sampling_params_args",
            return_value=params,
        ),
        patch.object(
            generator,
            "_send_to_scheduler_and_wait_for_response",
            return_value=scheduler_output,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator."
            "save_outputs",
            side_effect=AssertionError("text must not enter media materialization"),
        ) as save_mock,
    ):
        result = generator.generate({"prompt": "describe", "image_path": ["input.png"]})

    assert result is not None and not isinstance(result, list)
    assert result.text == "A blue square."
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 17
    assert result.completion_tokens == 4
    assert result.samples is None
    assert result.frames is None
    save_mock.assert_not_called()


def _chat_request(**overrides) -> ChatCompletionRequest:
    payload = {
        "model": "ByteDance-Seed/BAGEL-7B-MoT",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        "max_completion_tokens": 511,
        "temperature": 0,
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


def test_chat_api_returns_typed_text_and_openai_usage(tmp_path) -> None:
    request = _chat_request(reasoning_effort="high", seed=7)
    sampling = SimpleNamespace()
    batch = SimpleNamespace()
    scheduler_output = OutputBatch(
        text_outputs=[
            TextGenerationOutput(
                text="<think>brief</think> A blue square.",
                prompt_tokens=25,
                completion_tokens=8,
                finish_reason="stop",
            )
        ]
    )
    server_args = SimpleNamespace(
        input_save_path=None,
        pipeline_config=BagelUnderstandingPipelineConfig(),
    )
    save_image = AsyncMock(return_value=str(tmp_path / "input.png"))

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.chat_api."
            "get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.chat_api."
            "save_image_to_path",
            new=save_image,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.chat_api."
            "build_sampling_params",
            return_value=sampling,
        ) as build_sampling,
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.chat_api.prepare_request",
            return_value=batch,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.chat_api."
            "process_generation_batch",
            new=AsyncMock(return_value=([], scheduler_output)),
        ),
    ):
        response = asyncio.run(chat_completions(request, SimpleNamespace(headers={})))

    sampling_kwargs = build_sampling.call_args.kwargs
    assert sampling_kwargs["max_new_tokens"] == 512
    assert sampling_kwargs["do_sample"] is False
    assert sampling_kwargs["enable_thinking"] is True
    assert sampling_kwargs["save_output"] is False
    assert response.id.startswith("chatcmpl-")
    assert response.choices[0].message.content == (
        "<think>brief</think> A blue square."
    )
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 25
    assert response.usage.completion_tokens == 8
    assert response.usage.total_tokens == 33
    assert save_image.await_args.kwargs["prefer_remote_source"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"n": 2},
        {"stream": True},
        {"stop": "done"},
        {"top_p": 0.5},
        {"frequency_penalty": 0.5},
    ],
)
def test_chat_api_rejects_unsupported_openai_features(overrides) -> None:
    with pytest.raises(HTTPException) as error:
        _validate_request_features(_chat_request(**overrides))
    assert error.value.status_code == 400


def test_chat_api_token_limit_accounts_for_internal_bos() -> None:
    assert _resolve_max_new_tokens(_chat_request(max_completion_tokens=1)) == 2


def test_gpu_worker_never_materializes_text_as_media() -> None:
    worker = object.__new__(GPUWorker)
    output = OutputBatch(text_outputs=[TextGenerationOutput("answer", 10, 1, "stop")])
    req = SimpleNamespace(
        return_raw_frames=True,
        save_output=True,
        return_file_paths_only=True,
        return_frames=True,
    )
    save_paths = Mock(
        side_effect=AssertionError("text must not enter media materialization")
    )

    worker._materialize_output_transport(output, req, save_paths)

    assert output.text_outputs[0].text == "answer"
    save_paths.assert_not_called()
