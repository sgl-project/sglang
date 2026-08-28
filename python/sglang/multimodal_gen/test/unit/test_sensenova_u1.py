# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.sensenova_u1 import (
    SenseNovaU1PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1SamplingParams,
)
from sglang.multimodal_gen.configs.sensenova_u1 import (
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
    is_registered_diffusion_model_path,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1GenerationStage,
)


class _FakeSenseNovaModel:
    def __init__(self):
        self.call_kwargs = None

    def t2i_generate(self, tokenizer, prompt, **kwargs):
        self.call_kwargs = {"tokenizer": tokenizer, "prompt": prompt, **kwargs}
        return torch.tensor(
            [
                [
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                ]
            ]
        )


def test_sensenova_u1_registry_resolves_local_and_hf_paths(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: None,
    )
    get_model_info.cache_clear()

    local_path = "/models/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT"
    assert is_registered_diffusion_model_path(local_path)
    assert get_non_diffusers_pipeline_name(local_path) == "SenseNovaU1Pipeline"

    model_info = get_model_info("sensenova/SenseNova-U1.5-8B-MoT")
    assert model_info is not None
    assert model_info.pipeline_config_cls is SenseNovaU1PipelineConfig
    assert model_info.sampling_param_cls is SenseNovaU1SamplingParams
    get_model_info.cache_clear()


def test_sensenova_u1_registry_does_not_route_lora_only_repositories():
    lora_repo = "sensenova/SenseNova-U1.5-8B-MoT-LoRA"
    lora_path = "/models/SenseNova-U1.5-8B-MoT-LoRA"

    assert get_non_diffusers_pipeline_name(lora_repo) is None
    assert get_non_diffusers_pipeline_name(lora_path) is None


def test_sensenova_u1_sampling_params_keep_private_defaults_internal():
    params = SenseNovaU1SamplingParams(prompt="hello", width=2304, height=4096)

    assert params.guidance_scale == 4.0
    assert params.num_inference_steps == 50
    assert params.num_outputs_per_prompt == 1
    assert params.cfg_norm == "none"
    assert params.timestep_shift == 3.0

    extra = params.build_request_extra()[SENSENOVA_U1_REQUEST_EXTRA_KEY]
    assert extra == {
        "cfg_norm": "none",
        "timestep_shift": 3.0,
        "enable_timestep_shift": True,
        "cfg_interval": (0.0, 1.0),
        "t_eps": 0.02,
        "think_mode": False,
    }


def test_sensenova_u1_rejects_unaligned_resolution():
    with pytest.raises(ValueError, match="divisible by 32"):
        SenseNovaU1SamplingParams(width=2160, height=3840)


def test_sensenova_u1_accepts_openai_image_api_num_frames():
    params = SenseNovaU1SamplingParams(
        prompt="hello",
        width=2048,
        height=2048,
        num_frames=1,
    )

    assert params.num_frames == 1
    assert params.data_type == DataType.IMAGE


def test_sensenova_u1_scheduler_capabilities():
    config = SenseNovaU1PipelineConfig()

    assert not config.supports_dynamic_batching()
    assert config.supports_sequential_multi_output_inference()


def test_sensenova_u1_rejects_multi_gpu_during_arg_validation():
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="num_gpus=1"):
        config.validate_server_args(
            SimpleNamespace(
                num_gpus=2,
                enable_torch_compile=False,
                attention_backend=None,
                component_attention_backends={},
            )
        )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"enable_torch_compile": True}, "torch.compile"),
        ({"attention_backend": "fa"}, "custom attention backends"),
        (
            {"component_attention_backends": {"text_encoder": "torch_sdpa"}},
            "component attention backends",
        ),
        ({"attention_backend_config": {"foo": "bar"}}, "attention backend config"),
    ],
)
def test_sensenova_u1_rejects_unsupported_runtime_modes(override, expected):
    config = SenseNovaU1PipelineConfig()
    args = {
        "num_gpus": 1,
        "enable_torch_compile": False,
        "attention_backend": None,
        "component_attention_backends": {},
        "attention_backend_config": {},
    }
    args.update(override)

    with pytest.raises(ValueError, match=expected):
        config.validate_server_args(SimpleNamespace(**args))


def test_sensenova_u1_vision_config_round_trips_sequence_fields(tmp_path):
    config = NEOVisionConfig(llm_hidden_size=2048, downsample_ratio=0.5)
    config.save_pretrained(tmp_path)

    loaded = NEOVisionConfig.from_pretrained(tmp_path)

    assert loaded.llm_hidden_size == (2048,)
    assert loaded.downsample_ratio == (0.5,)


def test_sensenova_u1_vision_config_normalizes_nested_singletons():
    config = NEOVisionConfig(llm_hidden_size=[[2048]], downsample_ratio=[[0.5]])

    assert config.llm_hidden_size == (2048,)
    assert config.downsample_ratio == (0.5,)


def test_sensenova_u1_rejects_video_frame_count():
    with pytest.raises(ValueError, match="num_frames=1"):
        SenseNovaU1SamplingParams(width=2048, height=2048, num_frames=2)


def test_sensenova_u1_cli_args_expose_only_sglang_compatible_fields():
    args = SimpleNamespace(
        prompt="hello",
        width=2304,
        height=4096,
        guidance_scale=4.5,
        num_inference_steps=30,
        num_outputs_per_prompt=2,
        cfg_norm="global",
        timestep_shift=9.0,
        think_mode=True,
    )

    cli_args = SenseNovaU1SamplingParams.get_cli_args(args)

    assert cli_args["prompt"] == "hello"
    assert cli_args["width"] == 2304
    assert cli_args["height"] == 4096
    assert cli_args["guidance_scale"] == 4.5
    assert cli_args["num_inference_steps"] == 30
    assert cli_args["num_outputs_per_prompt"] == 2
    assert "cfg_norm" not in cli_args
    assert "timestep_shift" not in cli_args
    assert "think_mode" not in cli_args


def test_sensenova_u1_generation_stage_uses_sglang_params_and_single_model_batch():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        guidance_scale=4.5,
        num_inference_steps=30,
        seed=123,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        extra=sampling.build_request_extra(),
        metrics=None,
    )
    model = _FakeSenseNovaModel()
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")

    output = stage.forward(batch, server_args=SimpleNamespace())

    assert len(output.output) == 1
    assert torch.allclose(
        output.output[0],
        torch.tensor(
            [
                [[0.0, 0.5], [0.75, 1.0]],
                [[0.0, 0.5], [0.75, 1.0]],
                [[0.0, 0.5], [0.75, 1.0]],
            ]
        ),
    )
    assert model.call_kwargs["tokenizer"] == "tok"
    assert model.call_kwargs["prompt"] == "a mountain lake"
    assert model.call_kwargs["image_size"] == (2304, 4096)
    assert model.call_kwargs["cfg_scale"] == 4.5
    assert model.call_kwargs["num_steps"] == 30
    assert model.call_kwargs["batch_size"] == 1
    assert model.call_kwargs["seed"] == 123


def test_sensenova_u1_generation_stage_expects_expanded_multi_output_request():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
    )
    batch = SimpleNamespace(
        num_outputs_per_prompt=2,
        seed=42,
        extra=sampling.build_request_extra(),
    )
    stage = SenseNovaU1GenerationStage(model=_FakeSenseNovaModel(), tokenizer="tok")

    with pytest.raises(ValueError, match="output expansion"):
        stage.forward(batch, server_args=SimpleNamespace())
