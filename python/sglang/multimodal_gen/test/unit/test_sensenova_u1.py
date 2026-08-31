# SPDX-License-Identifier: Apache-2.0
import json
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
    _get_config_info,
    get_model_info,
    get_non_diffusers_pipeline_name,
    is_registered_diffusion_model_path,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    _randn_with_seed,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1GenerationStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot


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


def _force_generator_fallback(monkeypatch, device_type):
    original_generator = torch.Generator

    def unsupported_device_generator(device="cpu"):
        if torch.device(device).type == device_type:
            raise RuntimeError(f"Generator is unsupported on {device_type}")
        return original_generator(device)

    monkeypatch.setattr(torch, "Generator", unsupported_device_generator)


def test_sensenova_u1_randn_fallback_preserves_cpu_rng(monkeypatch):
    _force_generator_fallback(monkeypatch, "cpu")
    rng_state = torch.get_rng_state().clone()

    first = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_sensenova_u1_randn_fallback_preserves_device_rng(monkeypatch):
    device_type = current_platform.device_type
    if not device_type or device_type == "cpu":
        pytest.skip("No accelerator is available")

    device = torch.device(device_type, 0)
    device_module = torch.get_device_module(device)
    if not device_module.is_available():
        pytest.skip(f"{device_type} is not available")

    _force_generator_fallback(monkeypatch, device_type)
    cpu_rng_state = torch.get_rng_state().clone()
    device_rng_state = device_module.get_rng_state(device).clone()

    first = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    assert torch.equal(device_module.get_rng_state(device), device_rng_state)


def test_sensenova_u1_registry_resolves_local_and_hf_paths(tmp_path):
    _get_config_info.cache_clear()
    get_model_info.cache_clear()

    local_path = tmp_path / "checkpoint-revision-abc123"
    local_path.mkdir()
    (local_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NEOChatModel"],
                "model_type": "neo_chat",
            }
        )
    )

    assert is_registered_diffusion_model_path(str(local_path))
    assert get_non_diffusers_pipeline_name(str(local_path)) == "SenseNovaU1Pipeline"

    local_model_info = get_model_info(str(local_path))
    assert local_model_info is not None
    assert local_model_info.pipeline_config_cls is SenseNovaU1PipelineConfig
    assert local_model_info.sampling_param_cls is SenseNovaU1SamplingParams

    model_info = get_model_info("sensenova/SenseNova-U1.5-8B-MoT")
    assert model_info is not None
    assert model_info.pipeline_config_cls is SenseNovaU1PipelineConfig
    assert model_info.sampling_param_cls is SenseNovaU1SamplingParams

    modelscope_id = "SenseNova/SenseNova-U1.5-8B-MoT"
    assert is_registered_diffusion_model_path(modelscope_id)
    assert get_non_diffusers_pipeline_name(modelscope_id) == "SenseNovaU1Pipeline"
    assert get_model_info(modelscope_id) is not None
    get_model_info.cache_clear()


def test_sensenova_u1_registry_requires_exact_hub_id(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: {},
    )
    _get_config_info.cache_clear()
    get_model_info.cache_clear()

    unrelated_repo = "acme/SenseNova-U1.5-8B-MoT"
    assert not is_registered_diffusion_model_path(unrelated_repo)
    assert get_non_diffusers_pipeline_name(unrelated_repo) is None
    assert _get_config_info(unrelated_repo) is None

    _get_config_info.cache_clear()
    get_model_info.cache_clear()


def test_sensenova_u1_registry_does_not_route_lora_only_repositories(tmp_path):
    lora_repo = "sensenova/SenseNova-U1.5-8B-MoT-LoRA"
    lora_path = tmp_path / "SenseNova-U1.5-8B-MoT-LoRA"
    lora_path.mkdir()
    (lora_path / "adapter_config.json").write_text("{}")

    assert get_non_diffusers_pipeline_name(lora_repo) is None
    assert get_non_diffusers_pipeline_name(str(lora_path)) is None
    assert not is_registered_diffusion_model_path(lora_repo)
    assert not is_registered_diffusion_model_path(str(lora_path))


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


def test_sensenova_u1_multi_output_request_expands_before_generation_stage():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
    )
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=42,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    input_stage = InputValidationStage()
    stage = SenseNovaU1GenerationStage(model=_FakeSenseNovaModel(), tokenizer="tok")
    batch.metrics.record_stage("InputValidationStage", 0.125)
    batch.metrics.record_memory_snapshot(
        "after_validation",
        MemorySnapshot(
            allocated_mb=100.0,
            reserved_mb=200.0,
            peak_allocated_mb=300.0,
            peak_reserved_mb=400.0,
        ),
    )

    expanded = list(input_stage.iter_sequential_requests(batch, server_args))

    assert [req.num_outputs_per_prompt for req in expanded] == [1, 1]
    assert [req.seed for req in expanded] == [42, 43]
    assert [req.request_id for req in expanded] == ["req-0:0", "req-0:1"]
    assert [req.output_file_name for req in expanded] == [
        "sample_0.png",
        "sample_1.png",
    ]
    assert [req.metrics.request_id for req in expanded] == ["req-0:0", "req-0:1"]
    assert all(req.trace_ctx is batch.trace_ctx for req in expanded)
    assert all(req.metrics is not batch.metrics for req in expanded)
    assert expanded[0].metrics is not expanded[1].metrics
    assert all(
        req.metrics.stages == {"InputValidationStage": 125.0} for req in expanded
    )
    assert all(
        req.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for req in expanded
    )
    assert (
        expanded[0].metrics.memory_snapshots["after_validation"]
        is not expanded[1].metrics.memory_snapshots["after_validation"]
    )

    expanded[0].metrics.record_stage("child-only", 0.5)
    expanded[0].metrics.memory_snapshots["after_validation"].peak_reserved_mb = 999.0
    assert "child-only" not in expanded[1].metrics.stages
    assert "child-only" not in batch.metrics.stages
    assert (
        expanded[1].metrics.memory_snapshots["after_validation"].peak_reserved_mb
        == 400.0
    )
    assert batch.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0

    for req in expanded:
        output = stage.forward(req, server_args=SimpleNamespace())
        assert len(output.output) == 1
