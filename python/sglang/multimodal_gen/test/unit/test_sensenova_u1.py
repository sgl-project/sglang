# SPDX-License-Identifier: Apache-2.0
import asyncio
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
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.conversation import (
    get_conv_template,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    _randn_with_seed,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1GenerationStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs
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


class _RecordingTraceContext:
    tracing_enable = True

    def __init__(self):
        self.finish_count = 0
        self.started_slices = []
        self.finished_slices = []

    def trace_req_finish(self):
        self.finish_count += 1

    def trace_slice_start(self, name, level=0):
        self.started_slices.append((name, level))

    def trace_slice_end(self, name, level=0, **kwargs):
        self.finished_slices.append((name, level))


class _SequentialTestExecutor(PipelineExecutor):
    def __init__(self, server_args, *, fail=False, fail_request_ids=None):
        super().__init__(server_args)
        self.fail = fail
        self.fail_request_ids = set(fail_request_ids or [])
        self.executed_requests = []

    def execute_group(self, stages, batches, server_args):
        for batch in batches:
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
        return batches

    def execute(self, stages, batch, server_args):
        self.executed_requests.append(batch)
        if self.fail or batch.request_id in self.fail_request_ids:
            raise RuntimeError(f"generation failed for {batch.request_id}")
        return OutputBatch(
            output_file_paths=[batch.output_file_name],
            metrics=batch.metrics,
        )


class _SequentialTestPipeline:
    def __init__(self, server_args, *, fail=False, fail_request_ids=None):
        self.input_stage = InputValidationStage()
        self.executor = _SequentialTestExecutor(
            server_args,
            fail=fail,
            fail_request_ids=fail_request_ids,
        )

    def forward_batch_sequentially(self, batches, server_args):
        return self.executor.execute_group_sequentially(
            [self.input_stage, object()], batches, server_args
        )


class _WorkerBackedSchedulerClient:
    def __init__(self, worker):
        self.worker = worker

    async def forward(self, batches):
        return next(self.worker.execute_forward_sequentially(batches))


@pytest.mark.parametrize(
    ("template_name", "expected_system_message"),
    [
        (
            "Hermes-2",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "internlm2-chat",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "phi3-chat",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "internvl2_5",
            "\u4f60\u662f\u4e66\u751f\xb7\u4e07\u8c61\uff0c\u82f1\u6587\u540d\u662fInternVL\uff0c\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u3001\u6e05\u534e\u5927\u5b66\u53ca\u591a\u5bb6\u5408\u4f5c\u5355\u4f4d\u8054\u5408\u5f00\u53d1\u7684\u591a\u6a21\u6001\u5927\u8bed\u8a00\u6a21\u578b\u3002",
        ),
    ],
)
def test_sensenova_u1_conversation_preserves_upstream_system_prompt(
    template_name, expected_system_message
):
    assert get_conv_template(template_name).system_message == expected_system_message


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


@pytest.mark.parametrize("backend", ["auto", "sglang", "diffusers"])
def test_sensenova_u1_known_adapter_only_repo_rejected_before_backend_resolution(
    monkeypatch, backend
):
    def fail_model_index_download(_):
        raise AssertionError("adapter-only repo should not download model_index")

    def fail_diffusers_resolution(**_kwargs):
        raise AssertionError("adapter-only repo should not resolve diffusers info")

    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        fail_model_index_download,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry._get_diffusers_model_info",
        fail_diffusers_resolution,
    )
    get_model_info.cache_clear()

    loras_repo = "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"

    assert get_non_diffusers_pipeline_name(loras_repo) is None
    assert get_model_info(loras_repo, backend=backend) is None
    get_model_info.cache_clear()


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
                lora_path=None,
                attention_backend=None,
                component_attention_backends={},
            )
        )


def test_sensenova_u1_clears_auto_tuned_runtime_defaults():
    config = SenseNovaU1PipelineConfig()
    args = SimpleNamespace(
        num_gpus=1,
        enable_torch_compile=False,
        lora_path=None,
        component_residency={"transformer": "layerwise-offload"},
        cpu_offload_components=["transformer"],
        dit_cpu_offload=True,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        dit_layerwise_offload=True,
        layerwise_offload_components=["transformer"],
        quantization=None,
        quantization_ignored_layers=None,
        transformer_weights_path=None,
        component_paths={"model": "/tmp/component"},
        component_weights_paths={"model": "/tmp/model.safetensors"},
        component_quantizations={},
        component_quantization_ignored_layers={},
        component_precisions={},
        attention_backend="aiter",
        component_attention_backends={"text_encoder": "torch_sdpa"},
        attention_backend_config={"foo": "bar"},
        is_arg_explicitly_set=lambda _name: False,
    )

    config.validate_server_args(args)

    assert args.component_residency is None
    assert args.cpu_offload_components is None
    assert args.dit_cpu_offload is False
    assert args.text_encoder_cpu_offload is False
    assert args.image_encoder_cpu_offload is False
    assert args.vae_cpu_offload is False
    assert args.dit_layerwise_offload is False
    assert args.layerwise_offload_components is None
    assert args.component_paths == {}
    assert args.component_weights_paths == {}
    assert args.attention_backend is None
    assert args.component_attention_backends == {}
    assert args.attention_backend_config is None


def test_sensenova_u1_allows_explicit_resident_component_residency():
    config = SenseNovaU1PipelineConfig()
    args = SimpleNamespace(
        num_gpus=1,
        enable_torch_compile=False,
        lora_path=None,
        component_residency={"transformer": "resident"},
        cpu_offload_components=None,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_layerwise_offload=False,
        layerwise_offload_components=None,
        quantization=None,
        quantization_ignored_layers=None,
        transformer_weights_path=None,
        component_paths={},
        component_weights_paths={},
        component_quantizations={},
        component_quantization_ignored_layers={},
        component_precisions={},
        attention_backend=None,
        component_attention_backends={},
        attention_backend_config={},
        is_arg_explicitly_set=lambda name: name == "component_residency",
    )

    config.validate_server_args(args)

    assert args.component_residency == {"transformer": "resident"}


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"enable_torch_compile": True}, "torch.compile"),
        ({"lora_path": "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"}, "LoRA adapters"),
        (
            {"component_residency": {"transformer": "component-offload"}},
            "component residency offload",
        ),
        ({"cpu_offload_components": ["transformer"]}, "CPU offload"),
        ({"dit_cpu_offload": True}, "DiT CPU offload"),
        ({"text_encoder_cpu_offload": True}, "text encoder CPU offload"),
        ({"image_encoder_cpu_offload": True}, "image encoder CPU offload"),
        ({"vae_cpu_offload": True}, "VAE CPU offload"),
        ({"dit_layerwise_offload": True}, "DiT layerwise offload"),
        ({"layerwise_offload_components": ["transformer"]}, "layerwise offload"),
        ({"quantization": "fp8"}, "quantization"),
        ({"quantization_ignored_layers": ["foo"]}, "quantization ignored layers"),
        (
            {"transformer_weights_path": "/tmp/transformer.safetensors"},
            "pre-quantized transformer weights",
        ),
        ({"component_paths": {"model": "/tmp/component"}}, "component path overrides"),
        (
            {"component_weights_paths": {"model": "/tmp/model.safetensors"}},
            "component weight path overrides",
        ),
        ({"component_quantizations": {"transformer": "fp8"}}, "component quantization"),
        (
            {"component_quantization_ignored_layers": {"transformer": ["foo"]}},
            "component quantization ignored layers",
        ),
        ({"component_precisions": {"transformer": "fp16"}}, "component precision"),
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
        "lora_path": None,
        "component_residency": None,
        "cpu_offload_components": None,
        "dit_cpu_offload": None,
        "text_encoder_cpu_offload": None,
        "image_encoder_cpu_offload": None,
        "vae_cpu_offload": False,
        "dit_layerwise_offload": None,
        "layerwise_offload_components": None,
        "quantization": None,
        "quantization_ignored_layers": None,
        "transformer_weights_path": None,
        "component_paths": {},
        "component_weights_paths": {},
        "component_quantizations": {},
        "component_quantization_ignored_layers": {},
        "component_precisions": {},
        "attention_backend": None,
        "component_attention_backends": {},
        "attention_backend_config": {},
    }
    args.update(override)

    with pytest.raises(ValueError, match=expected):
        config.validate_server_args(SimpleNamespace(**args))


def test_sensenova_u1_rejects_direct_server_args_quantization():
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="quantization"):
        ServerArgs(
            model_path="sensenova/SenseNova-U1.5-8B-MoT",
            pipeline_config=config,
            quantization="fp8",
        )


def test_sensenova_u1_rejects_file_valued_component_paths(tmp_path):
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="component weight path overrides"):
        ServerArgs(
            model_path="sensenova/SenseNova-U1.5-8B-MoT",
            pipeline_config=config,
            component_paths={"model": str(tmp_path / "model.safetensors")},
        )


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


def test_sensenova_u1_multi_output_rejects_short_seed_list():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
        seed=[7],
    )
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())

    with pytest.raises(ValueError, match="seed list length"):
        list(InputValidationStage().iter_sequential_requests(batch, server_args))


def _make_sensenova_u1_sequential_entrypoint(*, fail=False, fail_request_ids=None):
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
        save_output=False,
        suppress_logs=True,
    )
    trace_ctx = _RecordingTraceContext()
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        seed=42,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
        trace_ctx=trace_ctx,
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    pipeline = _SequentialTestPipeline(
        server_args, fail=fail, fail_request_ids=fail_request_ids
    )
    worker = GPUWorker.__new__(GPUWorker)
    worker.pipeline = pipeline
    worker.server_args = server_args
    worker.is_output_rank = True
    worker._runtime_peak_reserved_mb = 0.0
    worker._release_warmup_pool_before_serving = False
    worker._realtime_sessions = SimpleNamespace(attach=lambda _req: None)
    return batch, trace_ctx, pipeline.executor, _WorkerBackedSchedulerClient(worker)


def _force_cpu_entrypoint(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_mps", lambda: False)
    monkeypatch.setattr(current_platform, "is_npu", lambda: False)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.utils.get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )


def test_sensenova_u1_multi_output_entrypoint_success(monkeypatch):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint()
    )

    paths, result = asyncio.run(process_generation_batch(scheduler_client, batch))

    assert paths == ["sample_0.png", "sample_1.png"]
    assert result.error is None
    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert [req.seed for req in executor.executed_requests] == [42, 43]
    assert result.metrics_list is not None
    assert [metrics.request_id for metrics in result.metrics_list] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(
        "InputValidationStage" in metrics.stages
        and "PipelineExecutor.sequential_wait" in metrics.stages
        and metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for metrics in result.metrics_list
    )
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


def test_sensenova_u1_multi_output_entrypoint_failure(monkeypatch):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint(fail=True)
    )

    with pytest.raises(RuntimeError, match="generation failed for req-0:0"):
        asyncio.run(process_generation_batch(scheduler_client, batch))

    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(
        "InputValidationStage" in req.metrics.stages
        and "PipelineExecutor.sequential_wait" in req.metrics.stages
        and req.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for req in executor.executed_requests
    )
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


@pytest.mark.parametrize("failed_request_id", ["req-0:0", "req-0:1"])
def test_sensenova_u1_multi_output_entrypoint_mixed_failure_fails_parent(
    monkeypatch, failed_request_id
):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint(fail_request_ids={failed_request_id})
    )

    with pytest.raises(
        RuntimeError, match=f"generation failed for {failed_request_id}"
    ):
        asyncio.run(process_generation_batch(scheduler_client, batch))

    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


# ===== Shared RoPE tables =====

_HIDDEN_DIM, _HEADS, _KV_HEADS, _HEAD_DIM, _LAYERS, _INTER = 8, 2, 1, 8, 2, 16


def _tiny_dense_config():
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
        NEOLLMConfig,
    )

    config = NEOLLMConfig(
        hidden_size=_HIDDEN_DIM,
        intermediate_size=_INTER,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_hidden_layers=_LAYERS,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
        vocab_size=32,
    )
    config.layer_types = ["full_attention"] * _LAYERS
    # forward_und asserts eager; the understanding path is unreachable otherwise.
    config._attn_implementation = "eager"
    return config


def _tiny_dense_attention(layer_idx=0):
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
        Qwen3Attention,
    )

    return Qwen3Attention(_tiny_dense_config(), layer_idx)


def test_sensenova_u1_shared_rope_tables_match_per_layer_computation():
    """Qwen3Model builds RoPE once per forward and shares it across layers.

    Guards the assumption that makes it valid: two identically configured layers
    derive identical tables from the same `indexes`. If the tables ever become
    layer-dependent, sharing them would silently change every attention output.
    """
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
        _build_neo_unify_rope_tables,
    )

    torch.manual_seed(0)
    first, other = _tiny_dense_attention(0), _tiny_dense_attention(1)
    x = torch.randn(1, 5, _HIDDEN_DIM)
    indexes = torch.arange(3 * 5).reshape(3, 5) % 3

    shared = _build_neo_unify_rope_tables(
        first.rotary_emb, first.rotary_emb_hw, indexes, x
    )
    reference_modules = (
        other.rotary_emb,
        other.rotary_emb_hw,
        other.rotary_emb_hw,
    )
    for (cos_shared, sin_shared), module, axis in zip(
        shared, reference_modules, (0, 1, 2)
    ):
        cos_ref, sin_ref = module(x, indexes[axis].unsqueeze(0))
        assert torch.equal(cos_shared, cos_ref), f"cos differs on axis {axis}"
        assert torch.equal(sin_shared, sin_ref), f"sin differs on axis {axis}"


class _RopeRebuildDetector(torch.nn.Module):
    """An nn.Module so it can take over a registered submodule slot."""

    def forward(self, *args, **kwargs):
        pytest.fail("rope tables rebuilt")


def test_sensenova_u1_attention_reuses_passed_rope_tables(monkeypatch):
    """Passing tables down must skip the per-layer rebuild.

    The failure mode is silent: reintroducing the rebuild keeps every output
    correct, so only this case notices that the sharing stopped happening.
    """
    attn = _tiny_dense_attention()
    shared = ("t", "h", "w")
    detector = _RopeRebuildDetector()
    monkeypatch.setattr(attn, "rotary_emb", detector)
    monkeypatch.setattr(attn, "rotary_emb_hw", detector)

    x = torch.randn(1, 5, _HIDDEN_DIM)
    indexes = torch.zeros(3, 5, dtype=torch.long)

    assert attn._resolve_rope_tables(x, indexes, shared) is shared


def test_sensenova_u1_model_shares_rope_tables_with_every_layer(monkeypatch):
    """Guards the wiring from the model loop down to attention.

    A dropped kwarg anywhere in that chain leaves every output correct, so
    without this case the sharing could stop happening unnoticed.
    """
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3Model,
    )

    model = Qwen3Model(_tiny_dense_config()).eval()
    seen = []
    original = Qwen3Attention._resolve_rope_tables

    def spy(self, hidden_states, indexes, rope_tables):
        seen.append(rope_tables is not None)
        return original(self, hidden_states, indexes, rope_tables)

    monkeypatch.setattr(Qwen3Attention, "_resolve_rope_tables", spy)
    with torch.no_grad():
        model(
            inputs_embeds=torch.randn(1, 5, _HIDDEN_DIM),
            image_gen_indicators=torch.ones(1, 5, dtype=torch.bool),
            indexes=torch.zeros(3, 5, dtype=torch.long),
            attention_mask={"full_attention": None},
        )

    assert len(seen) == _LAYERS, f"attention ran {len(seen)} times, want {_LAYERS}"
    assert all(seen), f"layers that had to rebuild their own tables: {seen}"


def test_sensenova_u1_rope_sharing_does_not_change_output(monkeypatch):
    """The end-to-end claim: sharing must not move the output by one bit.

    Ran with sharing and with sharing bypassed (each layer rebuilding its own
    tables, as before the change) on identical inputs.
    """
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3Model,
    )

    torch.manual_seed(0)
    model = Qwen3Model(_tiny_dense_config()).eval()
    embeds = torch.randn(1, 5, _HIDDEN_DIM)
    indicators = torch.ones(1, 5, dtype=torch.bool)
    indexes = torch.zeros(3, 5, dtype=torch.long)

    def run():
        with torch.no_grad():
            return model(
                inputs_embeds=embeds,
                image_gen_indicators=indicators,
                indexes=indexes,
                attention_mask={"full_attention": None},
            ).last_hidden_state

    shared = run()

    original = Qwen3Attention._resolve_rope_tables
    monkeypatch.setattr(
        Qwen3Attention,
        "_resolve_rope_tables",
        lambda self, hidden_states, idx, tables: original(
            self, hidden_states, idx, None
        ),
    )
    per_layer = run()

    assert torch.equal(shared, per_layer)


def test_sensenova_u1_rope_tables_reach_the_understanding_path(monkeypatch):
    """`forward_und` has its own pop site, so the gen-path cases do not cover it.

    All-zeros indicators select the understanding path used by the text prefix.
    """
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3Model,
    )

    model = Qwen3Model(_tiny_dense_config()).eval()
    seen = []
    original = Qwen3Attention._resolve_rope_tables

    def spy(self, hidden_states, indexes, rope_tables):
        seen.append(rope_tables is not None)
        return original(self, hidden_states, indexes, rope_tables)

    monkeypatch.setattr(Qwen3Attention, "_resolve_rope_tables", spy)
    with torch.no_grad():
        model(
            inputs_embeds=torch.randn(1, 5, _HIDDEN_DIM),
            image_gen_indicators=torch.zeros(1, 5, dtype=torch.bool),
            indexes=torch.zeros(3, 5, dtype=torch.long),
            attention_mask={"full_attention": None},
        )

    assert len(seen) == _LAYERS, f"attention ran {len(seen)} times, want {_LAYERS}"
    assert all(seen), f"layers that had to rebuild their own tables: {seen}"
