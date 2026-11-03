# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.dm05 import (
    DM05LiberoPipelineConfig,
    DM05MemoryPipelineConfig,
    DM05PipelineConfig,
    DM05Robotwin2PipelineConfig,
    dm05_model_id,
    is_dm05_model,
)
from sglang.multimodal_gen.configs.sample.dm05 import DM05SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.action import protocol
from sglang.multimodal_gen.runtime.models.vlas.dm05_policy import DM05PolicyModel
from sglang.multimodal_gen.runtime.pipelines.dm05 import DM05ObservationPreprocessStage
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
    RequestValidationError,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
)
from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch


def test_dm05_profiles_preserve_public_deployment_geometry():
    base = DM05PipelineConfig()
    libero = DM05LiberoPipelineConfig()
    twin = DM05Robotwin2PipelineConfig()
    memory = DM05MemoryPipelineConfig()
    assert (base.action_horizon, base.action_dim, base.output_action_dim) == (
        50,
        32,
        14,
    )
    assert len(base.image_keys) == 3 and base.use_absolute_action
    assert (libero.action_horizon, libero.output_action_dim, libero.state_dim) == (
        10,
        7,
        8,
    )
    assert len(libero.image_keys) == 2 and not libero.add_state
    assert libero.state_desc == ("joint",) * 6 + ("gripper",) * 2
    assert (twin.action_horizon, twin.output_action_dim, twin.max_token_len) == (
        50,
        14,
        1152,
    )
    assert not twin.use_absolute_action and not twin.clip_to_bounds
    assert memory.is_history and memory.max_history_images == 20
    assert memory.default_robot_type == "Dual ARX5"
    assert (
        memory.materialize_dtype == "fp32" and memory.precision_policy == "fp32_mixed"
    )
    assert not memory.use_absolute_action and not memory.clip_to_bounds
    for config in (base, libero, twin, memory):
        metadata = config.action_metadata(
            SimpleNamespace(served_model_name="dm05", model_path="local")
        )
        assert metadata["output"]["action_dim"] == config.output_action_dim


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_horizon": 0},
        {"action_horizon": True},
        {"action_dim": -1},
        {"output_action_dim": 33},
        {"max_token_len": 0},
        {"n_state_bins": 1},
        {"max_history_images": -1},
        {"action_cuda_graph_max_entries": -1},
        {"image_keys": ("head", "head", "right")},
        {"image_prompts": ("Head",)},
        {"materialize_dtype": "fp8"},
        {"precision_policy": "fp8"},
        {"precision_policy": "fp32_mixed"},
        {"llm_attn_implementation": "unknown"},
        {"enable_prefix_cuda_graph": True},
    ],
)
def test_dm05_rejects_invalid_deployment_contracts(kwargs):
    with pytest.raises(ValueError):
        DM05PipelineConfig(**kwargs)


def test_dm05_local_discovery_uses_config_and_preserves_recipe_override(tmp_path):
    arbitrary = tmp_path / "checkpoint-200000"
    arbitrary.mkdir()
    (arbitrary / "config.json").write_text(json.dumps({"model_type": "dm05"}))
    assert is_dm05_model(str(arbitrary))
    assert dm05_model_id(str(arbitrary)) == "Dexmal/DM05"
    misleading = tmp_path / "DM05-libero"
    misleading.mkdir()
    (misleading / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
    assert not is_dm05_model(str(misleading))
    assert is_dm05_model("Dexmal/DM05-libero")
    config = DM05LiberoPipelineConfig()
    config.apply_checkpoint_config(
        str(arbitrary), {"model_type": "dm05", "chunk_size": 50, "action_dim": 32}
    )
    assert config.action_horizon == 10
    with pytest.raises(ValueError, match="action_dim"):
        config.apply_checkpoint_config(
            str(arbitrary), {"model_type": "dm05", "action_dim": 14}
        )


def test_dm05_hf_cache_path_selects_the_known_recipe(tmp_path):
    path = tmp_path / "models--Dexmal--DM05-libero" / "snapshots" / "0123456789"
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type": "dm05"}')
    assert is_dm05_model(str(path))
    assert dm05_model_id(str(path)) == "Dexmal/DM05-libero"


def test_dm05_metadata_reports_its_own_history_and_precision():
    config = DM05MemoryPipelineConfig()
    metadata = config.action_metadata(
        SimpleNamespace(served_model_name="dm05", model_path="local")
    )
    assert metadata["policy_family"] == "dm05"
    assert (
        metadata["input"]["history"] and metadata["input"]["max_history_images"] == 20
    )
    assert metadata["runtime"]["precision_policy"] == "fp32_mixed"
    assert not metadata["runtime"]["cuda_graph"]["prefix_enabled"]
    assert metadata["output"]["padded_action_dim"] == 32


def test_dm05_sampling_uses_deployment_geometry_and_retains_history():
    args = SimpleNamespace(
        pipeline_config=DM05LiberoPipelineConfig(), output_path=None, comfyui_mode=False
    )
    sample = DM05SamplingParams(
        prompt="pick", history_images=[None], observation={"robot_type": "Franka"}
    )
    sample._adjust(args)
    assert (sample.action_horizon, sample.action_dim) == (10, 32)
    obs = sample.build_request_extra()["vla"]["observation"]
    assert obs["history_images"] == [None] and obs["robot_type"] == "Franka"
    with pytest.raises(ValueError, match="loaded DM05 deployment"):
        DM05SamplingParams(action_horizon=50)._adjust(args)
    with pytest.raises(ValueError, match="one action candidate"):
        DM05SamplingParams(num_outputs_per_prompt=2)


def test_dm05_request_seed_is_independent_and_explicit_noise_takes_precedence():
    class ZeroVelocityCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(4, 4)

        def velocity(self, prefix, values, time):
            return torch.zeros_like(values, dtype=torch.float32)

    config = SimpleNamespace(
        action_horizon=3,
        action_dim=4,
        default_num_inference_steps=2,
        precision_policy="bf16_mixed",
        enable_action_cuda_graph=False,
        action_cuda_graph_max_entries=0,
        generator_device="cpu",
    )
    policy = DM05PolicyModel(config, ZeroVelocityCore(), device="cpu")

    def preprocess(raw):
        return VLAObservationBatch(
            prompt=["pick"],
            images={},
            image_masks={},
            state=None,
            noise=raw.get("noise"),
            tokens=torch.tensor([[2, 3]]),
            token_masks=torch.ones(1, 2, dtype=torch.bool),
            batch_size=1,
            metadata={"dm05": {"action_mask": torch.ones(1, 1, 4)}},
        )

    stage = DM05ObservationPreprocessStage(preprocess, policy)
    denoise = VLAActionDenoisingStage(policy)
    args = SimpleNamespace(pipeline_config=config)
    before = torch.random.get_rng_state().clone()
    results = []
    for seed in (42, 99, 42):
        sampling = DM05SamplingParams(
            seed=seed, action_horizon=3, action_dim=4, generator_device="cpu"
        )
        request = Req(sampling_params=sampling)
        sampling.apply_request_extra(request)
        stage.forward(request, args)
        assert request.generator is None
        request.extra["vla"]["prefix_context"] = None
        denoise.forward(request, args)
        results.append(request.extra["vla"]["actions"].clone())
    assert torch.equal(results[0], results[2])
    assert not torch.equal(results[0], results[1])
    assert torch.equal(before, torch.random.get_rng_state())
    noise = torch.ones(1, 3, 4)
    sampling = DM05SamplingParams(seed=99, action_horizon=3, action_dim=4, noise=noise)
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)
    stage.forward(request, args)
    assert request.extra["vla"]["observation_batch"].noise is noise
    assert torch.equal(before, torch.random.get_rng_state())


def test_dm05_warmup_does_not_require_client_observation():
    def reject(_):
        pytest.fail("Warmup must not parse an absent client observation")

    request = Req(sampling_params=DM05SamplingParams())
    request.is_warmup = True
    stage = DM05ObservationPreprocessStage(reject, None)
    assert stage.forward(request, SimpleNamespace()) is request


def test_action_transport_preserves_dm05_fields_and_decodes_history(monkeypatch):
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    image = io.BytesIO()
    Image.fromarray(pixels).save(image, format="PNG")
    encoded = {"b64_json": base64.b64encode(image.getvalue()).decode()}
    payload = {
        "input": {
            "task": "pick",
            "images": {"head": encoded, "left_wrist": encoded},
            "state": [0.0] * 8,
            "history_images": [encoded],
            "robot_type": "Franka",
            "control_mode": "joint",
            "speed": "0.5",
            "action_mask": {"values": [[[1.0] * 7 + [0.0] * 25]], "dtype": "float32"},
        }
    }
    monkeypatch.setattr(
        protocol, "_resolve_action_sampling_params_cls", lambda _: DM05SamplingParams
    )
    args = SimpleNamespace(
        pipeline_config=DM05LiberoPipelineConfig(), output_path=None, comfyui_mode=False
    )
    sample = protocol.build_action_sampling_params(payload, args)
    obs = sample.build_request_extra()["vla"]["observation"]
    np.testing.assert_array_equal(np.asarray(obs["history_images"][0]), pixels)
    assert obs["robot_type"] == "Franka" and obs["control_mode"] == "joint"
    assert obs["action_mask"].shape == (1, 1, 32)
    assert sample.action_horizon == 10 and sample.action_dim == 32


def test_grouped_action_transform_sees_full_padded_actions_before_materialization():
    prefix = object()
    observation = object()
    seen = []
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    class Policy:
        def sample_actions(self, *args, **kwargs):
            return values

        def action_parallel_info(self, *args):
            return {"world_size": 1}

    def transform(actions, obs, config):
        assert actions.shape == (2, 3, 4)
        assert obs is observation
        seen.append(actions.clone())
        return actions[..., :2] + actions[..., 3:4]

    observation = SimpleNamespace(noise=None)
    batches = [
        SimpleNamespace(
            extra={
                "vla": {
                    "prefix_context_group": prefix,
                    "observation_group": observation,
                    "options": {"output_format": "list"},
                }
            },
            num_inference_steps=1,
        )
        for _ in range(2)
    ]
    stage = VLAActionDenoisingStage(Policy(), output_transform=transform)
    result = stage.run_grouped_requests(
        batches, SimpleNamespace(pipeline_config=SimpleNamespace(output_action_dim=2))
    )
    assert len(seen) == 1
    for index, batch in enumerate(result):
        expected = (values[index, :, :2] + values[index, :, 3:4]).tolist()
        assert batch.extra["vla"]["actions_output"] == expected


@pytest.mark.parametrize(
    "name", ["action_horizon", "action_dim", "num_inference_steps"]
)
@pytest.mark.parametrize("value", [True, False, 1.5, "10", None, 0, -1])
def test_dm05_transport_rejects_coerced_geometry(monkeypatch, name, value):
    monkeypatch.setattr(
        protocol, "_resolve_action_sampling_params_cls", lambda _: DM05SamplingParams
    )
    args = SimpleNamespace(
        pipeline_config=DM05LiberoPipelineConfig(), output_path=None, comfyui_mode=False
    )
    with pytest.raises(ValueError, match="positive integer"):
        protocol.build_action_sampling_params({"parameters": {name: value}}, args)


@pytest.mark.parametrize("value", [True, False, 1.5, "11", None, -1, [], [True]])
def test_dm05_transport_rejects_invalid_seeds(monkeypatch, value):
    monkeypatch.setattr(
        protocol, "_resolve_action_sampling_params_cls", lambda _: DM05SamplingParams
    )
    args = SimpleNamespace(
        pipeline_config=DM05LiberoPipelineConfig(), output_path=None, comfyui_mode=False
    )
    with pytest.raises(ValueError, match="seed"):
        protocol.build_action_sampling_params({"parameters": {"seed": value}}, args)


def test_dm05_python_sampling_rejects_boolean_steps():
    with pytest.raises(ValueError, match="positive integer"):
        DM05SamplingParams(num_inference_steps=True)


def test_dm05_preprocessing_marks_only_observation_validation_errors():
    sampling = DM05SamplingParams()
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)

    def invalid(_):
        raise ValueError("invalid state geometry")

    with pytest.raises(RequestValidationError, match="invalid state geometry"):
        DM05ObservationPreprocessStage(invalid, None).forward(
            request, SimpleNamespace()
        )

    def broken(_):
        raise RuntimeError("internal model failure")

    with pytest.raises(RuntimeError, match="internal model failure"):
        DM05ObservationPreprocessStage(broken, None).forward(request, SimpleNamespace())


@pytest.mark.parametrize("pixel", ["x", None])
def test_dm05_invalid_pixels_are_request_validation_errors(pixel):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dm05_preprocess import (
        preprocess_dm05_image,
    )

    sampling = DM05SamplingParams(observation={"image": [[[pixel] * 3]]})
    request = Req(sampling_params=sampling)
    sampling.apply_request_extra(request)

    def preprocess(raw):
        return preprocess_dm05_image(raw["image"])

    with pytest.raises(RequestValidationError, match="numeric"):
        DM05ObservationPreprocessStage(preprocess, None).forward(
            request, SimpleNamespace()
        )


@pytest.mark.parametrize(
    "error_code,expected", [("invalid_request", ValueError), (None, RuntimeError)]
)
def test_action_transport_preserves_typed_worker_error(
    monkeypatch, error_code, expected
):
    output = pickle.loads(
        pickle.dumps(OutputBatch(error="test failure", error_code=error_code))
    )

    async def forward(_):
        return output

    monkeypatch.setattr(protocol, "build_action_sampling_params", lambda *_: object())
    monkeypatch.setattr(protocol, "prepare_request", lambda *_: object())
    monkeypatch.setattr(protocol.async_scheduler_client, "forward", forward)
    with pytest.raises(expected, match="test failure"):
        asyncio.run(protocol.infer_action({}, SimpleNamespace()))


def test_dm05_invalid_group_isolates_other_requests():
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

    requests = [object(), object()]
    calls = []

    def execute(group):
        calls.append(group)
        if len(group) == 2 or group[0] is requests[1]:
            return OutputBatch(
                error="invalid observation", error_code="invalid_request"
            )
        return OutputBatch(output=[{"actions": [[1.0]]}])

    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(pipeline_config=DM05LiberoPipelineConfig()),
        worker=SimpleNamespace(execute_forward=execute),
    )
    scheduler._execute_generation_sequential = lambda group: (
        Scheduler._execute_generation_sequential(scheduler, group)
    )
    outputs = Scheduler._execute_generation_grouped(scheduler, requests)
    assert calls == [requests, [requests[0]], [requests[1]]]
    assert outputs[0].output == [{"actions": [[1.0]]}]
    assert outputs[1].error_code == "invalid_request"


@pytest.mark.parametrize("error_code,status", [("invalid_request", 400), (None, 500)])
def test_action_http_status_preserves_internal_errors(monkeypatch, error_code, status):
    import httpx
    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.entrypoints.action.api import router

    async def forward(_):
        return OutputBatch(error="test failure", error_code=error_code)

    monkeypatch.setattr(protocol, "build_action_sampling_params", lambda *_: object())
    monkeypatch.setattr(protocol, "prepare_request", lambda *_: object())
    monkeypatch.setattr(protocol.async_scheduler_client, "forward", forward)
    app = FastAPI()
    app.state.server_args = SimpleNamespace()
    app.include_router(router)

    async def request():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post("/v1/actions/generations", json={})

    assert asyncio.run(request()).status_code == status
