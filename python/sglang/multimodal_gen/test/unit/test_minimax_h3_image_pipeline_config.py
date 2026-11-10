# SPDX-License-Identifier: Apache-2.0
"""Image server-arg policy and declared-task admission on CPU; no model loading."""

import io
import json
from dataclasses import MISSING, fields
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_image import (
    MiniMaxH3ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3_image import (
    MiniMaxH3ImageSamplingParams,
)
from sglang.multimodal_gen.configs.task_type import DataType, ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as api_utils
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3ImagePipeline,
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY as PLAN,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def normalized_args(**overrides):
    # ServerArgs dataclass defaults without __post_init__ (no model load, no GPU
    # probe), then the sglang backend, the image pipeline config and every
    # parallel degree at 1.
    args = object.__new__(ServerArgs)
    for f in fields(ServerArgs):
        if f.default is not MISSING:
            setattr(args, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(args, f.name, f.default_factory())
    args.backend = "sglang"
    # hardware-dependent auto policy is covered by the ServerArgs tests
    args.performance_mode = "manual"
    # ServerArgs._adjust_attention_backend snapshots explicit requests first.
    args._requested_component_attention_backends = {}
    args.pipeline_config = MiniMaxH3ImagePipelineConfig()
    for name in (
        "tp_size",
        "sp_degree",
        "ulysses_degree",
        "ring_degree",
        "kv_gather_degree",
        "cfg_parallel_degree",
    ):
        setattr(args, name, 1)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_image_config_accepts_release_precisions_and_requests_fa():
    args = normalized_args(
        component_precisions={
            "transformer": "bf16",
            "text_encoder": "bf16",
            "video_vae": "fp32",
        },
        component_residency={"transformer": "component-offload"},
    )
    MiniMaxH3ImagePipelineConfig().validate_server_args(args)
    assert args.component_attention_backends["transformer"] == "fa"


def test_image_deployment_has_no_cfg_parallel():
    deployment = MiniMaxH3ImagePipelineConfig().get_model_deployment_config()
    assert not deployment.supports_cfg_parallel
    assert not deployment.auto_enable_cfg_parallel


@pytest.mark.parametrize(
    "override",
    [
        {"num_gpus": 2, "sp_degree": 2, "ulysses_degree": 2},
        {"num_gpus": 2, "tp_size": 2},
        {"num_gpus": 4, "sp_degree": 2, "ring_degree": 2},
        {"num_gpus": 4, "sp_degree": 4, "ulysses_degree": 4},
        {"num_gpus": 4, "tp_size": 2, "sp_degree": 2},
        # sp_degree=2 with no explicit split: the auto K/V-gather alias.
        {
            "num_gpus": 2,
            "sp_degree": 2,
            "kv_gather_degree": 2,
            "ulysses_degree": 2,
            "sp_split_auto": True,
        },
    ],
)
def test_image_config_accepts_tp_and_sp(override):
    MiniMaxH3ImagePipelineConfig().validate_server_args(normalized_args(**override))


@pytest.mark.parametrize(
    "override,match",
    [
        ({"num_gpus": 2, "dp_size": 2}, "dp_size=1"),
        ({"num_gpus": 2, "nnodes": 2}, "nnodes=1"),
        (
            {"num_gpus": 2, "sp_degree": 2, "kv_gather_degree": 2, "ulysses_degree": 2},
            "K/V-gather",
        ),
    ],
)
def test_image_config_rejects_unsupported_parallel_layouts(override, match):
    with pytest.raises(ValueError, match=match):
        MiniMaxH3ImagePipelineConfig().validate_server_args(normalized_args(**override))


@pytest.mark.parametrize(
    "override",
    [
        {"use_fsdp_inference": True},
        {"quantization": "nvfp4"},
        {"component_quantizations": {"video_vae": "fp8"}},
        {"lora_path": "adapter.safetensors"},
        {"component_precisions": {"video_vae": "bf16"}},
        {"component_precisions": {"audio_vae": "fp32"}},
    ],
)
def test_image_config_rejects_unsupported_server_args(override):
    with pytest.raises(ValueError):
        MiniMaxH3ImagePipelineConfig().validate_server_args(normalized_args(**override))


@pytest.mark.parametrize(
    "override",
    [{"dit_precision": "fp32"}, {"vae_decode_precision": "fp16"}, {"flow_shift": 12.0}],
)
def test_image_config_rejects_numeric_overrides(override):
    with pytest.raises(ValueError):
        MiniMaxH3ImagePipelineConfig(**override).validate_server_args(normalized_args())
    with pytest.raises(ValueError, match="Cache-DiT"):
        MiniMaxH3ImagePipelineConfig().get_cache_config()


@pytest.mark.parametrize("consumer", ["qwen", "vae"])
@pytest.mark.parametrize(
    "override,match",
    [
        ({"min": 1000}, "multiple of 32"),
        ({"min": 1024.0}, "multiple of 32"),
        ({"max": 0}, "multiple of 32"),
        ({"min": 4096}, "must not exceed"),
    ],
)
def test_edit_reference_bounds_must_be_ordered_multiples_of_32(
    consumer, override, match
):
    fields = {
        f"edit_reference_{consumer}_{bound}_short_edge": value
        for bound, value in override.items()
    }
    config = MiniMaxH3ImagePipelineConfig(**fields)
    with pytest.raises(ValueError, match=match):
        config.edit_reference_short_edge_bounds(consumer)
    with pytest.raises(ValueError, match=match):
        config.check_pipeline_config()


def test_edit_reference_bounds_defaults_and_unknown_consumer():
    config = MiniMaxH3ImagePipelineConfig()
    assert config.edit_reference_short_edge_bounds("qwen") == (1024, 1024)
    assert config.edit_reference_short_edge_bounds("vae") == (1024, 2048)
    with pytest.raises(ValueError, match="consumer"):
        config.edit_reference_short_edge_bounds("dit")


@pytest.mark.parametrize("value", [1023, 4096 * 4096 + 1, 4194304.0, True, "4MP"])
def test_min_generation_area_config_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="min_generation_area"):
        MiniMaxH3ImagePipelineConfig(min_generation_area=value).check_pipeline_config()


@pytest.mark.parametrize("value", [None, 1024, 2048 * 2048, 4096 * 4096])
def test_min_generation_area_config_accepts_null_and_the_range(value):
    MiniMaxH3ImagePipelineConfig(min_generation_area=value).check_pipeline_config()
    assert MiniMaxH3ImagePipelineConfig().min_generation_area == 2048 * 2048


# --------------------------------------------------------------------- registry
@pytest.mark.parametrize(
    "model_path",
    [
        "MiniMaxAI/MiniMax-H3-Image",
        "/models/MiniMax-H3-Image",
        "/cache/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/revision",
    ],
)
def test_registry_resolves_image_release_paths(model_path, monkeypatch):
    # "minimax-h3" is a prefix of every one of these paths; none may resolve
    # to the MiniMax-H3 video pipeline or its configs.
    monkeypatch.setattr(
        registry,
        "maybe_download_model_index",
        lambda _: {"_class_name": "MiniMaxH3ImagePipeline"},
    )
    assert registry.get_non_diffusers_pipeline_name(model_path) is None
    info = registry.get_model_info(model_path, backend="sglang")
    assert info.pipeline_cls is MiniMaxH3ImagePipeline
    assert info.pipeline_config_cls is MiniMaxH3ImagePipelineConfig
    assert info.sampling_param_cls is MiniMaxH3ImageSamplingParams


def test_registry_keeps_minimax_h3_cache_paths(monkeypatch):
    monkeypatch.setattr(
        registry,
        "maybe_download_model_index",
        lambda _: pytest.fail("MiniMax-H3 is resolved without model_index.json"),
    )
    path = "/cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/revision"
    assert registry.get_model_info(path, backend="sglang").pipeline_cls is (
        MiniMaxH3Pipeline
    )


# --------------------------------------------------------------- declared tasks
# Image releases are PDD checkpoints; pdd_sigmas=None writes one without heads.
PDD_GRID = [1.0, 0.75, 0.25, 0.0]


def write_model_dir(root, tasks, partition="image", pdd_sigmas=PDD_GRID):
    root.mkdir()
    # Image releases name their VAE `vae`; video partitions keep `video_vae`.
    vae = "vae" if partition == "image" else "video_vae"
    for name in ("processor", "text_encoder", "tokenizer", "transformer", vae):
        (root / name).mkdir()
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3DiTModel"})
    )
    pdd = {}
    if pdd_sigmas is not None:
        # The HTTP process reads only pdd_config.json; the heads load later.
        pdd = {"pdd_fused_heads": "pdd_fused_heads.safetensors"}
        (root / "pdd_config.json").write_text(
            json.dumps(
                {"num_inference_steps": len(pdd_sigmas) - 1, "sigmas": pdd_sigmas}
            )
        )
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ImagePipeline",
                "_diffusers_version": "0.37.0",
                "_minimax_h3": {
                    "schema_version": 1,
                    "partition": partition,
                    "tasks": tasks,
                    "task_aliases": {},
                    **(
                        {"sigma_shift_scales": {"video": 12.0, "audio": 3.0}}
                        if partition != "image"
                        else {}
                    ),
                    **pdd,
                },
                "scheduler": None,
                "processor": ["transformers", "Qwen3VLProcessor"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "transformer": ["diffusers", "MiniMaxH3DiTModel"],
                vae: ["diffusers", "AutoencoderKL"],
            }
        )
    )
    return str(root)


@pytest.mark.parametrize(
    "tasks,expected",
    [
        (["t2i", "edit"], (ModelTaskType.T2I, ModelTaskType.I2I)),
        (["t2i"], (ModelTaskType.T2I,)),
        (["edit"], (ModelTaskType.I2I,)),
    ],
)
def test_supported_task_types_follow_the_declared_release(tmp_path, tasks, expected):
    config = MiniMaxH3ImagePipelineConfig(
        model_path=write_model_dir(tmp_path / "release", tasks)
    )
    assert config.get_supported_task_types() == expected
    # No model path (bare contract configs): the class capability.
    assert MiniMaxH3ImagePipelineConfig().get_supported_task_types() == (
        ModelTaskType.T2I,
        ModelTaskType.I2I,
    )


def test_supported_task_types_reject_video_releases_and_missing_dirs(tmp_path):
    video = MiniMaxH3ImagePipelineConfig(
        model_path=write_model_dir(
            tmp_path / "video", ["ref2va"], partition="ref2va", pdd_sigmas=None
        )
    )
    with pytest.raises(ValueError, match="not an image release"):
        video.get_supported_task_types()
    with pytest.raises(ValueError, match="model_index.json"):
        MiniMaxH3ImagePipelineConfig(
            model_path=str(tmp_path / "missing")
        ).get_supported_task_types()


def test_resolve_task_type_pins_task_to_references(tmp_path):
    config = MiniMaxH3ImagePipelineConfig(
        model_path=write_model_dir(tmp_path / "release", ["t2i"])
    )
    assert config.resolve_task_type(data_type=DataType.IMAGE) is ModelTaskType.T2I
    assert config.resolve_task_type(ModelTaskType.T2I) is ModelTaskType.T2I
    with pytest.raises(ValueError, match="'edit' is not served"):
        config.resolve_task_type(data_type=DataType.IMAGE, has_image=True)
    with pytest.raises(ValueError, match="does not match"):
        config.resolve_task_type(ModelTaskType.I2I, has_image=False)
    with pytest.raises(ValueError, match="IMAGE output"):
        config.resolve_task_type(data_type=DataType.VIDEO)
    with pytest.raises(ValueError, match="image references only"):
        config.resolve_task_type(has_video=True)


@pytest.fixture
def image_router(monkeypatch, tmp_path):
    import sglang.multimodal_gen.registry as registry

    def make(tasks, pdd_sigmas=PDD_GRID):
        config = MiniMaxH3ImagePipelineConfig(
            model_path=write_model_dir(
                tmp_path / "release", tasks, pdd_sigmas=pdd_sigmas
            )
        )
        server = SimpleNamespace(
            model_path=config.model_path,
            pipeline_class_name="MiniMaxH3ImagePipeline",
            pipeline_config=config,
            backend="sglang",
            model_id=None,
            output_path=str(tmp_path),
            input_save_path=str(tmp_path),
            comfyui_mode=False,
            attention_backend_config=None,
            enable_trace=False,
            batching_max_size=1,
        )
        admitted = []

        async def generate(client, req, **kwargs):
            admitted.append(req)
            path = req.output_file_path()
            Image.new("RGB", (16, 16)).save(path)
            with open(path, "rb") as f:
                return [path], OutputBatch(output_image_bytes=[f.read()])

        async def upload(paths):
            return [None] * len(paths)

        # Mock only class selection; the real router, factory and config run.
        monkeypatch.setattr(
            registry,
            "get_pipeline_config_classes",
            lambda name: (MiniMaxH3ImagePipelineConfig, MiniMaxH3ImageSamplingParams),
        )
        monkeypatch.setattr(api_utils, "get_global_server_args", lambda: server)
        monkeypatch.setattr(image_api, "get_global_server_args", lambda: server)
        monkeypatch.setattr(image_api, "process_generation_batch", generate)
        monkeypatch.setattr(image_api.cloud_storage, "upload_all_and_cleanup", upload)
        app = FastAPI()
        app.include_router(image_api.router)
        return TestClient(app), admitted

    return make


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="PNG")
    return buffer.getvalue()


def _post(client, task):
    fields = {"prompt": "p", "size": "64x32"}
    if task == "t2i":
        return client.post("/v1/images/generations", json=fields)
    return client.post(
        "/v1/images/edits",
        data=fields,
        files=[("image[]", ("ref.png", _png(), "image/png"))],
    )


@pytest.mark.parametrize("declared", [["t2i"], ["edit"], ["t2i", "edit"]])
@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_http_rejects_undeclared_tasks_before_scheduling(image_router, declared, task):
    client, admitted = image_router(declared)
    with client:
        response = _post(client, task)
    if task in declared:
        assert response.status_code == 200, response.text
        assert [req.extra[PLAN].task for req in admitted] == [task]
    else:
        assert response.status_code == 400, response.text
        assert f"task '{task}' is not served" in response.text
        assert not admitted


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_http_pdd_release_runs_exactly_its_head_count(image_router, task):
    client, admitted = image_router(["t2i", "edit"])

    def post(**overrides):
        fields = {"prompt": "p", "size": "64x32", **overrides}
        if task == "t2i":
            return client.post("/v1/images/generations", json=fields)
        return client.post(
            "/v1/images/edits",
            data=fields,
            files=[("image[]", ("ref.png", _png(), "image/png"))],
        )

    with client:
        default = post()
        explicit = post(num_inference_steps=3)
        other = post(num_inference_steps=51)
    assert default.status_code == 200, default.text
    assert explicit.status_code == 200, explicit.text
    assert [req.num_inference_steps for req in admitted] == [3, 3]
    assert other.status_code == 400, other.text
    assert "num_inference_steps must be 3" in other.text


def test_image_release_without_pdd_heads_is_rejected(tmp_path):
    config = MiniMaxH3ImagePipelineConfig(
        model_path=write_model_dir(tmp_path / "release", ["t2i"], pdd_sigmas=None)
    )
    with pytest.raises(ValueError, match="PDD releases only"):
        config.release_pdd_steps()
