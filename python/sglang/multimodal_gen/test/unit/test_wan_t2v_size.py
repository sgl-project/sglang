from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.diffusers_generic import (
    DiffusersT2VPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_T2V_A14B_Config,
    WanI2V480PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_T2V_A14B_SamplingParam,
    WanI2V_14B_480P_SamplingParam,
    WanT2V_1_3B_SamplingParams,
    WanT2V_14B_SamplingParams,
)
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.warmup_request_builder import build_warmup_reqs


def _server_args(pipeline_config):
    return SimpleNamespace(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        model_id=None,
        backend="sglang",
        performance_mode="manual",
        pipeline_class_name=None,
        pipeline_config=pipeline_config,
        output_path=None,
        comfyui_mode=True,
        enable_cfg_parallel=False,
        enable_breakable_cuda_graph=False,
        enable_torch_compile=False,
        enable_trace=False,
        attention_backend_config={},
        num_gpus=1,
        warmup_steps=1,
        warmup_num_frames=None,
    )


def _build_video_sampling(server_args, sampling_cls, fields):
    with (
        patch.object(openai_utils, "get_global_server_args", return_value=server_args),
        patch.object(SamplingParams, "from_pretrained", return_value=sampling_cls()),
    ):
        return openai_utils.build_sampling_params(
            "size",
            prompt="a cat",
            request_data_type=DataType.VIDEO,
            num_frames=17,
            **fields,
        )


@pytest.mark.parametrize(
    "pipeline_config_cls,sampling_cls",
    [
        (WanT2V480PConfig, WanT2V_1_3B_SamplingParams),
        (WanT2V720PConfig, WanT2V_14B_SamplingParams),
        (Wan2_2_T2V_A14B_Config, Wan2_2_T2V_A14B_SamplingParam),
    ],
)
@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"size": "1920x1080"}, (1920, 1072)),
        ({"size": "640x360"}, (640, 352)),
        ({"size": "1000x1000"}, (992, 992)),
        ({"width": 1080, "height": 1920}, (1072, 1920)),
        ({"size": "854x480"}, (848, 480)),
        ({"size": "1280x720"}, (1280, 720)),
        ({"size": "832x480"}, (832, 480)),
    ],
)
def test_http_size_aligns_to_the_patch_grid(
    pipeline_config_cls, sampling_cls, fields, expected
):
    server_args = _server_args(pipeline_config_cls())
    sampling = _build_video_sampling(server_args, sampling_cls, fields)
    assert (sampling.width, sampling.height) == expected
    # the DiT splits the latent into 2x2 patches
    _, _, _, height, width = server_args.pipeline_config.prepare_latent_shape(
        Req(sampling_params=sampling), 1, 5
    )
    assert (height % 2, width % 2) == (0, 0)


def test_i2v_config_keeps_requested_size():
    server_args = _server_args(WanI2V480PConfig())
    sampling = _build_video_sampling(
        server_args,
        WanI2V_14B_480P_SamplingParam,
        {"size": "1000x1000", "image_path": ["cat.png"]},
    )
    assert (sampling.width, sampling.height) == (1000, 1000)


def test_diffusers_backend_keeps_requested_size():
    server_args = _server_args(DiffusersT2VPipelineConfig())
    server_args.backend = "diffusers"
    sampling = _build_video_sampling(
        server_args, WanT2V_1_3B_SamplingParams, {"size": "1920x1080"}
    )
    assert (sampling.width, sampling.height) == (1920, 1080)


def test_warmup_resolution_aligns_to_the_patch_grid():
    server_args = _server_args(WanT2V480PConfig())
    with patch(
        "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
        return_value=WanT2V_1_3B_SamplingParams(),
    ):
        reqs = build_warmup_reqs(
            server_args,
            warmup_resolutions=["1920x1080"],
            warmup_input_path=None,
            server_based_warmup=True,
        )
    assert [(r.width, r.height) for r in reqs] == [(1920, 1072)]
