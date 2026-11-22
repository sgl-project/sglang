from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.krea2 import Krea2PipelineConfig
from sglang.multimodal_gen.configs.sample.krea2 import Krea2SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.warmup_request_builder import build_warmup_reqs


def _server_args(pipeline_config):
    return SimpleNamespace(
        model_path="krea/Krea-2-Turbo",
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


def _build_image_sampling(server_args, fields):
    with (
        patch.object(openai_utils, "get_global_server_args", return_value=server_args),
        patch.object(
            SamplingParams, "from_pretrained", return_value=Krea2SamplingParams()
        ),
    ):
        return openai_utils.build_sampling_params(
            "size",
            prompt="a cat",
            request_data_type=DataType.IMAGE,
            num_frames=1,
            **fields,
        )


def _assert_patchifiable(pipeline_config, width, height):
    vsf = pipeline_config.get_vae_scale_factor()
    patch_size = pipeline_config.dit_config.arch_config.patch
    assert (height // vsf) % patch_size == 0
    assert (width // vsf) % patch_size == 0


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"size": "1920x1080"}, (1920, 1072)),
        ({"size": "854x480"}, (848, 480)),
        ({"size": "1000x1000"}, (992, 992)),
        ({"width": 1080, "height": 1920}, (1072, 1920)),
        ({"size": "1280x720"}, (1280, 720)),
        ({"size": "1024x1024"}, (1024, 1024)),
    ],
)
def test_http_size_aligns_to_the_patch_grid(fields, expected):
    pipeline_config = Krea2PipelineConfig()
    sampling = _build_image_sampling(_server_args(pipeline_config), fields)
    assert (sampling.width, sampling.height) == expected
    _assert_patchifiable(pipeline_config, sampling.width, sampling.height)


def test_warmup_resolution_aligns_to_the_patch_grid():
    server_args = _server_args(Krea2PipelineConfig())
    with patch(
        "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
        return_value=Krea2SamplingParams(),
    ):
        reqs = build_warmup_reqs(
            server_args,
            warmup_resolutions=["1920x1080"],
            warmup_input_path=None,
            server_based_warmup=True,
        )
    assert [(r.width, r.height) for r in reqs] == [(1920, 1072)]
