from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.diffusers_generic import (
    DiffusersTI2VPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.wan import Wan2_2_TI2V_5B_SamplingParam
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.warmup_request_builder import build_warmup_reqs


def _ti2v_server_args():
    return SimpleNamespace(
        model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_id=None,
        backend="sglang",
        performance_mode="manual",
        pipeline_class_name=None,
        pipeline_config=Wan2_2_TI2V_5B_Config(),
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


def _build_video_sampling(server_args, fields):
    with (
        patch.object(openai_utils, "get_global_server_args", return_value=server_args),
        patch.object(
            SamplingParams,
            "from_pretrained",
            return_value=Wan2_2_TI2V_5B_SamplingParam(),
        ),
    ):
        return openai_utils.build_sampling_params(
            "size",
            prompt="a cat",
            request_data_type=DataType.VIDEO,
            num_frames=17,
            **fields,
        )


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"size": "1280x720"}, (1280, 704)),
        ({"size": "854x480"}, (832, 480)),
        ({"size": "1920x1080"}, (1920, 1056)),
        ({"width": 720, "height": 1280}, (704, 1280)),
        ({"size": "1280x704"}, (1280, 704)),
        ({}, (1280, 704)),
    ],
)
def test_http_size_aligns_to_the_patch_grid(fields, expected):
    server_args = _ti2v_server_args()
    sampling = _build_video_sampling(server_args, fields)
    assert (sampling.width, sampling.height) == expected
    # the DiT splits the latent into 2x2 patches
    _, _, _, height, width = server_args.pipeline_config.prepare_latent_shape(
        Req(sampling_params=sampling), 1, 5
    )
    assert (height % 2, width % 2) == (0, 0)


def test_diffusers_backend_keeps_requested_size():
    server_args = _ti2v_server_args()
    server_args.backend = "diffusers"
    server_args.pipeline_config = DiffusersTI2VPipelineConfig()
    sampling = _build_video_sampling(server_args, {"size": "1280x720"})
    assert (sampling.width, sampling.height) == (1280, 720)


def test_text_only_warmup_resolution_aligns_to_the_patch_grid():
    server_args = _ti2v_server_args()
    # with breakable CUDA graphs the TI2V warmup carries no image, so the
    # requested resolution reaches the DiT as is
    server_args.enable_breakable_cuda_graph = True
    with patch(
        "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
        return_value=Wan2_2_TI2V_5B_SamplingParam(),
    ):
        reqs = build_warmup_reqs(
            server_args,
            warmup_resolutions=["1280x720"],
            warmup_input_path=None,
            server_based_warmup=True,
        )
    assert [(r.width, r.height) for r in reqs] == [(1280, 704)]
