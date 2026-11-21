# SPDX-License-Identifier: Apache-2.0
"""LTX-2 two-stage requests must reach the halve stage on its 64-pixel grid."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import LTX25PipelineConfig
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX23HQSamplingParams
from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.upsampling import (
    LTX2HalveResolutionStage,
)

TWO_STAGE = "LTX2TwoStagePipeline"


def _server_args(pipeline_class_name):
    return SimpleNamespace(
        model_path="Lightricks/LTX-2.5-Diffusers",
        model_id=None,
        backend="sglang",
        performance_mode="manual",
        pipeline_class_name=pipeline_class_name,
        pipeline_config=LTX25PipelineConfig(),
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


def _build_video_sampling(server_args, defaults, fields):
    from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils

    with (
        patch.object(openai_utils, "get_global_server_args", return_value=server_args),
        patch.object(SamplingParams, "from_pretrained", return_value=defaults),
    ):
        return openai_utils.build_sampling_params(
            "size",
            prompt="a cat",
            request_data_type=DataType.VIDEO,
            **fields,
        )


@pytest.mark.parametrize(
    "pipeline_class_name,defaults,fields,expected",
    [
        (TWO_STAGE, LTX25SamplingParams, {"size": "1280x720"}, (1280, 704)),
        (TWO_STAGE, LTX25SamplingParams, {"size": "1000x600"}, (960, 576)),
        (TWO_STAGE, LTX25SamplingParams, {"width": 40, "height": 40}, (64, 64)),
        (TWO_STAGE, LTX25SamplingParams, {}, (1920, 1088)),
        (TWO_STAGE, LTX23HQSamplingParams, {"size": "1280x720"}, (1280, 704)),
        # One-stage has no halve stage, so its size must stay untouched.
        ("LTX2Pipeline", LTX25SamplingParams, {"size": "1000x600"}, (1000, 600)),
    ],
)
def test_http_size_aligns_to_the_halve_stage_grid(
    pipeline_class_name, defaults, fields, expected
):
    """An unaligned two-stage size used to raise in the halve stage (HTTP 500)."""
    server_args = _server_args(pipeline_class_name)
    sampling = _build_video_sampling(server_args, defaults(), fields)
    assert (sampling.width, sampling.height) == expected
    if pipeline_class_name == TWO_STAGE:
        stage = object.__new__(LTX2HalveResolutionStage)
        stage.forward(Req(sampling_params=sampling), server_args)


@pytest.mark.parametrize(
    "pipeline_class_name,expected",
    [(TWO_STAGE, (1280, 704)), ("LTX2Pipeline", (1280, 720))],
)
def test_warmup_resolution_aligns_to_the_halve_stage_grid(
    pipeline_class_name, expected
):
    """Warmup sets its size after _adjust, so the warmup hook must align it too."""
    params = LTX25SamplingParams(width=1280, height=720)
    req = SimpleNamespace(extra={})
    params.prepare_synthetic_warmup_request_for_queue(
        req, _server_args(pipeline_class_name)
    )
    assert (params.width, params.height) == expected
