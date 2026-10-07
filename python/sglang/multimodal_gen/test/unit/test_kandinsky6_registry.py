# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 model routing and request defaults, without Hub access."""

import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6 import (
    Kandinsky6TI2VADistilledSamplingParams,
    Kandinsky6TI2VASamplingParams,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_pipeline import (
    Kandinsky6TI2VAPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_sr_pipeline import (
    Kandinsky6SRPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

SFT = "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers"
DISTILL = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
TI2VA = (
    Kandinsky6TI2VAPipeline,
    Kandinsky6TI2VAPipelineConfig,
    Kandinsky6TI2VASamplingParams,
)
DISTILLED = (
    Kandinsky6TI2VAPipeline,
    Kandinsky6TI2VAPipelineConfig,
    Kandinsky6TI2VADistilledSamplingParams,
)
SR = (Kandinsky6SRPipeline, Kandinsky6SRPipelineConfig, Kandinsky6SRSamplingParams)

ROUTES = [
    ("kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers", TI2VA),
    (SFT, TI2VA),
    (DISTILL, DISTILLED),
    ("kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", SR),
    ("kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", SR),
    ("/models/Kandinsky-6.0-Pro-5s-Diffusers", TI2VA),
    ("/models/Kandinsky-6.0-Pro-sft-5s-Diffusers", TI2VA),
    ("/models/Kandinsky-6.0-Pro-distill-5s-Diffusers/", DISTILLED),
    ("/models/Kandinsky-6.0-VSR-5s-Diffusers", SR),
    ("/models/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", SR),
    ("/distill_runs/Kandinsky-6.0-Pro-sft-5s-Diffusers", TI2VA),
    (
        "/hub/models--kandinskylab--Kandinsky-6.0-Pro-sft-5s-Diffusers/snapshots/4f2a",
        TI2VA,
    ),
    (
        "/hub/models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers/snapshots/4f2a",
        DISTILLED,
    ),
]


@pytest.fixture(autouse=True)
def clear_registry_cache():
    registry.get_model_info.cache_clear()
    registry._get_config_info.cache_clear()
    yield
    registry.get_model_info.cache_clear()
    registry._get_config_info.cache_clear()


@pytest.mark.parametrize("model_path,expected", ROUTES)
def test_model_routing(monkeypatch, model_path, expected):
    pipeline, _, _ = expected
    monkeypatch.setattr(
        registry,
        "maybe_download_model_index",
        lambda _: {"_class_name": pipeline.pipeline_name},
    )
    assert registry.is_registered_diffusion_model_path(model_path)
    info = registry.get_model_info(model_path, backend="sglang")
    assert (
        info.pipeline_cls,
        info.pipeline_config_cls,
        info.sampling_param_cls,
    ) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        (SFT, TI2VA),
        (DISTILL, DISTILLED),
        ("/models/kandinsky6-sr-native", SR),
        ("/models/Kandinsky_6.0_SR_x2", SR),
        ("Kandinsky6TI2VAPipeline", TI2VA),
        ("Kandinsky6T2VAPipeline", TI2VA),
        ("Kandinsky6SRPipeline", SR),
        ("/distill_runs/Kandinsky-6.0-Pro-sft-5s-Diffusers", TI2VA),
        ("kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", SR),
    ],
)
def test_detectors_do_not_overlap(name, expected):
    claimed = [
        registry._CONFIG_REGISTRY[model_id]
        for model_id, detector in registry._MODEL_NAME_DETECTORS
        if detector(name.lower())
    ]
    assert len(claimed) == 1
    assert (claimed[0].pipeline_config_cls, claimed[0].sampling_param_cls) == expected[
        1:
    ]


@pytest.mark.parametrize("expected", [TI2VA, SR])
def test_local_bundle_routing(tmp_path, expected):
    pipeline, _, _ = expected
    for component in ("transformer", "vae"):
        (tmp_path / component).mkdir()
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": pipeline.pipeline_name,
                "_diffusers_version": "0.37.0",
                "transformer": ["diffusers", "X"],
                "vae": ["diffusers", "Y"],
            }
        )
    )
    info = registry.get_model_info(str(tmp_path), backend="sglang")
    assert (
        info.pipeline_cls,
        info.pipeline_config_cls,
        info.sampling_param_cls,
    ) == expected
    if expected == TI2VA:
        info = registry.get_model_info(
            str(tmp_path),
            backend="sglang",
            model_id="Kandinsky-6.0-Pro-distill-5s-Diffusers",
        )
        assert info.sampling_param_cls is Kandinsky6TI2VADistilledSamplingParams


@pytest.mark.parametrize(
    "model_path,overrides,steps,guidance",
    [
        (SFT, {}, 50, 5.0),
        (DISTILL, {}, 10, 1.0),
        ("/models/Kandinsky-6.0-Pro-distill-5s-Diffusers", {}, 10, 1.0),
        (DISTILL, {"num_inference_steps": 8}, 8, 1.0),
        (DISTILL, {"num_inference_steps": 50}, 50, 1.0),
        (SFT, {"num_inference_steps": 25, "guidance_scale": 3.0}, 25, 3.0),
    ],
)
def test_request_defaults_and_overrides(
    monkeypatch, model_path, overrides, steps, guidance
):
    monkeypatch.setattr(
        registry,
        "maybe_download_model_index",
        lambda _: {"_class_name": "Kandinsky6TI2VAPipeline"},
    )
    args = SimpleNamespace(
        backend="sglang",
        model_id=None,
        pipeline_class_name=None,
        pipeline_config=Kandinsky6TI2VAPipelineConfig(),
        output_path=None,
        comfyui_mode=True,
        num_gpus=1,
    )
    params = SamplingParams.from_user_sampling_params_args(
        model_path, args, prompt="a cat", **overrides
    )
    assert (params.num_inference_steps, params.guidance_scale) == (steps, guidance)
    assert (params.height, params.width, params.num_frames, params.fps) == (
        512,
        768,
        121,
        24,
    )
    assert Req(sampling_params=params).do_classifier_free_guidance == (guidance > 1)


@pytest.mark.parametrize(
    "params_cls",
    [Kandinsky6TI2VASamplingParams, Kandinsky6TI2VADistilledSamplingParams],
)
@pytest.mark.parametrize("num_gpus", [1, 2, 3, 4, 8])
def test_request_frame_count_is_independent_of_gpu_count(params_cls, num_gpus):
    params = params_cls()
    params._adjust(
        SimpleNamespace(
            pipeline_config=Kandinsky6TI2VAPipelineConfig(),
            output_path=None,
            comfyui_mode=True,
            num_gpus=num_gpus,
        )
    )
    assert params.num_frames == 121
