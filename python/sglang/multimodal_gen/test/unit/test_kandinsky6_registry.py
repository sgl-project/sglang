# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 model routing and sampling defaults (no weights, no GPU, no Hub).

Four official Diffusers repos exist. The text(+image) -> video+audio pair:
``Kandinsky-6.0-Pro-sft-5s-Diffusers`` (flow-matching Euler, 50 steps at guidance
5.0) and the pi-Flow distilled ``Kandinsky-6.0-Pro-distill-5s-Diffusers`` (10 steps
at guidance 1.0). The video super-resolution pair: ``Kandinsky-6.0-VSR-5s-Diffusers``
and ``Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers``. The registry sends each repo
(and each local copy that keeps the repo name) to its pipeline classes, and picks
the distilled TI2VA sampling defaults by name.
"""

import dataclasses
from unittest.mock import MagicMock

import pytest

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
    _is_kandinsky6_t2va,
    _is_kandinsky6_t2va_distilled,
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

PRO = "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
SFT = "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers"
DISTILL = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
VSR = "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers"
VSR_DISTILLED = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"

# What the registry resolves a model path to:
# (pipeline class, pipeline config class, sampling params class)
TI2VA = (
    Kandinsky6TI2VAPipeline,
    Kandinsky6TI2VAPipelineConfig,
    Kandinsky6TI2VASamplingParams,
)
TI2VA_DISTILLED = (
    Kandinsky6TI2VAPipeline,
    Kandinsky6TI2VAPipelineConfig,
    Kandinsky6TI2VADistilledSamplingParams,
)
SR = (Kandinsky6SRPipeline, Kandinsky6SRPipelineConfig, Kandinsky6SRSamplingParams)

# Hugging Face cache snapshot directories are named after the snapshot hash only.
CACHED_SFT = (
    "/hub/models--kandinskylab--Kandinsky-6.0-Pro-sft-5s-Diffusers/snapshots/4f2a"
)
CACHED_DISTILL = (
    "/hub/models--kandinskylab--Kandinsky-6.0-Pro-distill-5s-Diffusers/snapshots/4f2a"
)

# Local copies keep the repo name.
ROUTES = [
    (PRO, TI2VA),
    (SFT, TI2VA),
    (DISTILL, TI2VA_DISTILLED),
    (VSR, SR),
    (VSR_DISTILLED, SR),
    ("/models/Kandinsky-6.0-Pro-sft-5s-Diffusers", TI2VA),
    ("/models/Kandinsky-6.0-Pro-5s-Diffusers", TI2VA),
    ("/models/Kandinsky-6.0-Pro-distill-5s-Diffusers", TI2VA_DISTILLED),
    ("/models/Kandinsky-6.0-Pro-distill-5s-Diffusers/", TI2VA_DISTILLED),
    ("/models/Kandinsky-6.0-VSR-5s-Diffusers", SR),
    ("/models/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", SR),
    (CACHED_SFT, TI2VA),
    (CACHED_DISTILL, TI2VA_DISTILLED),
]


@pytest.fixture
def fake_model_index(monkeypatch):
    """Make registry lookups read a ``model_index.json`` naming ``class_name``
    instead of asking the Hub."""

    def install(class_name):
        monkeypatch.setattr(
            registry,
            "maybe_download_model_index",
            lambda model_path: {"_class_name": class_name},
        )
        registry.get_model_info.cache_clear()
        registry._get_config_info.cache_clear()

    yield install
    registry.get_model_info.cache_clear()
    registry._get_config_info.cache_clear()


def _request_params(model_path, **user_kwargs):
    """The sampling params of a ``sglang generate`` / ``DiffGenerator`` request."""
    server_args = MagicMock()
    server_args.backend = "sglang"
    server_args.model_id = None
    server_args.pipeline_class_name = None
    return SamplingParams.from_user_sampling_params_args(
        model_path, server_args, prompt="a cat", **user_kwargs
    )


def test_official_repo_ids_and_legacy_alias_are_registered():
    registered = {
        model_path
        for model_path in registry._MODEL_HF_PATH_TO_NAME
        if model_path.startswith("kandinskylab/")
    }
    assert registered == {PRO, SFT, DISTILL, VSR, VSR_DISTILLED}


@pytest.mark.parametrize("model_path", [model_path for model_path, _ in ROUTES])
def test_official_ids_are_recognized_as_diffusion_models(model_path):
    """``sglang serve`` picks the diffusion server from this, before reading files."""
    assert registry.is_registered_diffusion_model_path(model_path)


@pytest.mark.parametrize(
    "name, expected",
    [
        (PRO, TI2VA),
        (SFT, TI2VA),
        ("Kandinsky6TI2VAPipeline", TI2VA),
        (DISTILL, TI2VA_DISTILLED),
        ("/models/Kandinsky-6.0-Pro-distill-5s-Diffusers/", TI2VA_DISTILLED),
        (VSR, SR),
        # "distill" in the name of an SR repo must not pick the TI2VA distilled entry
        (VSR_DISTILLED, SR),
        ("Kandinsky6SRPipeline", SR),
        # "distill" only in a parent directory: the last path component decides
        ("/distill_runs/Kandinsky-6.0-Pro-sft-5s-Diffusers", TI2VA),
    ],
)
def test_exactly_one_detector_claims_each_kandinsky6_name(name, expected):
    """The undistilled, distilled and SR detectors never overlap: no name is swallowed
    by the wrong entry, and none reaches the "more than one model matched" fallback
    of ``_get_config_info``."""
    claimed = [
        registry._CONFIG_REGISTRY[model_id]
        for model_id, detector in registry._MODEL_NAME_DETECTORS
        if detector(name.lower())
    ]
    _, pipeline_config_cls, sampling_param_cls = expected
    assert len(claimed) == 1
    assert claimed[0].pipeline_config_cls is pipeline_config_cls
    assert claimed[0].sampling_param_cls is sampling_param_cls


def test_distill_must_be_in_the_last_path_component_of_a_kandinsky6_name():
    assert _is_kandinsky6_t2va_distilled(DISTILL)
    assert not _is_kandinsky6_t2va(DISTILL)
    assert _is_kandinsky6_t2va(SFT)
    assert not _is_kandinsky6_t2va_distilled(SFT)
    assert not _is_kandinsky6_t2va_distilled("/distill/Kandinsky-6.0-Pro-sft")
    # not a Kandinsky6 text(+image) -> video+audio checkpoint at all
    assert not _is_kandinsky6_t2va_distilled("Wan-AI/Wan2.1-distill")
    assert not _is_kandinsky6_t2va_distilled(VSR_DISTILLED)


@pytest.mark.parametrize("model_path, expected", ROUTES)
def test_model_info_routes_each_official_id(fake_model_index, model_path, expected):
    pipeline_cls, pipeline_config_cls, sampling_param_cls = expected
    fake_model_index(pipeline_cls.pipeline_name)

    info = registry.get_model_info(model_path, backend="sglang")

    assert info.pipeline_cls is pipeline_cls
    assert info.pipeline_config_cls is pipeline_config_cls
    assert info.sampling_param_cls is sampling_param_cls


def test_model_id_override_gives_any_local_directory_the_repo_defaults(
    fake_model_index,
):
    fake_model_index("Kandinsky6TI2VAPipeline")

    info = registry.get_model_info(
        "/data/my_bundle",
        backend="sglang",
        model_id="Kandinsky-6.0-Pro-distill-5s-Diffusers",
    )
    assert info.sampling_param_cls is Kandinsky6TI2VADistilledSamplingParams

    # without the override the pipeline class name still selects the Kandinsky6 family
    info = registry.get_model_info("/data/my_bundle", backend="sglang")
    assert info.sampling_param_cls is Kandinsky6TI2VASamplingParams


def test_sampling_defaults_of_the_two_ti2va_checkpoints():
    base = Kandinsky6TI2VASamplingParams()
    distilled = Kandinsky6TI2VADistilledSamplingParams()
    assert (base.num_inference_steps, base.guidance_scale) == (50, 5.0)
    assert (distilled.num_inference_steps, distilled.guidance_scale) == (10, 1.0)

    # everything else is shared
    shared = {field.name for field in dataclasses.fields(base)} - {
        "num_inference_steps",
        "guidance_scale",
    }
    assert {name: getattr(distilled, name) for name in shared} == {
        name: getattr(base, name) for name in shared
    }
    assert (distilled.height, distilled.width) == (512, 768)
    assert (distilled.num_frames, distilled.fps) == (121, 24)


def test_distilled_repo_needs_no_steps_or_guidance_flags(fake_model_index):
    fake_model_index("Kandinsky6TI2VAPipeline")

    params = _request_params(DISTILL)
    assert isinstance(params, Kandinsky6TI2VADistilledSamplingParams)
    assert (params.num_inference_steps, params.guidance_scale) == (10, 1.0)

    params = _request_params("/models/Kandinsky-6.0-Pro-distill-5s-Diffusers")
    assert (params.num_inference_steps, params.guidance_scale) == (10, 1.0)


def test_explicit_steps_and_guidance_win_over_the_repo_defaults(fake_model_index):
    fake_model_index("Kandinsky6TI2VAPipeline")

    params = _request_params(DISTILL, num_inference_steps=8)
    assert (params.num_inference_steps, params.guidance_scale) == (8, 1.0)

    # an explicit value equal to the base default still counts as explicit
    params = _request_params(DISTILL, num_inference_steps=50)
    assert (params.num_inference_steps, params.guidance_scale) == (50, 1.0)

    params = _request_params(SFT)
    assert type(params) is Kandinsky6TI2VASamplingParams
    assert (params.num_inference_steps, params.guidance_scale) == (50, 5.0)

    params = _request_params(SFT, num_inference_steps=25, guidance_scale=3.0)
    assert (params.num_inference_steps, params.guidance_scale) == (25, 3.0)


def test_distilled_defaults_run_without_classifier_free_guidance(fake_model_index):
    """No guidance means no CFG: the negative prompt is never encoded and the DiT
    runs once per step."""
    fake_model_index("Kandinsky6TI2VAPipeline")

    distilled = Req(sampling_params=_request_params(DISTILL))
    base = Req(sampling_params=_request_params(SFT))

    assert not distilled.do_classifier_free_guidance
    assert base.do_classifier_free_guidance
