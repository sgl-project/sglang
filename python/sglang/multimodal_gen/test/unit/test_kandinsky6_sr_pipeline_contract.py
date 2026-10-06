# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6SRPipeline construction / stage-wiring contract (no weights, no GPU).

Same pattern as ``test_kandinsky6_pipeline_contract.py``: build the pipeline with
``object.__new__`` (skipping the weight loading of ``ComposedPipelineBase.__init__``),
hand it mocked modules and call ``create_pipeline_stages`` directly.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_sr_pipeline import (
    Kandinsky6SRPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.decode_stage import (
    Kandinsky6SRDecodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.denoising_stage import (
    Kandinsky6SRDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.encode_stage import (
    Kandinsky6SREncodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.input_stage import (
    Kandinsky6SRInputStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.latent_prep_stage import (
    Kandinsky6SRLatentPrepStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.output_stage import (
    Kandinsky6SROutputStage,
)

EXPECTED_STAGE_ORDER = [
    ("input_stage", Kandinsky6SRInputStage),
    ("encode_stage", Kandinsky6SREncodeStage),
    ("latent_prep_stage", Kandinsky6SRLatentPrepStage),
    ("denoising_stage", Kandinsky6SRDenoisingStage),
    ("decode_stage", Kandinsky6SRDecodeStage),
    ("output_stage", Kandinsky6SROutputStage),
]


def _make_pipeline(*, with_upscaler: bool = True) -> Kandinsky6SRPipeline:
    pipeline = object.__new__(Kandinsky6SRPipeline)
    names = ["transformer", "vae", "scheduler"] + (
        ["latent_upscaler"] if with_upscaler else []
    )
    pipeline.modules = {name: MagicMock(name=name) for name in names}
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline._disagg_role = RoleType.MONOLITHIC
    return pipeline


def _server_args():
    return SimpleNamespace(
        pipeline_config=Kandinsky6SRPipelineConfig(), component_precisions={}
    )


def test_pipeline_identity_matches_the_official_repo_contract():
    """``_class_name`` in model_index.json must equal ``pipeline_name`` or the backend
    silently falls back to diffusers; the modules are the official repo's components."""
    assert Kandinsky6SRPipeline.pipeline_name == "Kandinsky6SRPipeline"
    assert Kandinsky6SRPipeline.is_video_pipeline is True
    assert Kandinsky6SRPipeline._required_config_modules == [
        "transformer",
        "vae",
        "scheduler",
        "latent_upscaler",
    ]
    assert Kandinsky6SRPipeline._optional_config_modules == ("latent_upscaler",)


def test_disagg_role_rejects_non_monolithic_deployment():
    pipeline = object.__new__(Kandinsky6SRPipeline)
    for role in (RoleType.ENCODER, RoleType.DENOISER, RoleType.DECODER):
        with pytest.raises(ValueError, match="monolithic"):
            pipeline.validate_disagg_role(role)
    pipeline.validate_disagg_role(RoleType.MONOLITHIC)


@pytest.mark.parametrize("with_upscaler", [True, False])
def test_create_pipeline_stages_order_and_wiring(with_upscaler):
    pipeline = _make_pipeline(with_upscaler=with_upscaler)
    Kandinsky6SRPipeline.create_pipeline_stages(pipeline, MagicMock())

    assert [type(s) for s in pipeline.stages] == [c for _, c in EXPECTED_STAGE_ORDER]
    assert list(pipeline._stage_name_mapping) == [n for n, _ in EXPECTED_STAGE_ORDER]
    modules = pipeline.modules
    lu = modules.get("latent_upscaler")  # None for a pixel-path-only bundle
    encode = pipeline._stage_name_mapping["encode_stage"]
    latent_prep = pipeline._stage_name_mapping["latent_prep_stage"]
    denoising = pipeline._stage_name_mapping["denoising_stage"]
    decode = pipeline._stage_name_mapping["decode_stage"]
    assert encode.vae is modules["vae"] and encode.latent_upscaler is lu
    assert latent_prep.vae is modules["vae"] and latent_prep.latent_upscaler is lu
    assert latent_prep.transformer is modules["transformer"]
    assert latent_prep.scheduler is modules["scheduler"]
    assert denoising.transformer is modules["transformer"]
    assert denoising.scheduler is modules["scheduler"]
    assert decode.vae is modules["vae"]


def test_component_uses_load_each_component_only_when_it_is_needed():
    """The residency manager pre-loads declared components: the encode stage must not pull
    the VAE onto the GPU at stage entry when the pixel path skips it, and each model-phase
    stage must declare exactly its own phase so that offloading moves each component once
    per request instead of once per tile."""
    pipeline = _make_pipeline()
    Kandinsky6SRPipeline.create_pipeline_stages(pipeline, MagicMock())
    args = _server_args()

    encode_uses = pipeline._stage_name_mapping["encode_stage"].component_uses(args)
    assert [(u.component_name, u.start_at_stage_entry) for u in encode_uses] == [
        ("vae", False)
    ]

    latent_prep = pipeline._stage_name_mapping["latent_prep_stage"]
    uses = latent_prep.component_uses(args, "latent_prep_stage")
    assert [(u.component_name, u.phase) for u in uses] == [
        ("latent_upscaler", "upscale_tiles"),
        ("vae", "encode_tiles"),
    ]

    denoising = pipeline._stage_name_mapping["denoising_stage"]
    dit_uses = denoising.component_uses(args, "denoising_stage")
    assert [(u.component_name, u.phase) for u in dit_uses] == [
        ("transformer", "denoise_tiles")
    ]
    dit_use = dit_uses[0]
    assert dit_use.memory_intensive and dit_use.target_dtype == torch.bfloat16

    decode_uses = pipeline._stage_name_mapping["decode_stage"].component_uses(
        args, "decode_stage"
    )
    assert [(u.component_name, u.phase) for u in decode_uses] == [
        ("vae", "decode_tiles")
    ]

    pixel_only = _make_pipeline(with_upscaler=False)
    Kandinsky6SRPipeline.create_pipeline_stages(pixel_only, MagicMock())
    uses = pixel_only._stage_name_mapping["latent_prep_stage"].component_uses(args)
    assert "latent_upscaler" not in [u.component_name for u in uses]


def _pipeline_with_index(index):
    """Pipeline whose ``_load_config`` returns ``index`` (file discovery is not under test)."""
    pipeline = object.__new__(Kandinsky6SRPipeline)
    pipeline.model_path = "bundle"
    pipeline._load_config = lambda: dict(index)
    pipeline._required_config_modules = list(
        Kandinsky6SRPipeline._required_config_modules
    )
    return pipeline


def test_k6_video_sr_bundle_is_rejected_with_a_pointer_to_the_official_repos():
    """The k6_video bundle names its DiT ``dit``; without this check the loader would
    die with a bare KeyError on 'transformer'.  The error names the two official repos
    and nothing else to use."""
    pipeline = _pipeline_with_index(
        {
            "_class_name": ["pipeline_kandinsky6_sr", "Kandinsky6SRPipeline"],
            "_diffusers_version": "0.37.0",
            "dit": ["x", "Kandinsky6SRDiT"],
            "vae": ["x", "Kandinsky6SRVAE"],
        }
    )
    with pytest.raises(ValueError, match="k6_video SR bundle") as error:
        pipeline.load_modules(SimpleNamespace())
    message = str(error.value)
    assert "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers" in message
    assert "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers" in message
    assert "convert" not in message and "raw" not in message


def test_repo_without_latent_upscaler_loads_only_the_remaining_modules(monkeypatch):
    """``ComposedPipelineBase.load_modules`` indexes model_index with every required
    module; the pixel-path-only repo (no bank) must therefore not require it."""
    seen = {}
    monkeypatch.setattr(
        ComposedPipelineBase,
        "load_modules",
        lambda self, server_args, loaded_modules=None: seen.update(
            required=list(self._required_config_modules)
        ),
    )
    base = {
        "_class_name": "Kandinsky6SRPipeline",
        "_diffusers_version": "0.37.0",
        "transformer": ["diffusers", "Kandinsky6SRTransformer3DModel"],
        "vae": ["diffusers", "Kandinsky6SRVAE"],
        "scheduler": ["diffusers", "PiflowScheduler"],
    }
    _pipeline_with_index(base).load_modules(SimpleNamespace())
    assert seen["required"] == ["transformer", "vae", "scheduler"]

    bank = ["diffusers", "Kandinsky6SRLatentUpscalerBank"]
    _pipeline_with_index({**base, "latent_upscaler": bank}).load_modules(
        SimpleNamespace()
    )
    assert seen["required"] == ["transformer", "vae", "scheduler", "latent_upscaler"]
    _pipeline_with_index({**base, "latent_upscaler": None}).load_modules(
        SimpleNamespace()
    )
    assert seen["required"] == ["transformer", "vae", "scheduler"]
