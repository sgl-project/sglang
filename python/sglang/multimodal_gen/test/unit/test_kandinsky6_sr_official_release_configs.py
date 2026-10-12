# SPDX-License-Identifier: Apache-2.0
"""Validate release config shapes and component loading without downloading weights."""

import copy
import json
from collections import Counter
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from kandinsky6_sr_release_configs import (
    DISTILLED_MODEL_INDEX,
    DISTILLED_SCHEDULER_CONFIG,
    DISTILLED_TRANSFORMER_CONFIG,
    FLOW_MODEL_INDEX,
    FLOW_SCHEDULER_CONFIG,
    FLOW_SR_CONFIG,
    FLOW_TRANSFORMER_CONFIG,
    LATENT_UPSCALER_CONFIG,
    VAE_CONFIG,
)

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
    SchedulerLoader,
)
from sglang.multimodal_gen.runtime.models.dits.kandinsky6_sr import (
    Kandinsky6SRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
    PiflowScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.upsampler.kandinsky6_sr_latent_upscaler import (
    Kandinsky6SRLatentUpscalerBank,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae import Kandinsky6SRVAE
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_sr_pipeline import (
    Kandinsky6SRPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    build_sampling_spec,
)


class Bundle(NamedTuple):
    """One official repo: its configs and what they must resolve to."""

    model_index: dict
    transformer: dict
    scheduler: dict
    sr_config: dict | None
    scheduler_cls: type
    head_width: int  # ``out_visual_dim`` of the transformer: the TOTAL head width
    n_grid: int  # grids of the head: 1 (flow-matching) or the scheduler's ``n_grid``
    scheduler_scale: float  # ``sr_params.scheduler_scale``


FLOW = Bundle(
    model_index=FLOW_MODEL_INDEX,
    transformer=FLOW_TRANSFORMER_CONFIG,
    scheduler=FLOW_SCHEDULER_CONFIG,
    sr_config=FLOW_SR_CONFIG,
    scheduler_cls=FlowMatchEulerDiscreteScheduler,
    head_width=64,
    n_grid=1,
    scheduler_scale=5.0,
)
DISTILLED = Bundle(
    model_index=DISTILLED_MODEL_INDEX,
    transformer=DISTILLED_TRANSFORMER_CONFIG,
    scheduler=DISTILLED_SCHEDULER_CONFIG,
    sr_config=None,
    scheduler_cls=PiflowScheduler,
    head_width=640,
    n_grid=10,
    scheduler_scale=3.5,
)
both_repos = pytest.mark.parametrize(
    "bundle", [FLOW, DISTILLED], ids=["flow_matching", "distilled_pi_flow"]
)


pytestmark = pytest.mark.usefixtures("single_process_model_parallel")


def _prefixes(keys, depth=2):
    return Counter(".".join(key.split(".")[:depth]) for key in keys)


def _dit_config(bundle):
    config = Kandinsky6SRDitConfig()
    config.update_model_arch(copy.deepcopy(bundle.transformer))
    return config


def _load_scheduler(bundle, tmp_path):
    """The scheduler component through SGLang's own loader (a ``scheduler/`` dir)."""
    directory = tmp_path / "scheduler"
    directory.mkdir()
    (directory / "scheduler_config.json").write_text(json.dumps(bundle.scheduler))
    server_args = SimpleNamespace(pipeline_config=Kandinsky6SRPipelineConfig())
    return SchedulerLoader().load_customized(str(directory), server_args)


@both_repos
def test_real_transformer_config_builds_the_release_dit(bundle):
    config = _dit_config(bundle)
    with torch.device("meta"):
        model = Kandinsky6SRTransformer3DModel(
            config, copy.deepcopy(bundle.transformer)
        )
    state = model.state_dict()
    per_block = Counter(
        key.split(".")[1]
        for key in state
        if key.startswith("visual_transformer_blocks.")
    )
    assert (
        len(state) == 459 and len(per_block) == 32 and set(per_block.values()) == {14}
    )
    non_block = [k for k in state if not k.startswith("visual_transformer_blocks.")]
    assert len(non_block) == 11
    # ``out_visual_dim`` is the TOTAL head width (64 * n_grid); a pi-Flow repo keeps its
    # n_grid in the scheduler config
    assert state["out_layer.out_layer.weight"].shape == (bundle.head_width, 1792)
    assert bundle.n_grid * model.in_visual_dim == bundle.head_width
    assert model.instruct_type == "hybrid_anchor" and model.visual_cond
    assert model.visual_embed_dim == 2 * 64 + 1  # wide [x | cond | mask] input
    arch = config.arch_config
    assert arch.sr_scale_factor == {"512": [1.0, 2.0, 2.0]}
    assert arch.sr_scheduler_scale == bundle.scheduler_scale


@both_repos
def test_loaded_scheduler_controls_sampling_and_rejects_a_mismatched_head(
    bundle, tmp_path
):
    scheduler = _load_scheduler(bundle, tmp_path)
    # PiflowScheduler subclasses the flow-matching scheduler: compare exact types
    assert type(scheduler) is bundle.scheduler_cls
    assert scheduler.config["shift"] == bundle.scheduler["shift"]
    if bundle is DISTILLED:
        assert (scheduler.config["nfe"], scheduler.config["n_grid"]) == (2, 10)

    arch = _dit_config(bundle).arch_config
    for num_steps in (3, 5, 9):
        spec = _sampling_spec(arch, scheduler, num_steps)
        assert spec.is_piflow == (bundle is DISTILLED)
        assert spec.num_steps == (2 if bundle is DISTILLED else num_steps)
    mismatched = _dit_config(FLOW if bundle is DISTILLED else DISTILLED).arch_config
    with pytest.raises(ValueError, match="head.*same checkpoint"):
        _sampling_spec(mismatched, scheduler)


def _sampling_spec(arch, scheduler, num_steps=5):
    return build_sampling_spec(
        arch=arch,
        scheduler=scheduler,
        tiling_scale=2,
        seed=42,
        num_steps=num_steps,
        tiles_batch_size=1,
        tile_min_overlap=0.2,
    )


@pytest.mark.parametrize("nfe", [None, 0, -1])
def test_piflow_requires_a_positive_checkpoint_step_count(nfe):
    with pytest.raises(ValueError, match="nfe >= 1"):
        _sampling_spec(
            _dit_config(DISTILLED).arch_config, PiflowScheduler(nfe=nfe, n_grid=10)
        )


def test_sampling_rejects_missing_scheduler_steps_and_unsupported_cap():
    arch = _dit_config(DISTILLED).arch_config
    for scheduler in (None, object()):
        with pytest.raises(ValueError, match="scheduler|Scheduler"):
            _sampling_spec(arch, scheduler)
    scheduler = PiflowScheduler(nfe=2, n_grid=10)
    with pytest.raises(ValueError, match="num_inference_steps"):
        _sampling_spec(arch, scheduler, num_steps=0)
    arch.sr_cap_noise_timestep = True
    arch.attribute_overrides = {"instruct_type": "noise"}
    with pytest.raises(NotImplementedError, match="cap_noise_timestep"):
        _sampling_spec(arch, scheduler)


def test_real_vae_config_builds_the_release_kvae_and_maps_official_keys():
    config = Kandinsky6SRVAEConfig()
    config.update_model_arch(dict(VAE_CONFIG))
    with torch.device("meta"):
        vae = Kandinsky6SRVAE(config)
    # `self.encoder` / `self.decoder` are direct attributes of the wrapper (no intermediate
    # `model.*` nesting), so its state_dict keys already are the official checkpoint's own
    # `encoder.*` / `decoder.*` keys -- no stripping needed.
    official = dict(vae.state_dict())
    assert len(official) == 378
    assert _prefixes(official) == Counter(
        {
            "decoder.up": 238,
            "encoder.down": 86,
            "decoder.mid": 28,
            "encoder.mid": 12,
            "decoder.norm_out": 5,
            "decoder.conv_in": 2,
            "decoder.conv_out": 2,
            "encoder.conv_in": 2,
            "encoder.conv_out": 2,
            "encoder.norm_out": 1,
        }
    )
    assert vae.scaling_factor == pytest.approx(0.910344004631042)
    assert vae.spatial_factor == 16
    # the official keys load strictly, with no remapping
    vae.load_state_dict(official, strict=True, assign=True)


def test_real_latent_upscaler_config_builds_the_release_bank():
    with torch.device("meta"):
        bank = Kandinsky6SRLatentUpscalerBank(
            LATENT_UPSCALER_CONFIG["models"],
            LATENT_UPSCALER_CONFIG["scaling_factor"],
            scales=(2, 4),
        )
    state = bank.state_dict()
    assert len(state) == 489
    assert _prefixes(state) == Counter({"_models.0": 311, "_models.1": 178})
    x2 = _prefixes((key.split(".", 2)[2] for key in state if "_models.0." in key))
    assert x2["x2_branch.blocks"] == 44 and x2["x2_branch.adapter"] == 28
    assert bank.scales == (2, 4)


@both_repos
@pytest.mark.parametrize("upscaler", ["present", "missing", "null"])
def test_real_model_index_is_accepted_by_the_module_loading_path(
    bundle, tmp_path, monkeypatch, upscaler
):
    """``ComposedPipelineBase.load_modules`` sees a model_index.json whose extra
    ``_kandinsky6_sr`` entry is a dict (not a ``[library, class]`` pair), next to a root
    ``sr_config.json``: only the four components of the pipeline may be requested, with
    the scheduler class of the repo."""
    model_index = dict(bundle.model_index)
    if upscaler == "missing":
        model_index.pop("latent_upscaler")
    elif upscaler == "null":
        model_index["latent_upscaler"] = None
    (tmp_path / "model_index.json").write_text(json.dumps(model_index))
    if bundle.sr_config is not None:
        (tmp_path / "sr_config.json").write_text(json.dumps(bundle.sr_config))
    components = ["transformer", "vae", "scheduler"]
    if upscaler == "present":
        components.append("latent_upscaler")
    for name in components:
        (tmp_path / name).mkdir()
        if name != "scheduler":  # the scheduler is config-only
            (tmp_path / name / "diffusion_pytorch_model.safetensors").touch()

    requested = {}

    def record_component(
        component_name,
        component_model_path,
        transformers_or_diffusers,
        server_args,
        component_architecture=None,
        component_type=None,
        loader_cls=None,
        component_attn_backend=None,
        component_attn_name=None,
        component_backend_by_role=None,
    ):
        assert component_type == component_name
        assert loader_cls is None and component_attn_backend is None
        assert component_attn_name == component_name
        assert component_backend_by_role == {}
        requested[component_name] = (transformers_or_diffusers, component_architecture)
        return SimpleNamespace(name=component_name), 0.0

    monkeypatch.setattr(
        PipelineComponentLoader, "load_component", staticmethod(record_component)
    )
    server_args = SimpleNamespace(
        model_subfolder=None,
        revision=None,
        comfyui_mode=False,
        component_paths={},
        component_direct_gpu_weight_loading={},
        pipeline_config=Kandinsky6SRPipelineConfig(),
        resolve_component_attention_backend=lambda *names: (None, None),
        resolve_component_backend_by_role=lambda *names: {},
    )
    # the state ``ComposedPipelineBase.__init__`` sets up before it calls load_modules
    pipeline = object.__new__(Kandinsky6SRPipeline)
    pipeline.server_args = server_args
    pipeline.model_path = str(tmp_path)
    pipeline._required_config_modules = list(
        Kandinsky6SRPipeline._required_config_modules
    )
    pipeline._extra_config_module_map = {}
    pipeline._disagg_role = RoleType.MONOLITHIC
    pipeline.memory_usages = {}

    modules = pipeline.load_modules(server_args)

    expected = {
        "transformer": ("diffusers", "Kandinsky6SRTransformer3DModel"),
        "vae": ("diffusers", "Kandinsky6SRVAE"),
        "latent_upscaler": ("diffusers", "Kandinsky6SRLatentUpscalerBank"),
        "scheduler": ("diffusers", bundle.scheduler_cls.__name__),
    }
    if upscaler != "present":
        expected.pop("latent_upscaler")
    assert set(modules) == set(expected)
    assert requested == expected
    assert "latent_upscaler" in Kandinsky6SRPipeline._required_config_modules


def test_legacy_bundle_requires_official_diffusers_layout(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Kandinsky6SRPipeline",
                "_diffusers_version": "0.37.0",
                "dit": ["x", "Kandinsky6SRDiT"],
                "vae": ["x", "Kandinsky6SRVAE"],
            }
        )
    )
    for name in ("dit", "vae"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "diffusion_pytorch_model.safetensors").touch()
    pipeline = object.__new__(Kandinsky6SRPipeline)
    pipeline.server_args = SimpleNamespace(model_subfolder=None, revision=None)
    pipeline.model_path = str(tmp_path)
    with pytest.raises(
        ValueError, match="k6_video SR bundle.*kandinskylab/Kandinsky-6.0-VSR"
    ):
        pipeline.load_modules(pipeline.server_args)
