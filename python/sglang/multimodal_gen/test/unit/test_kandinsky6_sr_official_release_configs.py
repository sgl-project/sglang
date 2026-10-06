# SPDX-License-Identifier: Apache-2.0
"""The real component configs of the two official Kandinsky 6 VSR Diffusers repos (no
weights) against the SR modules.

The flow-matching repo (``kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers``) and the 2-step
pi-Flow repo (``...-VSR-distilled2steps-5s-Diffusers``) differ only in their scheduler,
the width of the DiT head and ``sr_params``.  Each component is built on the meta device
from the release config exactly as the loaders would, and the tensor counts / key
groups are compared with the safetensors headers of the Hub repos (459 DiT, 378 KVAE,
489 latent-upscaler tensors).  This keeps the port from drifting away from the official
release without a GPU, the weights or the Hub.
"""

import copy
import json
import os
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
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    PiflowParams,
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
    calls_per_tile: int  # DiT calls per tile at the default ``num_inference_steps`` 5


FLOW = Bundle(
    model_index=FLOW_MODEL_INDEX,
    transformer=FLOW_TRANSFORMER_CONFIG,
    scheduler=FLOW_SCHEDULER_CONFIG,
    sr_config=FLOW_SR_CONFIG,
    scheduler_cls=FlowMatchEulerDiscreteScheduler,
    head_width=64,
    n_grid=1,
    scheduler_scale=5.0,
    calls_per_tile=5,
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
    calls_per_tile=2,
)
both_repos = pytest.mark.parametrize(
    "bundle", [FLOW, DISTILLED], ids=["flow_matching", "distilled_pi_flow"]
)


@pytest.fixture(scope="module", autouse=True)
def single_process_model_parallel():
    """The K6 feed-forward uses TP-aware linears, which need a (size-1) TP group."""
    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29509",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            os.environ.setdefault(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


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
    assert model.base_out_visual_dim == 64 and model.n_grid == 1
    assert bundle.n_grid * model.base_out_visual_dim == bundle.head_width
    assert model.instruct_type == "hybrid_anchor" and model.visual_cond
    assert model.visual_embed_dim == 2 * 64 + 1  # wide [x | cond | mask] input
    arch = config.arch_config
    assert arch.sr_scale_factor == {"512": [1.0, 2.0, 2.0]}
    assert arch.sr_scheduler_scale == bundle.scheduler_scale and not arch.is_piflow


@both_repos
def test_real_scheduler_config_loads_through_the_scheduler_loader(bundle, tmp_path):
    scheduler = _load_scheduler(bundle, tmp_path)
    # PiflowScheduler subclasses the flow-matching scheduler: compare exact types
    assert type(scheduler) is bundle.scheduler_cls
    assert scheduler.config["shift"] == bundle.scheduler["shift"]
    if bundle is DISTILLED:
        assert (scheduler.config["nfe"], scheduler.config["n_grid"]) == (2, 10)


@both_repos
def test_real_scheduler_selects_the_sampler_of_its_repo(bundle, tmp_path):
    """The distilled repo resolves to pi-Flow (``nfe`` = 2 calls per tile, whatever
    ``num_inference_steps`` is), the flow-matching repo to flow-Euler
    (``num_inference_steps`` calls per tile directly -- this repo's own convention, not
    the upstream Diffusers pipeline's timestep-grid-point count).
    """
    scheduler = _load_scheduler(bundle, tmp_path)
    arch = _dit_config(bundle).arch_config

    def spec(num_steps):
        return build_sampling_spec(
            arch=arch,
            tiling_scale=2,
            seed=42,
            num_steps=num_steps,
            tiles_batch_size=1,
            tile_min_overlap=0.2,
            scheduler=scheduler,
        )

    assert spec(5).steps_per_chunk == bundle.calls_per_tile
    assert spec(5).scheduler_scale == bundle.scheduler_scale
    if bundle is DISTILLED:
        assert spec(5).piflow == PiflowParams(
            nfe=2,
            num_policy_substeps=128,
            final_step_size_scale=0.5,
            shift=3.5,
            n_grid=10,
            eps=1e-6,
        )
        assert spec(3).steps_per_chunk == spec(9).steps_per_chunk == 2
    else:
        assert spec(5).piflow is None
        assert (spec(3).steps_per_chunk, spec(9).steps_per_chunk) == (3, 9)


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
def test_real_model_index_is_accepted_by_the_module_loading_path(
    bundle, tmp_path, monkeypatch
):
    """``ComposedPipelineBase.load_modules`` sees a model_index.json whose extra
    ``_kandinsky6_sr`` entry is a dict (not a ``[library, class]`` pair), next to a root
    ``sr_config.json``: only the four components of the pipeline may be requested, with
    the scheduler class of the repo."""
    (tmp_path / "model_index.json").write_text(json.dumps(bundle.model_index))
    if bundle.sr_config is not None:
        (tmp_path / "sr_config.json").write_text(json.dumps(bundle.sr_config))
    for name in ("transformer", "vae", "scheduler", "latent_upscaler"):
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
    ):
        assert component_type == component_name
        assert loader_cls is None and component_attn_backend is None
        assert component_attn_name == component_name
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

    assert sorted(modules) == ["latent_upscaler", "scheduler", "transformer", "vae"]
    assert requested == {
        "transformer": ("diffusers", "Kandinsky6SRTransformer3DModel"),
        "vae": ("diffusers", "Kandinsky6SRVAE"),
        "latent_upscaler": ("diffusers", "Kandinsky6SRLatentUpscalerBank"),
        "scheduler": ("diffusers", bundle.scheduler_cls.__name__),
    }
