# SPDX-License-Identifier: Apache-2.0
"""Validate scheduler selection, DiT call counts and incompatible head rejection."""

import os
from typing import NamedTuple

import pytest
from kandinsky6_sr_release_configs import (
    DISTILLED_SCHEDULER_CONFIG,
    DISTILLED_TRANSFORMER_CONFIG,
    FLOW_SCHEDULER_CONFIG,
    FLOW_TRANSFORMER_CONFIG,
)
from kandinsky6_sr_tiny_components import (
    TINY_DIT,
    build_components,
    make_request,
    make_stage,
    random_video,
)

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
    PiflowScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.denoising_stage import (
    Kandinsky6SRDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.latent_prep_stage import (
    Kandinsky6SRLatentPrepStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_VIDEO_KEY,
    build_sampling_spec,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    RESOLUTIONS,
)

LATENT_WIDTH = TINY_DIT["in_visual_dim"]
N_GRID = (
    3  # the distilled bundle's own grid width, independent of TINY_DIT's legacy fields
)
NUM_TILES = 9  # the 64x96 clip below, cut into 32x48 tiles


def official_dit(head_width):
    """Tiny transformer config in the official layout: ``out_visual_dim`` is the TOTAL
    head width, and ``n_grid`` / ``piflow_*`` live in the scheduler config."""
    arch = {
        key: value
        for key, value in TINY_DIT.items()
        if key != "out_visual_dim" and not key.startswith(("n_grid", "piflow_", "sr_"))
    }
    return dict(
        arch,
        out_visual_dim=head_width,
        sr_params=dict(scale_factor={"512": [1.0, 1.0, 1.0]}, visual_size=[512]),
    )


FLOW_DIT = official_dit(LATENT_WIDTH)  # one latent per token: flow-matching head
DX_DIT = official_dit(N_GRID * LATENT_WIDTH)  # N_GRID latents per token: pi-Flow head


def flow_scheduler():
    return FlowMatchEulerDiscreteScheduler(shift=5.0)


def pi_flow_scheduler(*, nfe=2, n_grid=N_GRID):
    return PiflowScheduler(
        nfe=nfe,
        n_grid=n_grid,
        shift=5.0,
        eps=1e-6,
        final_step_size_scale=0.5,
        num_policy_substeps=8,
    )


def release_scheduler(scheduler_cls, config):
    return scheduler_cls(**{k: v for k, v in config.items() if not k.startswith("_")})


@pytest.fixture(scope="module", autouse=True)
def single_process_model_parallel():
    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29510",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            os.environ.setdefault(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


@pytest.fixture(autouse=True)
def tiny_resolutions(monkeypatch):
    monkeypatch.setitem(RESOLUTIONS, 512, [(64, 64), (64, 96), (96, 64)])


class SamplingRun(NamedTuple):
    tiles: int
    dit_calls: int
    progress_totals: list


def run_denoising(dit_arch, scheduler, *, num_steps=5):
    """Run latent-prep + denoising (the LU-less pixel path) on the 9-tile clip and count
    the DiT calls next to the totals the denoising progress bar was opened with."""
    vae, dit, _, server_args = build_components((), dit_arch)
    dit_calls = []
    dit.register_forward_hook(lambda module, args, output: dit_calls.append(1))
    latent_prep = make_stage(
        Kandinsky6SRLatentPrepStage, server_args, vae, dit, None, scheduler
    )
    denoise = make_stage(Kandinsky6SRDenoisingStage, server_args, dit, scheduler)
    progress_totals = []
    progress_bar = denoise.progress_bar

    def spy_on_progress_bar(**kwargs):
        progress_totals.append(kwargs["total"])
        return progress_bar(**kwargs)

    denoise.progress_bar = spy_on_progress_bar

    batch = make_request(random_video(9, 64, 96), num_steps=num_steps)
    batch = latent_prep.forward(batch, server_args)
    tiles = len(batch.extra["kandinsky6_sr_chunks"])
    denoise.forward(batch, server_args)
    return SamplingRun(tiles, len(dit_calls), progress_totals)


def sampling_spec(dit_arch, scheduler):
    config = Kandinsky6SRDitConfig()
    config.update_model_arch(dict(dit_arch))
    return build_sampling_spec(
        arch=config.arch_config,
        tiling_scale=2,
        seed=42,
        num_steps=5,
        tiles_batch_size=1,
        tile_min_overlap=0.2,
        scheduler=scheduler,
    )


# --------------------------------------------------------------------------- #
# DiT calls per tile
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nfe,num_steps", [(2, 3), (2, 5), (2, 9), (3, 5)])
def test_pi_flow_scheduler_makes_nfe_calls_per_tile_whatever_num_inference_steps(
    nfe, num_steps
):
    run = run_denoising(DX_DIT, pi_flow_scheduler(nfe=nfe), num_steps=num_steps)
    assert run.tiles == NUM_TILES
    assert run.dit_calls == NUM_TILES * nfe
    assert run.progress_totals == [NUM_TILES * nfe]


@pytest.mark.parametrize("num_steps,calls_per_tile", [(3, 3), (5, 5), (9, 9)])
def test_flow_euler_scheduler_makes_num_inference_steps_calls_per_tile(
    num_steps, calls_per_tile
):
    run = run_denoising(FLOW_DIT, flow_scheduler(), num_steps=num_steps)
    assert run.tiles == NUM_TILES
    assert run.dit_calls == NUM_TILES * calls_per_tile
    assert run.progress_totals == [NUM_TILES * calls_per_tile]


@pytest.mark.parametrize("num_steps", [3, 9])
def test_legacy_config_with_its_own_pi_flow_fields_keeps_its_sampler(num_steps):
    """A flat DiT config carries n_grid / piflow_* and needs no scheduler component: the
    denoising stage synthesizes a matching PiflowScheduler from those arch fields
    (run_spec.effective_scheduler) instead of leaving nothing to step."""
    run = run_denoising(TINY_DIT, None, num_steps=num_steps)
    assert run.dit_calls == NUM_TILES * TINY_DIT["piflow_nfe"]
    assert run.progress_totals == [run.dit_calls]


# --------------------------------------------------------------------------- #
# Samplers that do not fit the DiT head
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scheduler",
    [flow_scheduler(), PiflowScheduler(n_grid=N_GRID, nfe=None), None],
    ids=["flow_euler_scheduler", "pi_flow_scheduler_without_nfe", "no_scheduler"],
)
def test_dx_head_never_runs_flow_euler(scheduler):
    with pytest.raises(ValueError, match="flow-Euler"):
        sampling_spec(DX_DIT, scheduler)


@pytest.mark.parametrize(
    "dit_arch,n_grid,needed",
    [
        (FLOW_DIT, N_GRID, "3 x 4 = 12"),
        (DX_DIT, 2, "2 x 4 = 8"),
        (DX_DIT, 4, "4 x 4 = 16"),
    ],
    ids=["single_head", "too_few_grids", "too_many_grids"],
)
def test_pi_flow_needs_a_head_of_n_grid_latents(dit_arch, n_grid, needed):
    with pytest.raises(ValueError, match=f"needs a {needed} channel DX head"):
        sampling_spec(dit_arch, pi_flow_scheduler(n_grid=n_grid))


@pytest.mark.parametrize("nfe", [0, -1])
def test_pi_flow_needs_at_least_one_call_per_tile(nfe):
    with pytest.raises(ValueError, match="nfe >= 1"):
        sampling_spec(DX_DIT, pi_flow_scheduler(nfe=nfe))
    with pytest.raises(ValueError, match="nfe >= 1"):
        sampling_spec(dict(TINY_DIT, piflow_nfe=nfe), None)


def test_release_transformer_and_scheduler_of_different_repos_are_rejected():
    """The real configs of the two official repos: their transformers differ only in
    the head width, so one repo's transformer with the other's scheduler must fail."""
    flow = release_scheduler(FlowMatchEulerDiscreteScheduler, FLOW_SCHEDULER_CONFIG)
    distilled = release_scheduler(PiflowScheduler, DISTILLED_SCHEDULER_CONFIG)

    assert sampling_spec(FLOW_TRANSFORMER_CONFIG, flow).steps_per_chunk == 5
    assert sampling_spec(DISTILLED_TRANSFORMER_CONFIG, distilled).steps_per_chunk == 2
    with pytest.raises(ValueError, match="640 channels wide.*flow-Euler"):
        sampling_spec(DISTILLED_TRANSFORMER_CONFIG, flow)
    with pytest.raises(ValueError, match="needs a 10 x 64 = 640 channel DX head"):
        sampling_spec(FLOW_TRANSFORMER_CONFIG, distilled)


def test_a_mismatched_bundle_fails_before_any_tile_is_processed():
    vae, dit, _, server_args = build_components((), DX_DIT)
    dit_calls = []
    dit.register_forward_hook(lambda module, args, output: dit_calls.append(1))
    latent_prep = make_stage(
        Kandinsky6SRLatentPrepStage, server_args, vae, dit, None, flow_scheduler()
    )
    batch = make_request(random_video(9, 64, 96))

    with pytest.raises(ValueError, match="flow-Euler"):
        latent_prep.forward(batch, server_args)
    assert dit_calls == []
    assert SR_VIDEO_KEY in batch.extra  # the clip was never cut into tiles
