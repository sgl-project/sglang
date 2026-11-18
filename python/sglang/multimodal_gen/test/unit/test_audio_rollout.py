# SPDX-License-Identifier: Apache-2.0
"""Rollout of named latent streams: the per-stream hooks, per-sample noise, LTX-2 audio wiring, and the extra_streams response."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from sglang.multimodal_gen.configs.post_training.rl_rollout import (
    AUDIO_STREAM,
    RLRolloutArgs,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
)
from sglang.multimodal_gen.runtime.post_training.flow_rollout_controller import (
    truncate_rollout_sigmas,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    concat_rollout_trajectory_data,
    select_output_rollout_trajectory,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    PRIMARY_STREAM,
    derive_extra_stream_generators,
    finalize_rollout_if_enabled,
    init_rollout_env_if_enabled,
    pipeline_config_sp_fns,
    prepare_rollout_if_enabled,
    publish_rollout_if_enabled,
    rollout_scope,
    rollout_step_kwargs,
)

_PIPELINE_CONFIG = SimpleNamespace(
    shard_latents_for_sp=lambda batch, latents: (latents, False)
)


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
        lambda: 1,
    )
    req = Req(
        sampling_params=SamplingParams(
            prompt="test",
            seed=10,
            rollout=True,
            rollout_extra_streams=(AUDIO_STREAM,),
            rollout_return_dit_trajectory=True,
            rollout_return_step_indices=[0, 2],
        )
    )
    req.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    req.scheduler.set_timesteps(sigmas=[0.8, 0.4])
    req.timesteps = req.scheduler.timesteps
    req.latents = torch.zeros(2, 4, 2)
    req.audio_latents = torch.ones(2, 5, 3)
    req.generator = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    return req


def _prepare_primary_stream(batch):
    """What a denoising stage does for the stream its loop iterates."""
    shard_for_sp, gather_for_sp = pipeline_config_sp_fns(_PIPELINE_CONFIG)
    prepare_rollout_if_enabled(
        batch=batch,
        stream=PRIMARY_STREAM,
        scheduler=batch.scheduler,
        timesteps=batch.timesteps,
        latents=batch.latents,
        shard_for_sp=shard_for_sp,
        gather_for_sp=gather_for_sp,
    )


def _finalize_stream(batch, stream, latents):
    finalize_rollout_if_enabled(
        batch=batch,
        stream=stream,
        latents=latents,
        num_inference_steps=len(batch.timesteps),
        final_timestep=torch.tensor(0.0),
    )


def test_extra_stream_rng_depends_only_on_the_request_seed_and_the_stream_name(batch):
    """An extra stream's seed depends only on the request seed and the stream name."""
    request_states = [g.get_state() for g in batch.generator]
    audio = derive_extra_stream_generators(batch.generator, stream=AUDIO_STREAM)
    assert audio[0].initial_seed() == 635590486471046510
    action = derive_extra_stream_generators(batch.generator, stream="action")
    all_seeds = [g.initial_seed() for g in (*batch.generator, *audio, *action)]
    assert len(set(all_seeds)) == len(all_seeds)
    for request_generator, state in zip(batch.generator, request_states):
        torch.testing.assert_close(request_generator.get_state(), state, rtol=0, atol=0)


def _identity_for_sp(batch, tensor):
    return tensor


def _prepare_stream_on_its_own_grid(batch, stream, latents):
    """A FlowMatch Euler stream with the primary's step count but another shift."""
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(sigmas=[0.8, 0.4])
    uncut_sigmas = scheduler.sigmas.clone()
    prepare_rollout_if_enabled(
        batch=batch,
        stream=stream,
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        latents=latents,
        shard_for_sp=_identity_for_sp,
        gather_for_sp=_identity_for_sp,
    )
    return scheduler, uncut_sigmas


@pytest.mark.parametrize("stop_step_index", [None, 0])
def test_any_named_stream_with_its_own_sigmas_reaches_the_response(
    batch, stop_step_index
):
    """No code between the hook and the response branches on a stream's name."""
    stream = "action"
    batch.rollout_extra_streams = (stream,)
    batch.rollout_stop_step_index = stop_step_index
    batch.rollout_return_step_indices = None
    action_latents = batch.audio_latents
    with rollout_scope(batch):
        _prepare_primary_stream(batch)
        action_scheduler, uncut_sigmas = _prepare_stream_on_its_own_grid(
            batch, stream, action_latents
        )
        controller = batch.rollout_controllers[stream]
        expected_sigmas = (
            uncut_sigmas
            if stop_step_index is None
            else truncate_rollout_sigmas(uncut_sigmas, stop_step_index=stop_step_index)
        )
        torch.testing.assert_close(controller.sigmas, expected_sigmas, rtol=0, atol=0)
        assert not torch.equal(
            controller.sigmas, batch.rollout_controllers[PRIMARY_STREAM].sigmas
        )
        # A stream on its own grid steps with its controller's timesteps.
        for timestep, action_timestep in zip(batch.timesteps, controller.timesteps):
            batch.latents = batch.scheduler.step(
                torch.zeros_like(batch.latents),
                timestep,
                batch.latents,
                **rollout_step_kwargs(batch, stream=PRIMARY_STREAM),
                return_dict=False,
            )[0]
            action_latents = action_scheduler.step(
                torch.zeros_like(action_latents),
                action_timestep,
                action_latents,
                **rollout_step_kwargs(batch, stream=stream),
                return_dict=False,
            )[0]
        _finalize_stream(batch, PRIMARY_STREAM, batch.latents)
        _finalize_stream(batch, stream, action_latents)
        publish_rollout_if_enabled(batch=batch, pipeline_config=_PIPELINE_CONFIG)
    data = batch.rollout_trajectory_data
    assert set(data.streams) == {PRIMARY_STREAM, stream}
    assert torch.isfinite(data.streams[stream].rollout_log_probs).all()
    merged = concat_rollout_trajectory_data(
        [select_output_rollout_trajectory(data, i) for i in range(2)]
    )
    result = OutputBatch(output=torch.zeros(2, 1), rollout_trajectory_data=merged)
    for response in rollout_api._build_response("r", "p", 10, True, result):
        assert set(response.extra_streams) == {stream}
        assert response.extra_streams[stream]["generated_output"] is None
    stored = {}
    rollout_api._build_response("r", "p", 10, True, result, stored=stored)
    assert any(key.startswith(f"extra_streams.{stream}.") for key in stored)


def test_stream_not_requested_for_rollout_cuts_its_own_grid_at_the_stop(batch):
    """A stream the request does not roll out still ends on its own terminal sigma."""
    batch.rollout_extra_streams = ()
    batch.rollout_stop_step_index = 0
    _prepare_primary_stream(batch)
    action_latents = batch.audio_latents
    action_scheduler, uncut_sigmas = _prepare_stream_on_its_own_grid(
        batch, "action", action_latents
    )
    assert "action" not in batch.rollout_controllers
    torch.testing.assert_close(
        action_scheduler.sigmas,
        truncate_rollout_sigmas(uncut_sigmas, stop_step_index=0),
        rtol=0,
        atol=0,
    )
    assert len(action_scheduler.timesteps) == 1


def test_extra_stream_with_another_step_count_is_rejected(batch):
    batch.rollout_extra_streams = ("action",)
    _prepare_primary_stream(batch)
    action_latents = batch.audio_latents
    action_scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    action_scheduler.set_timesteps(sigmas=[0.9, 0.6, 0.3])
    with pytest.raises(ValueError, match="streams step together"):
        prepare_rollout_if_enabled(
            batch=batch,
            stream="action",
            scheduler=action_scheduler,
            timesteps=action_scheduler.timesteps,
            latents=action_latents,
            shard_for_sp=_identity_for_sp,
            gather_for_sp=_identity_for_sp,
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (
            {"rollout": False, "rollout_extra_streams": ("audio",)},
            "requires rollout=True",
        ),
        ({"rollout": True, "rollout_extra_streams": ("audio", "audio")}, "duplicates"),
        (
            {"rollout": True, "rollout_extra_streams": (PRIMARY_STREAM,)},
            "Invalid rollout extra stream",
        ),
        (
            {"rollout": True, "rollout_extra_streams": ("a.b",)},
            "Invalid rollout extra stream",
        ),
        ({"rollout": True, "rollout_extra_streams": "audio"}, "must be a list"),
    ],
)
def test_invalid_extra_stream_names_are_rejected_up_front(kwargs, error):
    """Stream names become store path segments, so they are checked up front."""
    with pytest.raises(ValueError, match=error):
        SamplingParams(prompt="test", **kwargs)
    with pytest.raises(ValueError, match=error):
        RLRolloutArgs.from_dict(kwargs).validate()


def test_batched_noise_rows_match_single_sample_draws(batch):
    """With B>1 each row's noise is its own seed's B=1 draw."""
    _prepare_primary_stream(batch)
    controller = batch.rollout_controllers[PRIMARY_STREAM]
    singles = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    for _ in range(2):
        expected = torch.cat([torch.randn(1, 4, 2, generator=g) for g in singles])
        actual = controller._rollout_variance_noise(batch, batch.latents)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_generators_fill_cuda_noise_for_each_sample(batch):
    _prepare_primary_stream(batch)
    controller = batch.rollout_controllers[PRIMARY_STREAM]
    expected = torch.cat(
        [
            torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(10 + i))
            for i in range(2)
        ]
    )
    actual = controller._rollout_variance_noise(batch, batch.latents.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_generators_fill_cuda_noise_in_place(batch, monkeypatch):
    """A "cuda" generator has no index, so it never compares equal to cuda:0."""
    batch.generator = [torch.Generator("cuda").manual_seed(10 + i) for i in range(2)]
    _prepare_primary_stream(batch)
    controller = batch.rollout_controllers[PRIMARY_STREAM]
    expected = torch.cat(
        [
            torch.randn(
                1,
                4,
                2,
                device="cuda",
                generator=torch.Generator("cuda").manual_seed(10 + i),
            )
            for i in range(2)
        ]
    )
    monkeypatch.setattr(
        torch.Tensor,
        "copy_",
        lambda *args, **kwargs: pytest.fail("same-device noise must not be copied"),
    )
    actual = controller._rollout_variance_noise(batch, batch.latents.cuda())
    monkeypatch.undo()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "stage_cls,denoise_method_name",
    [(DenoisingStage, "_denoise"), (Cosmos3DenoisingStage, "_denoise_once")],
)
def test_stage_cleanup_allows_retry(batch, stage_cls, denoise_method_name):
    """A failed stage run leaves no rollout state, so a retry can prepare again."""
    stage = object.__new__(stage_cls)
    stage._offload_for_torch_compile_warmup = lambda batch: nullcontext()
    stage.use_declared_component = lambda **kwargs: nullcontext()
    stage.transformer = Mock()
    fail = True

    def denoise(batch, args):
        _prepare_primary_stream(batch)
        batch._rollout_denoising_env_state = {"env": None}
        batch.rollout_trajectory_data = object()
        if fail:
            raise RuntimeError("model failed")
        return batch

    setattr(stage, denoise_method_name, denoise)
    with pytest.raises(RuntimeError, match="model failed"):
        stage.forward(batch, None)
    assert batch.rollout_controllers == {}
    assert batch._rollout_denoising_env_state is None
    assert batch.rollout_trajectory_data is None
    fail = False
    assert stage.forward(batch, None) is batch
    assert batch.rollout_controllers == {}


def test_requested_stream_the_stage_never_prepared_fails_the_request(batch):
    """A requested stream must not silently vanish from the response."""
    batch.rollout_extra_streams = ("action",)
    with pytest.raises(RuntimeError, match="were not prepared"):
        with rollout_scope(batch):
            _prepare_primary_stream(batch)
            publish_rollout_if_enabled(batch=batch, pipeline_config=_PIPELINE_CONFIG)


@pytest.mark.parametrize("return_env", [True, False])
def test_published_env_is_the_gathered_conditioning_at_the_loop_end(batch, return_env):
    """The env is read again at publish and passed through the config's SP gather."""
    batch.rollout_extra_streams = ()
    batch.rollout_return_denoising_env = return_env
    pos_cond_kwargs = {"kept": torch.ones(2), "dropped": torch.ones(2)}
    neg_cond_kwargs = {"kept": torch.zeros(2), "dropped": torch.zeros(2)}
    pipeline_config = SimpleNamespace(
        shard_latents_for_sp=_PIPELINE_CONFIG.shard_latents_for_sp,
        gather_denoising_env_static_for_sp=lambda batch, cond_kwargs: {
            key: value for key, value in cond_kwargs.items() if key != "dropped"
        },
    )
    with rollout_scope(batch):
        _prepare_primary_stream(batch)
        init_rollout_env_if_enabled(
            batch=batch,
            image_kwargs={},
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            guidance=None,
        )
        pos_cond_kwargs["added_in_the_loop"] = torch.ones(2)
        neg_cond_kwargs["added_in_the_loop"] = torch.zeros(2)
        for timestep in batch.timesteps:
            batch.latents = batch.scheduler.step(
                torch.zeros_like(batch.latents),
                timestep,
                batch.latents,
                **rollout_step_kwargs(batch, stream=PRIMARY_STREAM),
                return_dict=False,
            )[0]
        _finalize_stream(batch, PRIMARY_STREAM, batch.latents)
        publish_rollout_if_enabled(batch=batch, pipeline_config=pipeline_config)

    env = batch.rollout_trajectory_data.denoising_env
    if not return_env:
        assert env is None
        return
    assert set(env.pos_cond_kwargs) == {"kept", "added_in_the_loop"}
    assert set(env.neg_cond_kwargs) == {"kept", "added_in_the_loop"}


def test_extra_stream_needs_a_flow_match_euler_scheduler(batch):
    _prepare_primary_stream(batch)
    action_latents = batch.audio_latents
    with pytest.raises(ValueError, match="FlowMatch Euler"):
        prepare_rollout_if_enabled(
            batch=batch,
            stream="action",
            scheduler=SimpleNamespace(sigmas=batch.scheduler.sigmas),
            timesteps=batch.scheduler.timesteps,
            latents=action_latents,
            shard_for_sp=_identity_for_sp,
            gather_for_sp=_identity_for_sp,
        )
