# SPDX-License-Identifier: Apache-2.0
"""Rollout of named latent streams: the per-stream hooks, per-sample noise, LTX-2 audio wiring, and the extra_streams response."""

import asyncio
import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from fastapi import HTTPException

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.post_training.rl_rollout import (
    AUDIO_STREAM,
    RLRolloutArgs,
)
from sglang.multimodal_gen.configs.sample.ltx_2 import (
    LTX2SamplingParams,
    LTX23SamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_deserialize,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2Pipeline,
    LTX2TwoStagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2 import (
    denoising as ltx2_denoising,
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


def _prepare_both_streams(batch):
    """What the LTX-2 loop does before SP sharding; returns the audio scheduler."""
    audio_scheduler = copy.deepcopy(batch.scheduler)
    _prepare_primary_stream(batch)
    prepare_rollout_if_enabled(
        batch=batch,
        stream=AUDIO_STREAM,
        scheduler=audio_scheduler,
        timesteps=audio_scheduler.timesteps,
        latents=batch.audio_latents,
        shard_for_sp=ltx2_denoising._require_replicated_audio,
        gather_for_sp=ltx2_denoising._require_replicated_audio,
    )
    return audio_scheduler


def _step_both_streams(batch, audio_scheduler):
    for timestep in batch.timesteps:
        batch.latents = batch.scheduler.step(
            torch.zeros_like(batch.latents),
            timestep,
            batch.latents,
            **rollout_step_kwargs(batch, stream=PRIMARY_STREAM),
            return_dict=False,
        )[0]
        batch.audio_latents = audio_scheduler.step(
            torch.zeros_like(batch.audio_latents),
            timestep,
            batch.audio_latents,
            **rollout_step_kwargs(batch, stream=AUDIO_STREAM),
            return_dict=False,
        )[0]


def _step_both_streams_and_publish(batch, audio_scheduler):
    _step_both_streams(batch, audio_scheduler)
    _finalize_stream(batch, PRIMARY_STREAM, batch.latents)
    _finalize_stream(batch, AUDIO_STREAM, batch.audio_latents)
    publish_rollout_if_enabled(batch=batch, pipeline_config=_PIPELINE_CONFIG)


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


def test_audio_step_does_not_advance_the_request_generators(batch):
    with rollout_scope(batch):
        audio_scheduler = _prepare_both_streams(batch)
        request_states = [g.get_state() for g in batch.generator]
        audio_scheduler.step(
            torch.zeros_like(batch.audio_latents),
            batch.timesteps[0],
            batch.audio_latents,
            **rollout_step_kwargs(batch, stream=AUDIO_STREAM),
            return_dict=False,
        )
        for request_generator, state in zip(batch.generator, request_states):
            torch.testing.assert_close(
                request_generator.get_state(), state, rtol=0, atol=0
            )


def _coupled_rollout(batch, overrides):
    """Roll out both streams with model outputs that read the other stream's latents, as LTX-2's audio-video cross-attention does."""
    batch.rollout_extra_stream_sde_args = overrides
    batch.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    batch.scheduler.set_timesteps(sigmas=[0.8, 0.4])
    batch.timesteps = batch.scheduler.timesteps
    batch.latents = torch.zeros(2, 4, 2)
    batch.audio_latents = torch.ones(2, 5, 3)
    batch.generator = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    with rollout_scope(batch):
        audio_scheduler = _prepare_both_streams(batch)
        sde_kernels = {
            stream: (
                controller.sde_type,
                controller.noise_level,
                controller.log_prob_no_const,
                controller.sde_step_indices,
            )
            for stream, controller in batch.rollout_controllers.items()
        }
        for timestep in batch.timesteps:
            primary_output = batch.latents + batch.audio_latents.mean()
            audio_output = batch.audio_latents - batch.latents.mean()
            batch.latents = batch.scheduler.step(
                primary_output,
                timestep,
                batch.latents,
                **rollout_step_kwargs(batch, stream=PRIMARY_STREAM),
                return_dict=False,
            )[0]
            batch.audio_latents = audio_scheduler.step(
                audio_output,
                timestep,
                batch.audio_latents,
                **rollout_step_kwargs(batch, stream=AUDIO_STREAM),
                return_dict=False,
            )[0]
        _finalize_stream(batch, PRIMARY_STREAM, batch.latents)
        _finalize_stream(batch, AUDIO_STREAM, batch.audio_latents)
        publish_rollout_if_enabled(batch=batch, pipeline_config=_PIPELINE_CONFIG)
    generator_states = [g.get_state() for g in batch.generator]
    return (
        batch.rollout_trajectory_data.streams[PRIMARY_STREAM],
        sde_kernels,
        generator_states,
    )


@pytest.mark.parametrize(
    "top_level,audio_override",
    [
        (("sde", 0.7, False), ("ode", 0.0, True)),
        (("sde", 0.7, False), ("cps", 0.3, True)),
        (("cps", 0.5, False), ("sde", 0.2, True)),
    ],
)
def test_audio_override_reaches_only_the_audio_controller(
    batch, top_level, audio_override
):
    """An audio override sets the audio SDE kernel only: the primary keeps the top-level SDE kernel, its noise and its log-probs, though its trajectory follows the changed audio."""
    names = ("rollout_sde_type", "rollout_noise_level", "rollout_log_prob_no_const")
    for name, value in zip(names, top_level):
        setattr(batch, name, value)
    batch.rollout_debug_mode = True
    # Step 0 is stochastic, so the audio SDE kernel changes the latents the primary's step 1 reads.
    batch.rollout_sde_step_indices = [0]
    overrides = {AUDIO_STREAM: dict(zip(names, audio_override))}
    sent_overrides = copy.deepcopy(overrides)

    inherited, inherited_sde_kernels, inherited_states = _coupled_rollout(batch, {})
    overridden, sde_kernels, states = _coupled_rollout(batch, overrides)

    assert overrides == sent_overrides
    assert sde_kernels[PRIMARY_STREAM] == (*top_level, [0])
    assert sde_kernels[AUDIO_STREAM] == (*audio_override, [0])
    assert inherited_sde_kernels[AUDIO_STREAM] == (*top_level, [0])
    for state, inherited_state in zip(states, inherited_states):
        torch.testing.assert_close(state, inherited_state, rtol=0, atol=0)
    torch.testing.assert_close(
        overridden.rollout_log_probs, inherited.rollout_log_probs, rtol=0, atol=0
    )
    for name in ("rollout_variance_noises", "rollout_noise_std_devs"):
        torch.testing.assert_close(
            getattr(overridden.rollout_debug_tensors, name),
            getattr(inherited.rollout_debug_tensors, name),
            rtol=0,
            atol=0,
        )
    assert not torch.equal(
        overridden.dit_trajectory.latents, inherited.dit_trajectory.latents
    )


@pytest.mark.parametrize("stop_step_index,kept_steps", [(None, [0, 2]), (0, [0, 1])])
def test_audio_rollout_reaches_the_response_through_select_and_merge(
    batch, stop_step_index, kept_steps
):
    """Audio runs the primary's, possibly stopped, schedule; its rollout and the settings its controller used survive per-output select and merge and are returned under extra_streams."""
    batch.rollout_stop_step_index = stop_step_index
    batch.rollout_return_step_indices = kept_steps
    audio_override = {
        "rollout_sde_type": "cps",
        "rollout_noise_level": 0.3,
        "rollout_log_prob_no_const": True,
    }
    batch.rollout_extra_stream_sde_args = {AUDIO_STREAM: audio_override}
    with rollout_scope(batch):
        audio_scheduler = _prepare_both_streams(batch)
        primary_controller = batch.rollout_controllers[PRIMARY_STREAM]
        audio_controller = batch.rollout_controllers[AUDIO_STREAM]
        torch.testing.assert_close(
            audio_controller.sigmas, primary_controller.sigmas, rtol=0, atol=0
        )
        assert audio_controller.sigma_max == primary_controller.sigma_max
        _step_both_streams_and_publish(batch, audio_scheduler)
    data = batch.rollout_trajectory_data
    num_steps = 2 if stop_step_index is None else 1
    for stream_data, final_latents in (
        (data.streams[PRIMARY_STREAM], batch.latents),
        (data.streams[AUDIO_STREAM], batch.audio_latents),
    ):
        assert stream_data.dit_trajectory.latent_step_indices.tolist() == kept_steps
        assert stream_data.rollout_log_probs.shape == (2, num_steps)
        assert torch.isfinite(stream_data.rollout_log_probs).all()
        torch.testing.assert_close(
            stream_data.dit_trajectory.latents[:, -1], final_latents
        )
    per_output = [select_output_rollout_trajectory(data, i) for i in range(2)]
    result = OutputBatch(
        output=torch.zeros(2, 1),
        audio=torch.arange(16).view(2, 8).float(),
        audio_sample_rate=24000,
        rollout_trajectory_data=concat_rollout_trajectory_data(per_output),
    )
    responses = rollout_api._build_response("r", "p", 10, True, result)
    for i, response in enumerate(responses):
        audio_payload = _maybe_deserialize(response.extra_streams[AUDIO_STREAM])
        torch.testing.assert_close(
            audio_payload["rollout_log_probs"],
            data.streams[AUDIO_STREAM].rollout_log_probs[i],
        )
        dit_trajectory = audio_payload["dit_trajectory"]
        torch.testing.assert_close(
            dit_trajectory["latents"],
            data.streams[AUDIO_STREAM].dit_trajectory.latents[i],
        )
        torch.testing.assert_close(
            dit_trajectory["timesteps"],
            data.streams[AUDIO_STREAM].dit_trajectory.timesteps,
        )
        torch.testing.assert_close(
            dit_trajectory["sigmas"], data.streams[AUDIO_STREAM].dit_trajectory.sigmas
        )
        assert dit_trajectory["latent_step_indices"].tolist() == kept_steps
        torch.testing.assert_close(audio_payload["generated_output"], result.audio[i])
        assert audio_payload["sample_rate"] == 24000
        assert audio_payload["rollout_sde_kernel"] == audio_override


def test_sp_sharded_audio_fails_at_the_final_gather_of_an_ode_rollout(batch):
    """An ODE rollout draws no noise, so only the gather can catch SP-sharded audio latents."""
    batch.rollout_sde_type = "ode"
    batch.rollout_log_prob_no_const = True
    with pytest.raises(RuntimeError, match="replicated across SP ranks"):
        with rollout_scope(batch):
            audio_scheduler = _prepare_both_streams(batch)
            batch.did_sp_shard_audio_latents = True
            _step_both_streams_and_publish(batch, audio_scheduler)
    assert batch.rollout_trajectory_data is None


def test_audio_not_requested_for_rollout_stops_with_the_primary_and_returns_only_its_waveform(
    batch,
):
    """Without "audio" in rollout_extra_streams the audio stream has no controller and no rollout fields; its scheduler still stops and its waveform is returned."""
    batch.rollout_extra_streams = ()
    batch.rollout_stop_step_index = 0
    batch.rollout_return_step_indices = [0, 1]
    with rollout_scope(batch):
        audio_scheduler = _prepare_both_streams(batch)
        assert AUDIO_STREAM not in batch.rollout_controllers
        torch.testing.assert_close(
            audio_scheduler.sigmas,
            batch.rollout_controllers[PRIMARY_STREAM].sigmas,
            rtol=0,
            atol=0,
        )
        _step_both_streams_and_publish(batch, audio_scheduler)
    data = batch.rollout_trajectory_data
    assert set(data.streams) == {PRIMARY_STREAM}
    audio = torch.randn(2, 8)
    result = OutputBatch(
        output=torch.zeros(2, 3),
        audio=audio,
        audio_sample_rate=24000,
        rollout_trajectory_data=data,
    )
    for rollout in (True, False):
        responses = rollout_api._build_response("r", "p", 10, rollout, result)
        for i, response in enumerate(responses):
            audio_payload = _maybe_deserialize(response.extra_streams[AUDIO_STREAM])
            torch.testing.assert_close(audio_payload["generated_output"], audio[i])
            assert audio_payload["sample_rate"] == 24000
            for key in (
                "rollout_log_probs",
                "rollout_debug_tensors",
                "dit_trajectory",
                "rollout_sde_kernel",
            ):
                assert audio_payload[key] is None


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
    responses = rollout_api._build_response("r", "p", 10, True, result, stored=stored)
    assert any(key.startswith(f"extra_streams.{stream}.") for key in stored)
    # Without an override the stream echoes the top-level settings it inherited.
    assert responses[0].extra_streams[stream]["rollout_sde_kernel"] == {
        "rollout_sde_type": "sde",
        "rollout_noise_level": 0.7,
        "rollout_log_prob_no_const": False,
    }


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


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({PRIMARY_STREAM: {}}, "not in rollout_extra_streams"),
        ({"audio": "ode"}, "must be a dict"),
        ({"audio": {"rollout_sde_step_indices": [0]}}, "may set only"),
        ({"audio": {"rollout_sde_type": "SDE"}}, "rollout_sde_type must be one of"),
        ({"audio": {"rollout_noise_level": True}}, "finite non-negative number"),
        ({"audio": {"rollout_log_prob_no_const": "true"}}, "must be a bool"),
        (
            {"audio": {"rollout_sde_type": "ode"}},
            r"rollout_sde_type='ode' \(override\).*rollout_log_prob_no_const=False "
            r"\(inherited from the top level\)",
        ),
        (
            {"audio": {"rollout_noise_level": 0}},
            r"needs rollout_noise_level > 0, got rollout_noise_level=0 \(override\)",
        ),
    ],
)
def test_invalid_extra_stream_overrides_are_rejected_up_front(overrides, error):
    """An override is checked before any model work, as sent and against the rules the stream's controller would assert at its first step."""
    kwargs = {
        "rollout": True,
        "rollout_extra_streams": (AUDIO_STREAM,),
        "rollout_extra_stream_sde_args": overrides,
    }
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


def _assert_rejected(request, forward, *, match):
    with pytest.raises(HTTPException) as error:
        asyncio.run(rollout_api.rollout_generate(request))
    assert error.value.status_code == 400
    assert match in error.value.detail
    forward.assert_not_awaited()


def _patch_served_pipeline_and_forward(
    monkeypatch, model_pipeline, override, override_pipeline
):
    args = SimpleNamespace(
        pipeline_class_name=override,
        model_path="test",
        model_id=None,
        backend="sglang",
        attention_backend_config={},
        enable_trace=False,
    )
    monkeypatch.setattr(
        registry,
        "get_model_info",
        lambda *a, **kw: SimpleNamespace(pipeline_cls=model_pipeline),
    )
    monkeypatch.setattr(
        registry,
        "get_pipeline_class",
        lambda name: {override: override_pipeline}[name],
    )
    monkeypatch.setattr(rollout_api, "get_global_server_args", lambda: args)
    monkeypatch.setattr(
        rollout_api,
        "build_sampling_params",
        lambda rid, **kw: LTX23SamplingParams(**kw),
    )
    forward = AsyncMock(return_value=OutputBatch(output=torch.zeros(1, 1)))
    monkeypatch.setattr(rollout_api.async_scheduler_client, "forward", forward)
    monkeypatch.setattr(rollout_api, "_build_response", lambda *a, **kw: [])
    return forward


@pytest.mark.parametrize(
    "model_pipeline,override,override_pipeline,error",
    [
        (LTX2Pipeline, None, None, None),
        (LTX2TwoStagePipeline, "LTX2Pipeline", LTX2Pipeline, None),
        (
            LTX2Pipeline,
            "LTX2TwoStagePipeline",
            LTX2TwoStagePipeline,
            "does not support rollout of streams ['audio']",
        ),
    ],
)
def test_http_checks_extra_streams_against_the_served_pipeline(
    monkeypatch, model_pipeline, override, override_pipeline, error
):
    """The served pipeline is the --pipeline-class-name override when set, else the model's."""
    forward = _patch_served_pipeline_and_forward(
        monkeypatch, model_pipeline, override, override_pipeline
    )
    request = RolloutRequest(
        prompt="test", seed=42, rollout_extra_streams=[AUDIO_STREAM]
    )
    if error is None:
        assert asyncio.run(rollout_api.rollout_generate(request)).status_code == 200
        forward.assert_awaited_once()
    else:
        _assert_rejected(request, forward, match=error)


def test_http_rejects_extra_streams_without_rollout(monkeypatch):
    forward = _patch_served_pipeline_and_forward(monkeypatch, LTX2Pipeline, None, None)
    request = RolloutRequest(
        prompt="test", seed=42, rollout=False, rollout_extra_streams=[AUDIO_STREAM]
    )
    _assert_rejected(
        request, forward, match="rollout_extra_streams requires rollout=True"
    )


def test_http_rejects_an_override_for_a_stream_it_does_not_roll_out(monkeypatch):
    """A misspelled stream name must not be dropped, leaving audio on the primary's SDE kernel."""
    forward = _patch_served_pipeline_and_forward(monkeypatch, LTX2Pipeline, None, None)
    request = RolloutRequest(
        prompt="test",
        seed=42,
        rollout_extra_streams=[AUDIO_STREAM],
        rollout_extra_stream_sde_args={"audoi": {"rollout_noise_level": 0.3}},
    )
    _assert_rejected(request, forward, match="not in rollout_extra_streams")


def test_audio_rollout_requires_generated_audio():
    with pytest.raises(ValueError, match="generate_audio=True"):
        LTX2SamplingParams(
            prompt="test",
            rollout=True,
            rollout_extra_streams=(AUDIO_STREAM,),
            generate_audio=False,
        )


def test_primary_only_rollout_does_not_require_generated_audio():
    LTX2SamplingParams(prompt="test", rollout=True, generate_audio=False)


def test_http_rejects_audio_rollout_without_generated_audio(monkeypatch):
    """``extra_sampling_params`` is the only way a rollout request sets ``generate_audio``."""
    forward = _patch_served_pipeline_and_forward(monkeypatch, LTX2Pipeline, None, None)
    request = RolloutRequest(
        prompt="test",
        seed=42,
        rollout_extra_streams=[AUDIO_STREAM],
        extra_sampling_params={"generate_audio": False},
    )
    _assert_rejected(request, forward, match="generate_audio=True")


def test_requested_stream_the_stage_never_prepared_fails_the_request(batch):
    """A requested stream must not silently vanish from the response."""
    batch.rollout_extra_streams = ("action",)
    with pytest.raises(RuntimeError, match="were not prepared"):
        with rollout_scope(batch):
            _prepare_primary_stream(batch)
            publish_rollout_if_enabled(batch=batch, pipeline_config=_PIPELINE_CONFIG)


def test_requested_stream_the_stage_never_finalized_fails_the_request(batch):
    """Collecting a stream does not publish; publishing needs every requested stream."""
    with pytest.raises(RuntimeError, match="were not finalized"):
        with rollout_scope(batch):
            audio_scheduler = _prepare_both_streams(batch)
            _step_both_streams(batch, audio_scheduler)
            _finalize_stream(batch, PRIMARY_STREAM, batch.latents)
            assert batch.rollout_trajectory_data is None
            publish_rollout_if_enabled(batch=batch, pipeline_config=_PIPELINE_CONFIG)
    assert batch.rollout_trajectory_data is None


def test_stream_cannot_be_finalized_twice(batch):
    with rollout_scope(batch):
        audio_scheduler = _prepare_both_streams(batch)
        _step_both_streams(batch, audio_scheduler)
        _finalize_stream(batch, AUDIO_STREAM, batch.audio_latents)
        with pytest.raises(RuntimeError, match="was already finalized"):
            _finalize_stream(batch, AUDIO_STREAM, batch.audio_latents)


def test_ltx2_collects_audio_before_post_processing_and_publishes_after_it(batch):
    """The stage finalizes its audio stream on the last step's latents, before
    post-processing replaces them, and the generic loop publishes after that."""
    last_step = {}

    def prepare_denoising_loop(batch, server_args):
        audio_scheduler = _prepare_both_streams(batch)
        return SimpleNamespace(
            scheduler=batch.scheduler,
            audio_scheduler=audio_scheduler,
            timesteps=batch.timesteps,
            latents=batch.latents,
            audio_latents=batch.audio_latents,
            image_kwargs={},
            pos_cond_kwargs={},
            neg_cond_kwargs={},
            guidance=None,
            trajectory_latents=[],
            trajectory_timesteps=[],
            trajectory_audio_latents=[],
            is_warmup=False,
        )

    def run_denoising_loop(ctx, batch, server_args):
        for timestep in ctx.timesteps:
            ctx.latents = ctx.scheduler.step(
                torch.zeros_like(ctx.latents),
                timestep,
                ctx.latents,
                **rollout_step_kwargs(batch, stream=PRIMARY_STREAM),
                return_dict=False,
            )[0]
            ctx.audio_latents = ctx.audio_scheduler.step(
                torch.zeros_like(ctx.audio_latents),
                timestep,
                ctx.audio_latents,
                **rollout_step_kwargs(batch, stream=AUDIO_STREAM),
                return_dict=False,
            )[0]
        ctx.audio_latents = torch.full_like(ctx.audio_latents, 7.0)
        last_step.update(primary=ctx.latents, audio=ctx.audio_latents)
        return ctx.timesteps.cpu()

    def post_denoising_loop(**kwargs):
        batch.latents = torch.full_like(batch.latents, -1.0)
        batch.audio_latents = torch.full_like(batch.audio_latents, -1.0)

    stage = object.__new__(ltx2_denoising.LTX2DenoisingStage)
    stage._prepare_denoising_loop = prepare_denoising_loop
    stage._run_denoising_loop = run_denoising_loop
    stage._post_denoising_loop = post_denoising_loop
    with rollout_scope(batch):
        stage._denoise(batch, SimpleNamespace(pipeline_config=_PIPELINE_CONFIG))

    streams = batch.rollout_trajectory_data.streams
    assert set(streams) == {PRIMARY_STREAM, AUDIO_STREAM}
    for stream, expected in ((PRIMARY_STREAM, "primary"), (AUDIO_STREAM, "audio")):
        torch.testing.assert_close(
            streams[stream].dit_trajectory.latents[:, -1],
            last_step[expected],
            rtol=0,
            atol=0,
        )


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
