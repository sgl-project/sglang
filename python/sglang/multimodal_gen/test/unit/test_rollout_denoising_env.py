"""Unit tests for the rollout denoising env: what ``init_rollout_env_if_enabled``
records and what ``publish_rollout_if_enabled`` returns.

    denoising stage --pos_cond_kwargs, neg_cond_kwargs--> init_rollout_env_if_enabled
        not a rollout, or no env asked for ...... records nothing, checks nothing
        pos required (neg is None without a negative cond); a second record fails
        drop None -> keep the registered args (no registration: keep every arg)
    publish_rollout_if_enabled: SP gather of each recorded branch -> pos / neg of the env;
        an env asked for but never recorded fails

    test_declared_config_keeps_its_env_args_per_branch ... filter, None, pos / neg
    test_undeclared_config_keeps_every_arg ............... no vocabulary, no filter
    test_env_without_conditional_kwargs_fails ............ pos required
    test_env_is_recorded_once ............................ second record fails
    test_requests_without_env_record_nothing ............. serving and env-off requests
    test_env_asked_for_but_not_recorded_fails_publish
    test_env_is_the_recording_from_before_the_loop ....... in-loop writes stay out; SP gather
    test_generic_stage_pushes_its_cfg_branches ........... branch kwargs + guidance, pos / neg;
                                                           neg cond without CFG -> neg
    test_registration_is_inherited_until_one_registers_none  MRO lookup, None, duplicates
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.post_training.denoising_env import (
    denoising_env_args_of,
    register_denoising_env_args,
)
from sglang.multimodal_gen.configs.post_training.rl_rollout import PRIMARY_STREAM
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    finalize_rollout_if_enabled,
    init_rollout_env_if_enabled,
    pipeline_config_sp_fns,
    prepare_rollout_if_enabled,
    publish_rollout_if_enabled,
    rollout_scope,
    rollout_step_kwargs,
)


def _config(env_args=None, gather=None):
    """A pipeline config of a fresh class, registered with ``env_args`` unless None."""

    class Config:
        def shard_latents_for_sp(self, batch, latents):
            return latents, False

        def gather_denoising_env_static_for_sp(self, batch, kwargs):
            return gather(batch, kwargs) if gather is not None else kwargs

    if env_args is not None:
        register_denoising_env_args(env_args)(Config)
    return Config()


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
        lambda: 1,
    )
    req = Req(
        sampling_params=SamplingParams(
            prompt="test", seed=10, rollout=True, rollout_return_denoising_env=True
        )
    )
    req.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    req.scheduler.set_timesteps(sigmas=[0.8, 0.4])
    req.timesteps = req.scheduler.timesteps
    req.latents = torch.zeros(2, 4, 2)
    req.generator = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    return req


def _record(batch, config, pos_cond_kwargs, neg_cond_kwargs=None):
    init_rollout_env_if_enabled(
        batch=batch,
        pos_cond_kwargs=pos_cond_kwargs,
        neg_cond_kwargs=neg_cond_kwargs,
        pipeline_config=config,
    )


def _prepare_primary_stream(batch, config):
    shard_for_sp, gather_for_sp = pipeline_config_sp_fns(config)
    prepare_rollout_if_enabled(
        batch=batch,
        stream=PRIMARY_STREAM,
        scheduler=batch.scheduler,
        timesteps=batch.timesteps,
        latents=batch.latents,
        shard_for_sp=shard_for_sp,
        gather_for_sp=gather_for_sp,
    )


def _run_primary_loop(batch, latents):
    for timestep in batch.timesteps:
        latents = batch.scheduler.step(
            torch.zeros_like(latents),
            timestep,
            latents,
            **rollout_step_kwargs(batch, stream=PRIMARY_STREAM),
            return_dict=False,
        )[0]
    return latents


def _prepare_run_and_publish(batch, config):
    """The primary stream's prepare, loop and finalize, then the publish."""
    _prepare_primary_stream(batch, config)
    finalize_rollout_if_enabled(
        batch=batch,
        stream=PRIMARY_STREAM,
        latents=_run_primary_loop(batch, batch.latents),
        num_inference_steps=len(batch.timesteps),
        final_timestep=torch.tensor(0.0),
    )
    publish_rollout_if_enabled(batch=batch, pipeline_config=config)


def test_declared_config_keeps_its_env_args_per_branch(batch):
    """Only the declared args of each branch reach the env, a None arg is left out,
    and the conditional kwargs fill pos while the unconditional ones fill neg."""
    config = _config(env_args=frozenset({"prompt", "mask"}))
    with rollout_scope(batch):
        _record(
            batch,
            config,
            {"prompt": torch.ones(2), "mask": None, "guidance": 6.0},
            {"prompt": torch.zeros(2), "mask": torch.ones(2)},
        )
        _prepare_run_and_publish(batch, config)
    env = batch.rollout_trajectory_data.denoising_env
    assert set(env.pos_cond_kwargs) == {"prompt"}
    assert set(env.neg_cond_kwargs) == {"prompt", "mask"}
    torch.testing.assert_close(env.pos_cond_kwargs["prompt"], torch.ones(2))
    torch.testing.assert_close(env.neg_cond_kwargs["prompt"], torch.zeros(2))


def test_undeclared_config_keeps_every_arg(batch):
    """A config with no registration returns every arg its stage passed, and no
    unconditional kwargs mean no neg."""
    config = _config()
    with rollout_scope(batch):
        _record(batch, config, {"prompt": torch.ones(2), "unknown": 1})
        _prepare_run_and_publish(batch, config)
    env = batch.rollout_trajectory_data.denoising_env
    assert set(env.pos_cond_kwargs) == {"prompt", "unknown"}
    assert env.neg_cond_kwargs is None


def test_env_without_conditional_kwargs_fails(batch):
    """A stage with no CFG branch cannot give the env a request asks for."""
    with rollout_scope(batch):
        with pytest.raises(ValueError, match="no conditional branch"):
            _record(batch, _config(), None, {"prompt": torch.zeros(2)})


def test_env_is_recorded_once(batch):
    """A second push in one request would silently replace the first."""
    with rollout_scope(batch):
        _record(batch, _config(), {})
        with pytest.raises(RuntimeError, match="already recorded"):
            _record(batch, _config(), {})


@pytest.mark.parametrize("rollout,return_env", [(False, False), (True, False)])
def test_requests_without_env_record_nothing(batch, rollout, return_env):
    """A stage without CFG branches pushes no conditional kwargs on every request;
    only a request that asks for the env may fail on it."""
    batch.rollout = rollout
    batch.rollout_return_denoising_env = return_env
    with rollout_scope(batch):
        _record(batch, _config(env_args=frozenset()), None)
        if rollout:
            assert batch._rollout_denoising_env_state is None


def test_env_asked_for_but_not_recorded_fails_publish(batch):
    with rollout_scope(batch):
        with pytest.raises(RuntimeError, match="recorded none"):
            _prepare_run_and_publish(batch, _config())


def test_env_is_the_recording_from_before_the_loop(batch):
    """Args written into the stage's dict during the loop stay out of the env, and
    each branch goes through the config's SP gather."""
    config = _config(
        gather=lambda batch, kwargs: {k: v for k, v in kwargs.items() if k != "dropped"}
    )
    cond_kwargs = {"kept": torch.ones(2), "dropped": torch.ones(2)}
    with rollout_scope(batch):
        _record(batch, config, cond_kwargs)
        cond_kwargs["filled_in_the_loop"] = torch.ones(2)
        _prepare_run_and_publish(batch, config)
    assert set(batch.rollout_trajectory_data.denoising_env.pos_cond_kwargs) == {"kept"}


@pytest.mark.parametrize(
    "cfg,neg_cond_kwargs",
    [
        (True, {"prompt": torch.zeros(2)}),
        (False, {"prompt": torch.zeros(2)}),
        (False, {}),
    ],
)
def test_generic_stage_pushes_its_cfg_branches(batch, cfg, neg_cond_kwargs):
    """The generic stage records what its forward unpacks: the kwargs the default
    CFGPolicy.build gives each branch plus the guidance its DiT accepts, so a build
    whose branches stop matching the env fails here. A negative cond built without
    CFG still reaches the env as the branch CFG would run."""

    def forward(hidden_states, timestep, prompt=None, guidance=None, **kwargs):
        raise AssertionError("not called")

    batch.do_classifier_free_guidance = cfg
    image_kwargs = {"image": torch.ones(1)}
    pos_cond_kwargs = {"prompt": torch.ones(2)}
    config = _config()

    def prepare_denoising_loop(batch, server_args):
        _prepare_primary_stream(batch, config)
        return SimpleNamespace(
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            guidance=torch.full((2,), 6.0),
            latents=batch.latents,
            timesteps=batch.timesteps,
        )

    def run_denoising_loop(ctx, batch, server_args):
        ctx.latents = _run_primary_loop(batch, ctx.latents)
        return ctx.timesteps.cpu()

    stage = object.__new__(DenoisingStage)
    stage._extra_func_kwarg_names_cache = {}
    stage.transformer = SimpleNamespace(forward=forward)
    stage._prepare_denoising_loop = prepare_denoising_loop
    stage._run_denoising_loop = run_denoising_loop
    stage._finalize_denoising_loop = lambda ctx, batch, server_args: None
    with rollout_scope(batch):
        stage._denoise(batch, SimpleNamespace(pipeline_config=config))

    env = batch.rollout_trajectory_data.denoising_env
    batch.do_classifier_free_guidance = True
    cond, uncond = (
        CFGPolicy()
        .build(batch, image_kwargs, pos_cond_kwargs, neg_cond_kwargs)
        .branches
    )
    _assert_branch_kwargs_and_guidance(env.pos_cond_kwargs, cond.kwargs)
    if not neg_cond_kwargs:
        assert env.neg_cond_kwargs is None
        return
    _assert_branch_kwargs_and_guidance(env.neg_cond_kwargs, uncond.kwargs)


def _assert_branch_kwargs_and_guidance(env_kwargs, branch_kwargs):
    assert set(env_kwargs) == {*branch_kwargs, "guidance"}
    for name, value in branch_kwargs.items():
        torch.testing.assert_close(env_kwargs[name], value)


def test_registration_is_inherited_until_one_registers_none():
    """A config subclass takes its nearest registered base's args, so a Wan variant
    with another DiT must register None to stop inheriting; registering twice fails."""

    @register_denoising_env_args(["prompt"])
    class Base:
        pass

    class Variant(Base):
        pass

    @register_denoising_env_args(None)
    class OtherDit(Variant):
        pass

    class OtherDitVariant(OtherDit):
        pass

    assert denoising_env_args_of(Variant()) == frozenset({"prompt"})
    assert denoising_env_args_of(OtherDitVariant()) is None
    with pytest.raises(ValueError, match="already registered"):
        register_denoising_env_args(["mask"])(Base)
