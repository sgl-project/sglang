import math

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchScheduler

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_ltx2_flow_match import (
    LTX2FlowMatchScheduler,
    LTX2FlowMatchSchedulerOutput,
)


def test_custom_sigmas_build_timesteps_and_reset_indices():
    scheduler = LTX2FlowMatchScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
    )
    scheduler._step_index = 4
    scheduler._begin_index = 2

    scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.25], device="cpu")

    assert scheduler.num_inference_steps == 3
    assert torch.allclose(
        scheduler.timesteps, torch.tensor([1000.0, 500.0, 250.0])
    )
    assert torch.allclose(
        scheduler.sigmas, torch.tensor([1.0, 0.5, 0.25, 0.0])
    )
    assert scheduler.step_index is None
    assert scheduler.begin_index is None


def test_dynamic_shifting_uses_ltx_fp32_ndarray_path():
    scheduler = LTX2FlowMatchScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
    )
    steps = 4
    mu = 0.7

    scheduler.set_timesteps(num_inference_steps=steps, mu=mu, device="cpu")

    base_sigmas = np.linspace(
        scheduler._sigma_to_t(scheduler.sigma_max),
        scheduler._sigma_to_t(scheduler.sigma_min),
        steps,
    ).astype(np.float32)
    base_sigmas = base_sigmas / scheduler.config.num_train_timesteps
    expected = (
        math.exp(mu)
        / (
            math.exp(mu)
            + (1 / torch.from_numpy(base_sigmas).to(torch.float32) - 1) ** 1.0
        )
    ).numpy()

    assert scheduler.timesteps.shape == (steps,)
    assert scheduler.sigmas.shape == (steps + 1,)
    assert scheduler.timesteps.dtype == torch.float32
    assert scheduler.sigmas.dtype == torch.float32
    assert torch.allclose(scheduler.sigmas[:-1], torch.from_numpy(expected))
    assert scheduler.sigmas[-1].item() == 0.0

    shifted = scheduler._time_shift_exponential(mu, 1.0, base_sigmas)
    assert isinstance(shifted, np.ndarray)
    assert shifted.dtype == np.float32
    assert np.allclose(shifted, expected)


def test_step_euler_update_advances_step_index():
    scheduler = LTX2FlowMatchScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(sigmas=[1.0, 0.6, 0.2], device="cpu")

    sample = torch.tensor([[[[1.0, -1.0], [0.5, -0.5]]]], dtype=torch.float32)
    model_output = torch.tensor(
        [[[[0.25, -0.5], [1.0, -1.5]]]], dtype=torch.float32
    )

    output = scheduler.step(model_output, scheduler.timesteps[0], sample)

    expected = sample + (scheduler.sigmas[1] - scheduler.sigmas[0]) * model_output
    assert isinstance(output, LTX2FlowMatchSchedulerOutput)
    assert torch.allclose(output.prev_sample, expected)
    assert scheduler.step_index == 1

    next_output = scheduler.step(
        model_output,
        scheduler.timesteps[1],
        output.prev_sample,
        return_dict=False,
    )[0]
    next_expected = expected + (scheduler.sigmas[2] - scheduler.sigmas[1]) * model_output
    assert torch.allclose(next_output, next_expected)
    assert scheduler.step_index == 2


def test_rollout_mixin_methods_are_available():
    scheduler = LTX2FlowMatchScheduler()

    assert callable(scheduler.prepare_rollout)
    assert callable(scheduler.flow_sde_sampling)
    assert callable(scheduler.collect_rollout_log_probs)
    assert not isinstance(scheduler, DiffusersFlowMatchScheduler)
