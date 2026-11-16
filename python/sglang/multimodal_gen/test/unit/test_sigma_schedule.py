# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from diffusers import EulerDiscreteScheduler

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)


@pytest.mark.parametrize(
    "cls", [FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler]
)
@pytest.mark.parametrize("schedule", ["karras", "exponential", "beta"])
@pytest.mark.parametrize(
    "bounds",
    [
        {},
        {"sigma_min": None, "sigma_max": None},
        {"sigma_min": 0.02},
        {"sigma_max": 0.7},
    ],
)
def test_sigma_schedule_conversion_and_steps(cls, schedule, bounds):
    scheduler = cls(**{f"use_{schedule}_sigmas": True})
    scheduler.register_to_config(**bounds)
    reference = EulerDiscreteScheduler()
    reference.register_to_config(**bounds)
    methods = lambda s: {
        "karras": s._convert_to_karras,
        "exponential": s._convert_to_exponential,
        "beta": s._convert_to_beta,
    }
    for steps in (1, 7):
        sigmas = torch.linspace(0.9, 0.01, 16)
        actual = methods(scheduler)[schedule](sigmas, steps)
        expected = methods(reference)[schedule](sigmas, steps)
        np.testing.assert_array_equal(actual, expected)
    outputs = []
    for _ in range(2):
        scheduler.set_timesteps(7)
        assert torch.isfinite(scheduler.sigmas).all()
        assert (scheduler.sigmas[1:] <= scheduler.sigmas[:-1]).all()
        sample = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4)
        for timestep in scheduler.timesteps:
            sample = scheduler.step(sample * 0.1, timestep, sample).prev_sample
        assert torch.isfinite(sample).all()
        outputs.append(sample)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
