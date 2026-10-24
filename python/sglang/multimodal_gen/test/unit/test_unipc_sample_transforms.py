# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)


@pytest.mark.parametrize("cls", [FlowUniPCMultistepScheduler, UniPCMultistepScheduler])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unipc_sample_transforms_and_state(cls, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    scheduler = cls(dynamic_thresholding_ratio=0.5, sample_max_value=5.0)
    restored = cls.from_config(scheduler.config)
    assert dict(restored.config) == dict(scheduler.config)
    x = torch.tensor([-8, -2, 0, 2, 4, 20], device=device, dtype=dtype)
    x = x.repeat(2).reshape(2, 1, 2, 3).transpose(-1, -2)
    original = x.clone()
    expected = (x.float().clamp(-3, 3) / 3).to(dtype)
    torch.testing.assert_close(scheduler._threshold_sample(x), expected, rtol=0, atol=0)
    assert scheduler.scale_model_input(x, 0) is x

    scheduler.timesteps = torch.tensor([90, 60, 60, 10], device=device)
    scheduler.sigmas = torch.tensor([0.9, 0.6, 0.4, 0.1, 0], device=device)
    timesteps = torch.tensor([60, 10])
    noise = torch.ones_like(x)
    for mode, indices in (
        ("training", [2, 3]),
        ("img2img", [1, 1]),
        ("inpaint", [2, 2]),
    ):
        if mode != "training":
            scheduler.set_begin_index(1)
        if mode == "inpaint":
            scheduler._step_index = 2
        state = scheduler.begin_index, scheduler.step_index
        sigma = scheduler.sigmas.to(dtype)[indices].reshape(2, 1, 1, 1)
        if cls is FlowUniPCMultistepScheduler:
            alpha = 1 - sigma
        else:
            alpha = 1 / ((sigma**2 + 1) ** 0.5)
            sigma = sigma * alpha
        expected = alpha * x + sigma * noise
        torch.testing.assert_close(
            scheduler.add_noise(x, noise, timesteps), expected, rtol=0, atol=0
        )
        assert (scheduler.begin_index, scheduler.step_index) == state
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    scheduler.set_timesteps(5, device=device)
    assert scheduler.begin_index is None and scheduler.step_index is None
    scheduler._init_step_index(scheduler.timesteps[0])
    assert scheduler.step_index == 0
    scheduler.set_begin_index(2)
    scheduler._init_step_index(scheduler.timesteps[0])
    assert scheduler.step_index == 2


def test_unipc_missing_timestep_behavior_is_preserved():
    regular, flow = UniPCMultistepScheduler(), FlowUniPCMultistepScheduler()
    regular.set_timesteps(5)
    flow.set_timesteps(5)
    assert regular.index_for_timestep(-1) == 4
    with pytest.raises(IndexError):
        flow.index_for_timestep(-1)
