# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
"""Flow-matching solvers for FLUX 3 Action over a dict of noised streams.

Adapted from the FLUX Action reference implementation
(``flux_action/inference/sampling.py``):

* ``cosmos_unipc``: Cosmos UniPC (order 2, bh2, predict-x0) on SGLang's
  ``FlowUniPCMultistepScheduler``, one denoiser call per step; the model
  receives integer ticks as ``tick / 1000``.
* ``euler``: rectified-flow Euler on the rationally shifted schedule.

The rectified flow convention is ``x_t = t * eps + (1 - t) * x0`` with the
velocity target ``eps - x0``. Solver state stays fp32.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)

Samples = dict[str, torch.Tensor]
Predictor = Callable[[Samples, float], Samples]
NUM_TRAIN_TIMESTEPS = 1000


def timeshift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def euler(
    samples: Samples, predict: Predictor, *, n_steps: int, shift: float
) -> Samples:
    timesteps = timeshift(shift, torch.linspace(1.0, 0.0, n_steps + 1)).tolist()
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        velocity = predict(samples, t_curr)
        samples = {
            k: (samples[k].float() + (t_prev - t_curr) * velocity[k].float()).to(
                samples[k].dtype
            )
            for k in samples
        }
    return samples


def cosmos_unipc(
    samples: Samples, predict: Predictor, *, n_steps: int, shift: float
) -> Samples:
    """Cosmos UniPC: order 2, bh2, predict-x0, one scheduler state per stream.

    The schedule shift is applied in ``set_timesteps`` only (``shift=1.0`` at
    construction), giving the reference grid ``linspace(0.999, 0, N + 1)``
    shifted and truncated to integer ticks.
    """
    device = next(iter(samples.values())).device
    template = FlowUniPCMultistepScheduler(
        solver_order=2,
        solver_type="bh2",
        predict_x0=True,
        lower_order_final=True,
        final_sigmas_type="zero",
        shift=1.0,
    )
    template.set_timesteps(n_steps, device=device, shift=shift)
    schedulers = {k: copy.deepcopy(template) for k in samples}
    for scheduler in schedulers.values():
        # Ticks can repeat at high step counts; index by step, not by tick.
        scheduler.set_begin_index(0)
    for tick in template.timesteps:
        # The reference feeds float32(tick) / 1000 to the model.
        t = torch.tensor(float(tick), dtype=torch.float32) / NUM_TRAIN_TIMESTEPS
        velocity = predict(samples, t.item())
        samples = {
            k: schedulers[k].step(velocity[k], tick, samples[k], return_dict=False)[0]
            for k in samples
        }
    return samples


SAMPLERS = {"cosmos_unipc": cosmos_unipc, "euler": euler}
