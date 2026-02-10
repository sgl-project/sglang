# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Literal, TypedDict

import torch


class RolloutRequest(TypedDict, total=False):
    enabled: bool
    mode: Literal["logprob_rollout"]
    adapter: str
    strict: bool
    params: dict[str, Any]


class RolloutMetadata(TypedDict, total=False):
    timesteps: torch.Tensor
    latents: torch.Tensor
    next_latents: torch.Tensor
    log_prob_old: torch.Tensor
    prev_latents_mean: torch.Tensor
