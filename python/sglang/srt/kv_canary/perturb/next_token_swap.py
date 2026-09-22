"""Swap two requests' sampled next tokens at the sampler exit.

KV path is untouched, so kv_canary KV-side fail_reasons stay silent. The
token-oracle input check downstream MUST report fail_reason=write_token — this
validates that the input-check link is genuinely active.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class NextTokenSwapConfig:
    prob: float
    warmup_steps: int

    @classmethod
    def from_env(cls) -> NextTokenSwapConfig:
        return cls(
            prob=envs.SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB.get(),
            warmup_steps=envs.SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS.get(),
        )


_config: Optional[NextTokenSwapConfig] = None
_step_counter: int = 0


def _get_config() -> NextTokenSwapConfig:
    global _config
    if _config is None:
        _config = NextTokenSwapConfig.from_env()
    return _config


def maybe_perturb_swap_next_tokens(
    batch_next_token_ids: torch.Tensor,
) -> torch.Tensor:
    global _step_counter

    config = _get_config()
    step = _step_counter
    _step_counter += 1

    if config.prob <= 0.0:
        return batch_next_token_ids
    if step < config.warmup_steps:
        return batch_next_token_ids
    if batch_next_token_ids.shape[0] < 2:
        return batch_next_token_ids

    if random.random() >= config.prob:
        return batch_next_token_ids

    batch_size = batch_next_token_ids.shape[0]
    i = random.randrange(batch_size)
    j = random.randrange(batch_size)
    while j == i:
        j = random.randrange(batch_size)

    swapped = batch_next_token_ids.clone()
    swapped[i], swapped[j] = (
        batch_next_token_ids[j].clone(),
        batch_next_token_ids[i].clone(),
    )

    logger.info(
        "kv_canary perturb next_token_swap: swapped i=%d j=%d step=%d",
        i,
        j,
        step,
    )
    return swapped
