"""Pseudo-mode sampler-side override.

Wraps :meth:`ModelRunner.sample` so the real sampler still runs (top-p /
penalty / NaN detect / sampler kernel paths all still execute), and the
returned ``next_token_ids`` tensor is overwritten in place by the
oracle's prediction. Patching ``model_runner.sample`` (the unique
sampling funnel) rather than ``Sampler.forward`` keeps the wrapper a
single bound-method swap and also automatically covers the overlap
scheduler's ``delay_sample_func`` closure (the closure also calls
``model_runner.sample``).
"""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.pseudo_mode.oracle import PseudoOracle

logger = logging.getLogger(__name__)

_SAMPLE_PATCHED_ATTR = "_pseudo_mode_sample_patched"


def install_sampler_override(
    *,
    model_runner: "ModelRunner",
    oracle: "PseudoOracle",
) -> None:
    """Install the pseudo-mode sampler override on ``model_runner.sample``.

    Idempotent: a second call on the same ``model_runner`` is a no-op.
    Prefill-only batches are left alone — the real ``sample`` already
    returns a dummy zero tensor on that path and the output is not
    consumed downstream.

    ``SGLANG_PSEUDO_INPUT_PERTURB_PROB`` (env, float in ``[0, 1]``)
    perturbs the override by writing the wrong token with that
    probability per request, so tests can exercise the canary's
    ``INPUT_TOKEN_MISMATCH`` path.
    """
    if getattr(model_runner, _SAMPLE_PATCHED_ATTR, False):
        return

    original_sample = model_runner.sample
    perturb_prob = _clamp_unit(envs.SGLANG_PSEUDO_INPUT_PERTURB_PROB.get())
    perturb_seed = int(envs.SGLANG_PSEUDO_INPUT_PERTURB_SEED.get())

    def patched_sample(
        self: "ModelRunner",
        logits_output: "LogitsProcessorOutput",
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        real_next_tokens = original_sample(logits_output, forward_batch)
        if forward_batch.is_prefill_only:
            return real_next_tokens

        forced = oracle.predict_next_tokens_for_active_batch(
            forward_batch=forward_batch,
            device=real_next_tokens.device,
        )
        if perturb_prob > 0.0:
            forced = _maybe_perturb_tokens(
                tokens=forced,
                prob=perturb_prob,
                seed=perturb_seed,
                vocab_size=oracle.vocab_size,
            )
        real_next_tokens.copy_(forced)
        return real_next_tokens

    model_runner.sample = types.MethodType(patched_sample, model_runner)
    setattr(model_runner, _SAMPLE_PATCHED_ATTR, True)


def _clamp_unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _maybe_perturb_tokens(
    *,
    tokens: torch.Tensor,
    prob: float,
    seed: int,
    vocab_size: int,
) -> torch.Tensor:
    """Replace a random subset of ``tokens`` with an offset wrong value.

    The replacement token is ``(original + 1) % vocab_size`` so the
    perturbed value is always distinct from the oracle prediction yet
    still in-vocab. Used by the canary self-test path to exercise
    ``INPUT_TOKEN_MISMATCH``.
    """
    if prob <= 0.0 or tokens.numel() == 0:
        return tokens
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    mask = (
        torch.rand(tokens.shape, device=tokens.device, generator=generator) < prob
    )
    if not bool(mask.any()):
        return tokens
    perturbed = tokens.clone()
    perturbed[mask] = (perturbed[mask] + 1) % vocab_size
    return perturbed
