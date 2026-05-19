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
    perturb_prob = _read_perturb_prob()
    perturb_seed = int(envs.SGLANG_PSEUDO_INPUT_PERTURB_SEED.get())
    perturb_state: _PerturbState = _PerturbState(
        prob=perturb_prob,
        vocab_size=oracle.vocab_size,
        seed=perturb_seed,
    )

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
        if perturb_state.prob > 0.0:
            forced = perturb_state.maybe_perturb(forced)
        real_next_tokens.copy_(forced)
        return real_next_tokens

    model_runner.sample = types.MethodType(patched_sample, model_runner)
    setattr(model_runner, _SAMPLE_PATCHED_ATTR, True)


def _read_perturb_prob() -> float:
    raw = envs.SGLANG_PSEUDO_INPUT_PERTURB_PROB.get()
    if not 0.0 <= raw <= 1.0:
        clamped = max(0.0, min(1.0, raw))
        logger.warning(
            "pseudo-mode: SGLANG_PSEUDO_INPUT_PERTURB_PROB %f out of [0,1]; "
            "clamped to %f",
            raw,
            clamped,
        )
        return clamped
    return raw


class _PerturbState:
    """Stateful RNG holder for the input-token perturbation self-test.

    Holding the ``torch.Generator`` across calls (instead of rebuilding
    it with the same seed every step) ensures the perturbation mask
    actually advances; otherwise the same positions would be perturbed
    every forward and the test would not exercise the canary's full
    coverage.
    """

    __slots__ = ("prob", "vocab_size", "_seed", "_generators")

    def __init__(self, *, prob: float, vocab_size: int, seed: int) -> None:
        self.prob: float = prob
        self.vocab_size: int = vocab_size
        self._seed: int = seed
        self._generators: dict = {}

    def maybe_perturb(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.prob <= 0.0 or tokens.numel() == 0:
            return tokens
        generator = self._generators.get(tokens.device)
        if generator is None:
            generator = torch.Generator(device=tokens.device).manual_seed(self._seed)
            self._generators[tokens.device] = generator
        mask = (
            torch.rand(tokens.shape, device=tokens.device, generator=generator)
            < self.prob
        )
        if not bool(mask.any()):
            return tokens
        perturbed = tokens.clone()
        perturbed[mask] = (perturbed[mask] + 1) % self.vocab_size
        return perturbed
