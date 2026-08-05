# SPDX-License-Identifier: Apache-2.0
"""Compile-compatible EasyCache control for diffusion transformer block stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.easycache import EasyCacheParams


_REQUEST_STATE_KEY = "_easycache_state"


@dataclass
class EasyCacheState:
    """Mutable cache state owned by one generation request."""

    active_step: int = -1
    should_compute: bool = True
    previous_input: torch.Tensor | None = None
    previous_output: torch.Tensor | None = None
    previous_output_norm: float | None = None
    input_output_rate: float | None = None
    accumulated_change: float = 0.0
    residual: torch.Tensor | None = None

    def reset(self) -> None:
        self.active_step = -1
        self.should_compute = True
        self.previous_input = None
        self.previous_output = None
        self.previous_output_norm = None
        self.input_output_rate = None
        self.accumulated_change = 0.0
        self.residual = None


@dataclass(frozen=True)
class EasyCacheDecision:
    """Decision and request context for one model forward."""

    state: EasyCacheState
    params: EasyCacheParams
    step_index: int
    is_negative_branch: bool
    should_compute: bool


class EasyCacheController:
    """Online EasyCache policy independent of any particular DiT architecture.

    A model calls :meth:`begin_forward` after producing its transformer-block
    input, then either computes the block stack and calls :meth:`after_compute`
    or calls :meth:`reuse`.

    Cache state lives in ``Req.extra`` rather than on the model, so separate
    requests cannot inherit each other's residuals. With serial two-pass CFG,
    the conditional branch makes the decision and both branches share the same
    residual, matching the Sol-Engine SANA-Video behavior.
    """

    @staticmethod
    def _subsample(tensor: torch.Tensor, stride: int) -> torch.Tensor:
        return tensor[:, ::stride].float()

    @classmethod
    def _decide(
        cls,
        state: EasyCacheState,
        params: EasyCacheParams,
        step_index: int,
        block_input: torch.Tensor,
    ) -> bool:
        if (
            step_index < params.warmup_steps
            or state.previous_input is None
            or state.input_output_rate is None
            or state.previous_output_norm is None
            or state.residual is None
        ):
            return True

        current_input = cls._subsample(block_input, params.subsample_stride)
        input_change = (current_input - state.previous_input).abs().mean()
        approximate_change = float(
            (
                state.input_output_rate
                * input_change
                / max(state.previous_output_norm, 1e-6)
            ).item()
        )
        state.accumulated_change += approximate_change
        return state.accumulated_change >= params.threshold

    @classmethod
    @torch.compiler.disable
    def begin_forward(cls, block_input: torch.Tensor) -> EasyCacheDecision | None:
        """Return a request-local decision, or ``None`` when EasyCache is off."""

        try:
            forward_context = get_forward_context()
        except AssertionError:
            # Direct model calls (unit tests and component probes) do not
            # necessarily install a pipeline forward context.
            return None

        batch = forward_context.forward_batch
        if batch is None or not getattr(batch, "enable_easycache", False):
            return None

        params = getattr(batch, "easycache_params", None)
        if params is None or float(params.threshold) <= 0.0:
            return None

        state = batch.extra.get(_REQUEST_STATE_KEY)
        if not isinstance(state, EasyCacheState):
            state = EasyCacheState()
            batch.extra[_REQUEST_STATE_KEY] = state

        step_index = int(forward_context.current_timestep)
        is_negative_branch = bool(getattr(batch, "is_cfg_negative", False))

        # The default CFG policy runs conditional before unconditional.
        if step_index == 0 and not is_negative_branch:
            state.reset()

        if state.active_step != step_index:
            state.active_step = step_index
            state.should_compute = cls._decide(state, params, step_index, block_input)

        return EasyCacheDecision(
            state=state,
            params=params,
            step_index=step_index,
            is_negative_branch=is_negative_branch,
            should_compute=state.should_compute,
        )

    @classmethod
    @torch.compiler.disable
    def after_compute(
        cls,
        decision: EasyCacheDecision,
        block_input: torch.Tensor,
        block_output: torch.Tensor,
    ) -> None:
        """Update the online estimator and cache the latest block residual."""

        state = decision.state
        params = decision.params

        # In serial CFG the conditional branch owns the estimator. The
        # unconditional branch still replaces the shared residual so a skipped
        # step adds the same common-mode transform to both predictions.
        if not decision.is_negative_branch:
            current_input = cls._subsample(block_input, params.subsample_stride)
            current_output = cls._subsample(block_output, params.subsample_stride)

            if state.previous_input is not None and state.previous_output is not None:
                input_change = (current_input - state.previous_input).abs().mean()
                output_change = (current_output - state.previous_output).abs().mean()
                input_change_value = float(input_change.item())
                if input_change_value > 1e-12:
                    state.input_output_rate = float(
                        (output_change / input_change).item()
                    )

            state.previous_input = current_input.detach()
            state.previous_output = current_output.detach()
            state.previous_output_norm = float(block_output.float().abs().mean().item())
            state.accumulated_change = 0.0

        state.residual = (block_output - block_input).detach()

    @staticmethod
    @torch.compiler.disable
    def reuse(decision: EasyCacheDecision, block_input: torch.Tensor) -> torch.Tensor:
        """Apply the most recently computed block-stack residual."""

        residual = decision.state.residual
        if residual is None:
            raise RuntimeError("EasyCache selected reuse before a residual was cached")
        return block_input + residual
