# SPDX-License-Identifier: Apache-2.0
"""Request-owned TaylorSeer acceleration for BAGEL denoising.

This module implements the Taylor forecasting schedule described by
`TaylorSeer <https://github.com/Shenyi-Z/TaylorSeer>`_ and adapted by the
`official BAGEL implementation <https://github.com/ByteDance-Seed/Bagel/blob/57c390a038976a763ced0ffedd60b6b7885a6009/modeling/cache_utils/taylorseer.py>`_.
State is owned by one request (or one scheduler-merged request), never by the
shared model, and each classifier-free-guidance branch has an independent cache
and evaluation counter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from weakref import WeakSet

import torch
from torch import Tensor


@dataclass(frozen=True)
class TaylorSeerConfig:
    """Configure BAGEL's native per-layer Taylor forecast.

    Args:
        max_order: Highest derivative order retained for forecasting.
        fresh_threshold: Refresh period measured in denoising evaluations.
        first_enhance: Number of initial evaluations that always run in full.

    Raises:
        ValueError: If any setting is not a positive integer.
    """

    max_order: int = 6
    fresh_threshold: int = 3
    first_enhance: int = 5

    def __post_init__(self) -> None:
        for name, value in (
            ("max_order", self.max_order),
            ("fresh_threshold", self.fresh_threshold),
            ("first_enhance", self.first_enhance),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"TaylorSeer {name} must be a positive integer")


@dataclass
class _TaylorLayerCache:
    derivatives: dict[int, Tensor] = field(default_factory=dict)
    last_refresh_step: int | None = None


@dataclass
class _TaylorRunHealth:
    failed: bool = False
    released: bool = False
    states: WeakSet[TaylorSeerState] = field(default_factory=WeakSet)


class TaylorSeerState:
    """Track Taylor derivatives for one BAGEL CFG branch.

    Args:
        num_layers: Number of transformer layers cached by this branch.
        num_steps: Number of denoising evaluations in the request.
        config: Taylor refresh and forecast settings.

    Raises:
        ValueError: If layer or step counts are not positive integers.
    """

    def __init__(
        self,
        num_layers: int,
        num_steps: int,
        config: TaylorSeerConfig | None = None,
    ) -> None:
        for name, value in (("num_layers", num_layers), ("num_steps", num_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"TaylorSeer {name} must be a positive integer")
        self.config = config or TaylorSeerConfig()
        self.num_layers = num_layers
        self.num_steps = num_steps
        self._layers = [_TaylorLayerCache() for _ in range(num_layers)]
        self._completed_steps = 0
        self._active_step: int | None = None
        self._refresh_step = True
        self._cache_counter = 0
        self._full_steps = 0
        self._run_health = _TaylorRunHealth()
        self._run_health.states.add(self)

    @property
    def step_type(self) -> str:
        """Return ``full`` for a refresh or ``Taylor`` for a forecast step."""
        return "full" if self._refresh_step else "Taylor"

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def is_failed(self) -> bool:
        return self._run_health.failed

    @property
    def is_refresh_step(self) -> bool:
        return self._refresh_step

    def begin_step(self, step: int) -> str:
        """Begin one sequential denoising evaluation.

        Args:
            step: Zero-based denoising evaluation index.

        Returns:
            ``full`` when layers must refresh, otherwise ``Taylor``.

        Raises:
            RuntimeError: If another evaluation is active.
            ValueError: If the index is invalid or not sequential.
        """
        self._require_healthy()
        if self._active_step is not None:
            raise RuntimeError(
                "TaylorSeer cannot begin a step before ending the prior one"
            )
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("TaylorSeer step must be an integer")
        if step != self._completed_steps or step >= self.num_steps:
            raise ValueError(
                "TaylorSeer steps must be sequential and remain within the request"
            )

        initial_refresh = step < self.config.first_enhance
        interval_refresh = self._cache_counter == self.config.fresh_threshold - 1
        self._refresh_step = initial_refresh or interval_refresh
        if self._refresh_step:
            self._cache_counter = 0
        else:
            self._cache_counter += 1
        self._active_step = step
        return self.step_type

    def begin_next_step(self) -> str:
        """Begin the next actual evaluation of this CFG branch.

        Returns:
            ``full`` when layers must refresh, otherwise ``Taylor``.

        Raises:
            RuntimeError: If another evaluation is active.
            ValueError: If the branch exceeds the request's denoising budget.
        """
        return self.begin_step(self._completed_steps)

    def end_step(self) -> None:
        """Finish the active denoising evaluation.

        Raises:
            RuntimeError: If no evaluation is active.
        """
        self._require_healthy()
        if self._active_step is None:
            raise RuntimeError("TaylorSeer has no active step to end")
        if self._refresh_step:
            self._full_steps += 1
        self._active_step = None
        self._completed_steps += 1

    def should_compute(self, layer_index: int) -> bool:
        """Return whether a layer must execute at the active step.

        Args:
            layer_index: Zero-based transformer layer index.

        Returns:
            True for scheduled refreshes or an uninitialized layer cache.

        Raises:
            RuntimeError: If no evaluation is active.
            IndexError: If the layer index is outside the transformer.
        """
        self._require_healthy()
        layer = self._layer(layer_index)
        self._require_active_step()
        return self._refresh_step or not layer.derivatives

    def update_cache(self, layer_index: int, output: Tensor) -> None:
        """Refresh one layer's derivative cache from a full output.

        Args:
            layer_index: Zero-based transformer layer index.
            output: Full layer output for the active denoising evaluation.

        Raises:
            RuntimeError: If no evaluation is active or the layer was updated twice.
            IndexError: If the layer index is outside the transformer.
        """
        self._require_healthy()
        layer = self._layer(layer_index)
        step = self._require_active_step()
        if layer.last_refresh_step == step:
            raise RuntimeError(
                "TaylorSeer layer cache cannot refresh twice in one step"
            )

        updated = {0: output.detach()}
        if layer.last_refresh_step is not None:
            delta = step - layer.last_refresh_step
            if delta <= 0:
                raise RuntimeError("TaylorSeer layer refresh steps must increase")
            if step > self.config.first_enhance - 2:
                for order in range(self.config.max_order):
                    previous = layer.derivatives.get(order)
                    if previous is None:
                        break
                    updated[order + 1] = (updated[order] - previous) / delta

        layer.derivatives = updated
        layer.last_refresh_step = step

    def approximate(self, layer_index: int) -> Tensor:
        """Forecast one layer output at the active denoising step.

        Args:
            layer_index: Zero-based transformer layer index.

        Returns:
            Taylor forecast with the same shape and dtype as the cached output.

        Raises:
            RuntimeError: If no evaluation is active or the layer has no cache.
            IndexError: If the layer index is outside the transformer.
        """
        self._require_healthy()
        layer = self._layer(layer_index)
        step = self._require_active_step()
        if not layer.derivatives or layer.last_refresh_step is None:
            raise RuntimeError("TaylorSeer cannot forecast an uninitialized layer")

        offset = step - layer.last_refresh_step
        output = torch.zeros_like(layer.derivatives[0])
        for order in sorted(layer.derivatives):
            output = output + (
                layer.derivatives[order] * (offset**order) / math.factorial(order)
            )
        return output

    def get_stats(self) -> dict[str, int]:
        return {
            "total_steps": self._completed_steps,
            "full_steps": self._full_steps,
            "taylor_steps": self._completed_steps - self._full_steps,
        }

    def poison(self) -> None:
        """Invalidate all CFG caches after a partial model failure.

        Cache updates are committed layer by layer and cannot be rolled back
        cheaply. A failed branch therefore invalidates the complete request
        context and drops retained tensors so it cannot be silently retried.
        """
        self._run_health.failed = True
        for state in tuple(self._run_health.states):
            state._active_step = None
            state._layers = []

    def release(self) -> None:
        """Release all CFG branch tensors after denoising.

        This operation is idempotent and invalidates subsequent use. Explicit
        release guarantees that large CUDA caches disappear before VAE decode
        instead of waiting for Python garbage collection.
        """
        if self._run_health.released:
            return
        for state in tuple(self._run_health.states):
            state._active_step = None
            state._layers = []
        self._run_health.released = True
        self._run_health.states.clear()

    def _layer(self, layer_index: int) -> _TaylorLayerCache:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.num_layers
        ):
            raise IndexError("TaylorSeer layer index is outside the transformer")
        return self._layers[layer_index]

    def _require_active_step(self) -> int:
        if self._active_step is None:
            raise RuntimeError("TaylorSeer operation requires an active step")
        return self._active_step

    def _require_healthy(self) -> None:
        if self._run_health.failed:
            raise RuntimeError(
                "TaylorSeer request state is invalid after a failed model evaluation"
            )
        if self._run_health.released:
            raise RuntimeError("TaylorSeer request state was released after denoising")

    def _bind_run_health(self, run_health: _TaylorRunHealth) -> None:
        if (
            self._run_health.failed
            or self._run_health.released
            or len(self._layers) != self.num_layers
            or self._active_step is not None
            or self._completed_steps
            or any(layer.derivatives for layer in self._layers)
        ):
            raise ValueError("TaylorSeer can bind CFG branches only before evaluation")
        self._run_health = run_health
        run_health.states.add(self)


class BagelTaylorSeerContext:
    """Own independent Taylor state for every BAGEL CFG branch.

    Args:
        conditional: Cache for the full conditional branch.
        unconditional: Cache for the text-unconditional branch.
        secondary_unconditional: Optional third cache used by Thinking or Editing.

    Raises:
        ValueError: If branches share state or use different model geometry.
    """

    def __init__(
        self,
        conditional: TaylorSeerState,
        unconditional: TaylorSeerState,
        secondary_unconditional: TaylorSeerState | None = None,
    ) -> None:
        states = [conditional, unconditional]
        if secondary_unconditional is not None:
            states.append(secondary_unconditional)
        if len({id(state) for state in states}) != len(states):
            raise ValueError("BAGEL TaylorSeer CFG branches must not share state")
        if (
            len({state.num_layers for state in states}) != 1
            or len({state.num_steps for state in states}) != 1
        ):
            raise ValueError("BAGEL TaylorSeer CFG states must use one geometry")
        run_health = _TaylorRunHealth()
        for state in states:
            state._bind_run_health(run_health)
        self.conditional = conditional
        self.unconditional = unconditional
        self.secondary_unconditional = secondary_unconditional
        self.num_steps = conditional.num_steps

    @classmethod
    def create(
        cls,
        *,
        num_layers: int,
        num_steps: int,
        has_secondary: bool,
        config: TaylorSeerConfig | None = None,
    ) -> BagelTaylorSeerContext:
        """Create isolated state for a BAGEL request.

        Args:
            num_layers: Number of BAGEL transformer layers.
            num_steps: Number of request denoising evaluations.
            has_secondary: Whether Thinking or Editing needs a third CFG branch.
            config: Taylor refresh and forecast settings.

        Returns:
            A request-owned multi-branch Taylor context.

        Raises:
            ValueError: If layer/step counts or config values are invalid.
        """
        if not isinstance(has_secondary, bool):
            raise ValueError("BAGEL TaylorSeer has_secondary must be a boolean")
        resolved_config = config or TaylorSeerConfig()

        def make_state() -> TaylorSeerState:
            return TaylorSeerState(num_layers, num_steps, resolved_config)

        return cls(
            make_state(),
            make_state(),
            make_state() if has_secondary else None,
        )

    def validate_branch_count(self, *, has_secondary: bool) -> None:
        """Validate that cached branches match the current BAGEL context.

        Args:
            has_secondary: Whether the BAGEL context has a third CFG branch.

        Raises:
            ValueError: If the request context and Taylor state disagree.
        """
        if has_secondary != (self.secondary_unconditional is not None):
            raise ValueError("BAGEL TaylorSeer state does not match CFG branch count")

    def get_stats(self) -> dict[str, dict[str, int]]:
        stats = {
            "conditional": self.conditional.get_stats(),
            "unconditional": self.unconditional.get_stats(),
        }
        if self.secondary_unconditional is not None:
            stats["secondary_unconditional"] = self.secondary_unconditional.get_stats()
        return stats

    @property
    def is_failed(self) -> bool:
        return self.conditional.is_failed

    def release(self) -> None:
        self.conditional.release()
