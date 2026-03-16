# SPDX-License-Identifier: Apache-2.0
"""
Base class for diffusion model cache strategies (TeaCache, MagCache, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.cache.magcache import MagCacheState
    from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheState


class DiffusionCache:
    """
    Base class for diffusion model caching strategies.

    Each subclass owns its own state (positive + negative CFG branch) and
    context extraction logic. CachableDiT holds a single
    `self.cache: DiffusionCache | None` and delegates all decisions here.

    Subclasses must implement: reset, get_context, should_skip.
    maybe_cache, retrieve, and calibrate have default implementations.

    Subclasses set self.state / self.state_neg to strategy-specific state
    objects (positive and negative CFG branches respectively). state_neg is
    None when CFG negative-branch caching is disabled.

    Typical forward pass usage in CachableDiT:

        ctx = self.cache.get_context(self.cnt)
        if ctx and self.cache.should_skip(ctx, timestep_proj=..., temb=...):
            hidden_states = self.cache.retrieve(hidden_states, ctx)
        else:
            original_hidden_states = hidden_states.clone()
            # ... run transformer blocks ...
            if calibrate_cache:
                self.cache.calibrate(hidden_states, original_hidden_states, ctx)
            else:
                self.cache.maybe_cache(hidden_states, original_hidden_states, ctx)
    """

    def __init__(self) -> None:
        self.state: MagCacheState | TeaCacheState | None = None
        self.state_neg: MagCacheState | TeaCacheState | None = None

    def reset(self) -> None:
        """Reset all state at the start of a new generation."""
        raise NotImplementedError

    def get_context(self, cnt: int):
        """
        Read the global forward_context / forward_batch and return a
        strategy-specific context dataclass, or None to bypass caching.

        cnt is the monotonically increasing forward-call index owned by the
        model (model.cnt), incremented on every call regardless of whether
        the forward pass was skipped.
        """
        raise NotImplementedError

    def should_skip(self, ctx, **kwargs) -> bool:
        """
        Decide whether to skip the transformer forward pass and reuse the
        cached residual. kwargs carries model-specific tensors (e.g.
        timestep_proj, temb) needed by some strategies.
        """
        raise NotImplementedError

    def maybe_cache(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        ctx,
    ) -> None:
        """Store residual after a full forward pass for future reuse."""
        state = (
            self.state_neg
            if (ctx.is_cfg_negative and self.state_neg is not None)
            else self.state
        )
        state.previous_residual = hidden_states.squeeze(0) - original_hidden_states

    def retrieve(self, hidden_states: torch.Tensor, ctx) -> torch.Tensor:
        """Reconstruct output from cached residual."""
        state = (
            self.state_neg
            if (ctx.is_cfg_negative and self.state_neg is not None)
            else self.state
        )
        return hidden_states + state.previous_residual

    def calibrate(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        ctx,
    ) -> None:
        """Log calibration metrics. No-op by default."""
        pass
