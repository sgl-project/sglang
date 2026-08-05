# SPDX-License-Identifier: Apache-2.0
"""LTX-2 Pyramid Attention Broadcast skeleton.

PAB caches attention *outputs* inside a transformer block and broadcasts them
to later denoising steps. This module is intentionally add-only and does not
wrap ``LTX2TransformerBlock.forward``; it installs small wrappers around the
attention submodules that PAB needs to control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import nn


class LTX2PABAttentionKind(str, Enum):
    """Attention hook names inside ``LTX2TransformerBlock``."""

    VIDEO_SELF = "video_self"
    AUDIO_SELF = "audio_self"
    PROMPT_VIDEO = "prompt_video"
    PROMPT_AUDIO = "prompt_audio"
    AUDIO_TO_VIDEO = "audio_to_video"
    VIDEO_TO_AUDIO = "video_to_audio"


_DEFAULT_KIND_WINDOWS: dict[LTX2PABAttentionKind, str] = {
    # LTX-2 does not expose separate spatial and temporal video self-attn
    # blocks. Treat mutable latent self-attn conservatively.
    LTX2PABAttentionKind.VIDEO_SELF: "spatial",
    LTX2PABAttentionKind.AUDIO_SELF: "temporal",
    # Text/prompt KV is stable across denoising steps.
    LTX2PABAttentionKind.PROMPT_VIDEO: "cross",
    LTX2PABAttentionKind.PROMPT_AUDIO: "cross",
    # A/V cross-attn uses mutable latent streams, so start conservative.
    LTX2PABAttentionKind.AUDIO_TO_VIDEO: "spatial",
    LTX2PABAttentionKind.VIDEO_TO_AUDIO: "spatial",
}


@dataclass(frozen=True)
class LTX2PABCacheKey:
    """Per-request cache key for one attention sub-block output."""

    stage: str
    branch: str
    pass_id: str
    block_idx: int
    attention_kind: LTX2PABAttentionKind
    mask_signature: tuple[Any, ...]


@dataclass
class LTX2PABEntry:
    output: torch.Tensor
    source_step: int
    window: int


@dataclass
class LTX2PABStats:
    calls: int = 0
    computes: int = 0
    hits: int = 0
    stores: int = 0
    disabled: int = 0

    @property
    def hit_rate(self) -> float:
        usable = self.hits + self.computes
        return float(self.hits) / float(usable) if usable else 0.0


@dataclass
class LTX2PABConfig:
    """Runtime knobs for LTX-2 PAB.

    A broadcast window of ``N`` means an output computed at step ``s`` can be
    reused while ``current_step - s < N``. ``N=1`` effectively disables reuse
    for that attention type while still exercising the hook.
    """

    enabled: bool = True
    spatial_broadcast_window: int = 2
    temporal_broadcast_window: int = 4
    cross_broadcast_window: int = 6
    warmup_steps: int = 0
    active_start_step: int | None = None
    active_end_step: int | None = None
    active_start_fraction: float | None = 0.15
    active_end_fraction: float | None = 0.85
    stage1_enabled: bool = True
    stage2_enabled: bool = False
    stage2_spatial_broadcast_window: int = 1
    stage2_temporal_broadcast_window: int = 1
    stage2_cross_broadcast_window: int = 1
    stage2_active_start_step: int | None = None
    stage2_active_end_step: int | None = None
    stage2_active_start_fraction: float | None = None
    stage2_active_end_fraction: float | None = None
    disable_audio_self: bool = False
    disable_audio_video_cross: bool = True
    disable_when_perturbed: bool = True
    detach_on_store: bool = True
    clone_on_hit: bool = False
    per_kind_window_overrides: dict[LTX2PABAttentionKind, int] = field(
        default_factory=dict
    )


@dataclass
class LTX2PABRuntimeContext:
    current_step: int
    num_inference_steps: int
    stage: str = "stage1"
    branch: str = "default"
    pass_id: str = "default"


class LTX2PABCoordinator:
    """State manager shared by all attention wrappers on one transformer."""

    def __init__(self, config: LTX2PABConfig | None = None) -> None:
        self.config = config or LTX2PABConfig()
        self._cache: dict[LTX2PABCacheKey, LTX2PABEntry] = {}
        self._stats: dict[LTX2PABAttentionKind, LTX2PABStats] = {
            kind: LTX2PABStats() for kind in LTX2PABAttentionKind
        }
        self._last_step: int | None = None
        self._manual_context: LTX2PABRuntimeContext | None = None

    @property
    def stats(self) -> dict[str, LTX2PABStats]:
        return {kind.value: stats for kind, stats in self._stats.items()}

    def reset(self) -> None:
        self._cache.clear()
        for stats in self._stats.values():
            stats.calls = 0
            stats.computes = 0
            stats.hits = 0
            stats.stores = 0
            stats.disabled = 0
        self._last_step = None

    def begin_step(
        self,
        *,
        current_step: int,
        num_inference_steps: int,
        stage: str = "stage1",
        branch: str = "default",
        pass_id: str = "default",
    ) -> None:
        """Optional explicit context injection for callers without forward_context."""

        self._manual_context = LTX2PABRuntimeContext(
            current_step=current_step,
            num_inference_steps=num_inference_steps,
            stage=stage,
            branch=branch,
            pass_id=pass_id,
        )

    def end_request(self) -> None:
        self._manual_context = None
        self.reset()

    def forward_attention(
        self,
        attention: nn.Module,
        block_idx: int,
        attention_kind: LTX2PABAttentionKind,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        stats = self._stats[attention_kind]
        stats.calls += 1

        ctx = self._get_runtime_context()
        if not self._should_attempt_broadcast(ctx, attention_kind, kwargs):
            stats.disabled += 1
            return attention(*args, **kwargs)

        assert ctx is not None
        key = LTX2PABCacheKey(
            stage=ctx.stage,
            branch=ctx.branch,
            pass_id=ctx.pass_id,
            block_idx=int(block_idx),
            attention_kind=attention_kind,
            mask_signature=self._mask_signature(kwargs),
        )
        entry = self._cache.get(key)
        if entry is not None and ctx.current_step - entry.source_step < entry.window:
            stats.hits += 1
            return entry.output.clone() if self.config.clone_on_hit else entry.output

        stats.computes += 1
        output = attention(*args, **kwargs)
        self._cache[key] = LTX2PABEntry(
            output=output.detach() if self.config.detach_on_store else output,
            source_step=ctx.current_step,
            window=self._window_for(ctx.stage, attention_kind),
        )
        stats.stores += 1
        return output

    def _get_runtime_context(self) -> LTX2PABRuntimeContext | None:
        if self._manual_context is not None:
            ctx = self._manual_context
        else:
            ctx = self._context_from_forward_context()
            if ctx is None:
                return None

        if ctx.current_step == 0 and self._last_step not in (None, 0):
            self._cache.clear()
        self._last_step = ctx.current_step
        return ctx

    def _context_from_forward_context(self) -> LTX2PABRuntimeContext | None:
        try:
            from sglang.multimodal_gen.runtime.managers.forward_context import (
                get_forward_context,
            )
        except ImportError:
            return None

        try:
            forward_context = get_forward_context()
        except AssertionError:
            return None

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        forward_batch = getattr(forward_context, "forward_batch", None)

        stage = None
        num_steps = 0
        if attn_metadata is not None:
            stage = getattr(attn_metadata, "ltx2_stage", None)
            num_steps = int(getattr(attn_metadata, "ltx2_num_steps", 0) or 0)
        if stage is None and forward_batch is not None:
            stage = getattr(forward_batch, "ltx2_stage", None) or getattr(
                forward_batch, "stage", None
            )
        if stage is None:
            stage = "stage1"
        if num_steps <= 0 and forward_batch is not None:
            num_steps = int(getattr(forward_batch, "num_inference_steps", 0) or 0)

        branch = "default"
        pass_id = "default"
        if forward_batch is not None:
            branch = "negative" if getattr(forward_batch, "is_cfg_negative", False) else (
                "positive"
                if getattr(forward_batch, "do_classifier_free_guidance", False)
                else "default"
            )
            pass_id = str(getattr(forward_batch, "ltx2_pass_id", branch))

        return LTX2PABRuntimeContext(
            current_step=int(getattr(forward_context, "current_timestep", 0) or 0),
            num_inference_steps=num_steps,
            stage=str(stage),
            branch=str(branch),
            pass_id=str(pass_id),
        )

    def _should_attempt_broadcast(
        self,
        ctx: LTX2PABRuntimeContext | None,
        attention_kind: LTX2PABAttentionKind,
        kwargs: dict[str, Any],
    ) -> bool:
        if ctx is None or not self.config.enabled:
            return False
        if ctx.stage == "stage1" and not self.config.stage1_enabled:
            return False
        if ctx.stage == "stage2" and not self.config.stage2_enabled:
            return False
        if ctx.current_step < self.config.warmup_steps:
            return False
        if not self._within_active_segment(ctx):
            return False
        if self._window_for(ctx.stage, attention_kind) <= 1:
            return False
        if self.config.disable_audio_self and attention_kind == (
            LTX2PABAttentionKind.AUDIO_SELF
        ):
            return False
        if self.config.disable_audio_video_cross and attention_kind in {
            LTX2PABAttentionKind.AUDIO_TO_VIDEO,
            LTX2PABAttentionKind.VIDEO_TO_AUDIO,
        }:
            return False
        if self.config.disable_when_perturbed and (
            kwargs.get("all_perturbed")
            or kwargs.get("perturbation_mask") is not None
        ):
            return False
        return True

    def _within_active_segment(self, ctx: LTX2PABRuntimeContext) -> bool:
        if ctx.stage == "stage2":
            start = self.config.stage2_active_start_step
            end = self.config.stage2_active_end_step
            start_fraction = self.config.stage2_active_start_fraction
            end_fraction = self.config.stage2_active_end_fraction
        else:
            start = self.config.active_start_step
            end = self.config.active_end_step
            start_fraction = self.config.active_start_fraction
            end_fraction = self.config.active_end_fraction
        if start is None and start_fraction is not None:
            start = int(ctx.num_inference_steps * start_fraction)
        if end is None and end_fraction is not None:
            end = int(ctx.num_inference_steps * end_fraction)
        if start is not None and ctx.current_step < start:
            return False
        if end is not None and ctx.current_step >= end:
            return False
        return True

    def _window_for(
        self, stage: str, attention_kind: LTX2PABAttentionKind
    ) -> int:
        override = self.config.per_kind_window_overrides.get(attention_kind)
        if override is not None:
            return max(1, int(override))

        bucket = _DEFAULT_KIND_WINDOWS[attention_kind]
        if stage == "stage2":
            value = {
                "spatial": self.config.stage2_spatial_broadcast_window,
                "temporal": self.config.stage2_temporal_broadcast_window,
                "cross": self.config.stage2_cross_broadcast_window,
            }[bucket]
        else:
            value = {
                "spatial": self.config.spatial_broadcast_window,
                "temporal": self.config.temporal_broadcast_window,
                "cross": self.config.cross_broadcast_window,
            }[bucket]
        return max(1, int(value))

    @staticmethod
    def _mask_signature(kwargs: dict[str, Any]) -> tuple[Any, ...]:
        """Cheap structural signature; production code should add key invariants."""

        mask = kwargs.get("mask")
        if mask is None:
            return ("mask", None)
        return (
            "mask",
            tuple(mask.shape),
            str(mask.dtype),
            str(mask.device),
        )


class LTX2PABAttentionWrapper(nn.Module):
    """Small module wrapper that preserves the original attention call API."""

    def __init__(
        self,
        inner: nn.Module,
        coordinator: LTX2PABCoordinator,
        block_idx: int,
        attention_kind: LTX2PABAttentionKind,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.coordinator = coordinator
        self.block_idx = int(block_idx)
        self.attention_kind = attention_kind

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.coordinator.forward_attention(
            self.inner,
            self.block_idx,
            self.attention_kind,
            *args,
            **kwargs,
        )


class LTX2PABMixin:
    """TeaCacheMixin-style state hooks for a model-native LTX-2 PAB path."""

    def _init_ltx2_pab_state(
        self, config: LTX2PABConfig | None = None
    ) -> None:
        self.ltx2_pab = LTX2PABCoordinator(config=config)
        install_ltx2_pab_hooks(self, coordinator=self.ltx2_pab)

    def reset_ltx2_pab_state(self) -> None:
        if hasattr(self, "ltx2_pab"):
            self.ltx2_pab.reset()

    def enable_ltx2_pab(self) -> None:
        self.ltx2_pab.config.enabled = True

    def disable_ltx2_pab(self) -> None:
        self.ltx2_pab.config.enabled = False


def install_ltx2_pab_hooks(
    transformer: nn.Module,
    *,
    config: LTX2PABConfig | None = None,
    coordinator: LTX2PABCoordinator | None = None,
) -> LTX2PABCoordinator:
    """Install PAB wrappers on every ``LTX2TransformerBlock`` attention module."""

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise AttributeError("Expected transformer.transformer_blocks for LTX-2 PAB.")

    coordinator = coordinator or LTX2PABCoordinator(config=config)
    mapping = {
        "attn1": LTX2PABAttentionKind.VIDEO_SELF,
        "audio_attn1": LTX2PABAttentionKind.AUDIO_SELF,
        "attn2": LTX2PABAttentionKind.PROMPT_VIDEO,
        "audio_attn2": LTX2PABAttentionKind.PROMPT_AUDIO,
        "audio_to_video_attn": LTX2PABAttentionKind.AUDIO_TO_VIDEO,
        "video_to_audio_attn": LTX2PABAttentionKind.VIDEO_TO_AUDIO,
    }

    for block_idx, block in enumerate(blocks):
        for attr_name, attention_kind in mapping.items():
            attention = getattr(block, attr_name, None)
            if attention is None:
                continue
            if isinstance(attention, LTX2PABAttentionWrapper):
                attention.coordinator = coordinator
                continue
            setattr(
                block,
                attr_name,
                LTX2PABAttentionWrapper(
                    attention,
                    coordinator=coordinator,
                    block_idx=block_idx,
                    attention_kind=attention_kind,
                ),
            )

    setattr(transformer, "ltx2_pab", coordinator)
    return coordinator


__all__ = [
    "LTX2PABAttentionKind",
    "LTX2PABCacheKey",
    "LTX2PABConfig",
    "LTX2PABCoordinator",
    "LTX2PABEntry",
    "LTX2PABMixin",
    "LTX2PABRuntimeContext",
    "LTX2PABStats",
    "install_ltx2_pab_hooks",
]
