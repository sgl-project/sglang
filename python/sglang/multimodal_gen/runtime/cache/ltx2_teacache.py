# SPDX-License-Identifier: Apache-2.0
"""LTX-2 model-level TeaCache-style residual replay.

This is an opt-in runtime hook for LTX-2.3 cache experiments. It caches the
post-transformer-block residual for the video/audio streams and replays it on
later denoising steps when a simple timestep-embedding distance policy accepts
reuse. The output norm/projection/unpatchify path still runs every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, value, default)
        return default


@dataclass(frozen=True)
class LTX2TeaCacheConfig:
    enabled: bool = True
    threshold: float = 0.04
    start_step: int = 6
    end_step: int | None = None
    stage1_enabled: bool = True
    stage2_enabled: bool = False
    max_continuous_hits: int = 1
    periodic_recompute_steps: int = 0
    include_audio: bool = True
    detach_on_store: bool = True
    clone_on_hit: bool = False
    log_decisions: bool = False


@dataclass
class LTX2TeaCacheStats:
    calls: int = 0
    computes: int = 0
    hits: int = 0
    disabled: int = 0
    threshold_recomputes: int = 0
    periodic_recomputes: int = 0
    missing_recomputes: int = 0
    boundary_recomputes: int = 0
    skipped_steps: list[int] = field(default_factory=list)
    computed_steps: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        eligible = self.computes + self.hits
        return 0.0 if eligible == 0 else self.hits / eligible

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "computes": self.computes,
            "hits": self.hits,
            "disabled": self.disabled,
            "hit_rate": round(self.hit_rate, 4),
            "threshold_recomputes": self.threshold_recomputes,
            "periodic_recomputes": self.periodic_recomputes,
            "missing_recomputes": self.missing_recomputes,
            "boundary_recomputes": self.boundary_recomputes,
            "skipped_steps": sorted(set(self.skipped_steps)),
            "computed_steps": sorted(set(self.computed_steps)),
        }


@dataclass
class _Entry:
    previous_feature: torch.Tensor | None = None
    video_residual: torch.Tensor | None = None
    audio_residual: torch.Tensor | None = None
    accumulated_distance: float = 0.0
    last_compute_step: int | None = None
    continuous_hits: int = 0


@dataclass(frozen=True)
class LTX2TeaCacheDecision:
    should_skip: bool
    key: tuple[Any, ...]
    reason: str
    hidden_states: torch.Tensor | None = None
    audio_hidden_states: torch.Tensor | None = None


class LTX2TeaCacheCoordinator:
    def __init__(self, config: LTX2TeaCacheConfig) -> None:
        self.config = config
        self._entries: dict[tuple[Any, ...], _Entry] = {}
        self._stats: dict[str, LTX2TeaCacheStats] = {}
        self._last_stage_step: tuple[str, int] | None = None

    def reset(self) -> None:
        self._entries.clear()
        self._last_stage_step = None

    def stats_summary(self) -> dict[str, Any]:
        return {stage: stats.as_dict() for stage, stats in sorted(self._stats.items())}

    def _runtime_context(self) -> tuple[str, int, int, str]:
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return "stage1", 0, 0, "default"
        attn_metadata = getattr(forward_context, "attn_metadata", None)
        forward_batch = getattr(forward_context, "forward_batch", None)
        stage = None
        num_steps = 0
        if attn_metadata is not None:
            stage = getattr(attn_metadata, "ltx2_stage", None)
            num_steps = int(getattr(attn_metadata, "ltx2_num_steps", 0) or 0)
        if stage is None and forward_batch is not None:
            stage = getattr(forward_batch, "ltx2_stage", None) or getattr(forward_batch, "stage", None)
        if stage is None:
            stage = "stage1"
        if num_steps <= 0 and forward_batch is not None:
            num_steps = int(getattr(forward_batch, "num_inference_steps", 0) or 0)
        step = int(getattr(forward_context, "current_timestep", 0) or 0)
        pass_id = "default"
        if forward_batch is not None:
            pass_id = str(getattr(forward_batch, "ltx2_pass_id", pass_id))
        return str(stage), step, num_steps, pass_id

    def _feature(self, temb: torch.Tensor, temb_audio: torch.Tensor) -> torch.Tensor:
        tensors = [temb.detach().float().reshape(-1)]
        if self.config.include_audio:
            tensors.append(temb_audio.detach().float().reshape(-1))
        return torch.cat(tensors)

    def _key(
        self,
        *,
        stage: str,
        pass_id: str,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        skip_video_self_attn_blocks: Any,
        skip_audio_self_attn_blocks: Any,
        disable_a2v_cross_attn: bool,
        disable_v2a_cross_attn: bool,
        perturbation_configs: Any,
    ) -> tuple[Any, ...]:
        return (
            stage,
            pass_id,
            tuple(hidden_states.shape),
            tuple(audio_hidden_states.shape),
            str(hidden_states.dtype),
            str(audio_hidden_states.dtype),
            tuple(skip_video_self_attn_blocks or ()),
            tuple(skip_audio_self_attn_blocks or ()),
            bool(disable_a2v_cross_attn),
            bool(disable_v2a_cross_attn),
            bool(perturbation_configs is not None),
        )

    def _eligible(self, stage: str, step: int) -> bool:
        if not self.config.enabled:
            return False
        if stage == "stage1" and not self.config.stage1_enabled:
            return False
        if stage == "stage2" and not self.config.stage2_enabled:
            return False
        if step < self.config.start_step:
            return False
        if self.config.end_step is not None and step >= self.config.end_step:
            return False
        return True

    def lookup(
        self,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
        skip_video_self_attn_blocks: Any = None,
        skip_audio_self_attn_blocks: Any = None,
        disable_a2v_cross_attn: bool = False,
        disable_v2a_cross_attn: bool = False,
        perturbation_configs: Any = None,
        pass_id: str | None = None,
    ) -> LTX2TeaCacheDecision:
        stage, step, _num_steps, context_pass_id = self._runtime_context()
        if pass_id is None:
            pass_id = context_pass_id
        if self._last_stage_step is not None and step == 0 and self._last_stage_step != (stage, 0):
            self.reset()
        self._last_stage_step = (stage, step)

        stats = self._stats.setdefault(stage, LTX2TeaCacheStats())
        stats.calls += 1
        key = self._key(
            stage=stage,
            pass_id=pass_id,
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            skip_video_self_attn_blocks=skip_video_self_attn_blocks,
            skip_audio_self_attn_blocks=skip_audio_self_attn_blocks,
            disable_a2v_cross_attn=disable_a2v_cross_attn,
            disable_v2a_cross_attn=disable_v2a_cross_attn,
            perturbation_configs=perturbation_configs,
        )
        if not self._eligible(stage, step):
            stats.disabled += 1
            stats.boundary_recomputes += 1
            stats.computed_steps.append(step)
            return LTX2TeaCacheDecision(False, key, "disabled_or_boundary")

        entry = self._entries.setdefault(key, _Entry())
        if entry.video_residual is None or entry.audio_residual is None or entry.previous_feature is None:
            stats.computes += 1
            stats.missing_recomputes += 1
            stats.computed_steps.append(step)
            return LTX2TeaCacheDecision(False, key, "missing_cache")

        if self.config.periodic_recompute_steps > 0 and entry.last_compute_step is not None:
            if step - entry.last_compute_step >= self.config.periodic_recompute_steps:
                entry.continuous_hits = 0
                stats.computes += 1
                stats.periodic_recomputes += 1
                stats.computed_steps.append(step)
                return LTX2TeaCacheDecision(False, key, "periodic_recompute")

        if self.config.max_continuous_hits >= 0 and entry.continuous_hits >= self.config.max_continuous_hits:
            entry.continuous_hits = 0
            stats.computes += 1
            stats.periodic_recomputes += 1
            stats.computed_steps.append(step)
            return LTX2TeaCacheDecision(False, key, "continuous_hit_cap")

        feature = self._feature(temb, temb_audio)
        prev = entry.previous_feature
        if prev.shape != feature.shape:
            entry.continuous_hits = 0
            stats.computes += 1
            stats.missing_recomputes += 1
            stats.computed_steps.append(step)
            return LTX2TeaCacheDecision(False, key, "feature_shape_changed")
        rel_l1 = float(((feature - prev).abs().mean() / prev.abs().mean().clamp_min(1e-6)).detach().cpu().item())
        accumulated = entry.accumulated_distance + rel_l1
        if accumulated >= self.config.threshold:
            entry.accumulated_distance = 0.0
            entry.continuous_hits = 0
            stats.computes += 1
            stats.threshold_recomputes += 1
            stats.computed_steps.append(step)
            return LTX2TeaCacheDecision(False, key, f"threshold:{accumulated:.6f}")

        entry.accumulated_distance = accumulated
        entry.continuous_hits += 1
        stats.hits += 1
        stats.skipped_steps.append(step)
        video = hidden_states + (entry.video_residual.clone() if self.config.clone_on_hit else entry.video_residual)
        audio = audio_hidden_states + (entry.audio_residual.clone() if self.config.clone_on_hit else entry.audio_residual)
        if self.config.log_decisions:
            logger.info("LTX2 TeaCache hit stage=%s step=%s pass=%s rel_l1=%.6f accum=%.6f", stage, step, pass_id, rel_l1, accumulated)
        return LTX2TeaCacheDecision(True, key, "cache_hit", video, audio)

    def store(
        self,
        decision: LTX2TeaCacheDecision | None,
        *,
        original_hidden_states: torch.Tensor,
        original_audio_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        temb_audio: torch.Tensor,
    ) -> None:
        if decision is None or decision.should_skip:
            return
        entry = self._entries.setdefault(decision.key, _Entry())
        video_residual = hidden_states - original_hidden_states
        audio_residual = audio_hidden_states - original_audio_hidden_states
        if self.config.detach_on_store:
            video_residual = video_residual.detach()
            audio_residual = audio_residual.detach()
        entry.video_residual = video_residual
        entry.audio_residual = audio_residual
        entry.previous_feature = self._feature(temb, temb_audio)
        try:
            _stage, step, _num_steps, _pass_id = self._runtime_context()
            entry.last_compute_step = step
        except Exception:
            pass


def make_ltx2_teacache_config_from_env() -> LTX2TeaCacheConfig:
    end_step = _env_int("SGLANG_LTX2_TEACACHE_END", -1)
    return LTX2TeaCacheConfig(
        enabled=_env_flag("SGLANG_LTX2_TEACACHE_ENABLED", False),
        threshold=_env_float("SGLANG_LTX2_TEACACHE_THRESH", 0.04),
        start_step=_env_int("SGLANG_LTX2_TEACACHE_START", 6),
        end_step=None if end_step < 0 else end_step,
        stage1_enabled=_env_flag("SGLANG_LTX2_TEACACHE_STAGE1_ENABLED", True),
        stage2_enabled=not _env_flag("SGLANG_LTX2_TEACACHE_STAGE2_DISABLE", True),
        max_continuous_hits=_env_int("SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS", 1),
        periodic_recompute_steps=_env_int("SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS", 0),
        include_audio=_env_flag("SGLANG_LTX2_TEACACHE_INCLUDE_AUDIO", True),
        detach_on_store=_env_flag("SGLANG_LTX2_TEACACHE_DETACH_ON_STORE", True),
        clone_on_hit=_env_flag("SGLANG_LTX2_TEACACHE_CLONE_ON_HIT", False),
        log_decisions=_env_flag("SGLANG_LTX2_TEACACHE_LOG_DECISIONS", False),
    )


def get_ltx2_teacache_coordinator(transformer: object) -> LTX2TeaCacheCoordinator | None:
    config = make_ltx2_teacache_config_from_env()
    if not config.enabled:
        return None
    coordinator = getattr(transformer, "ltx2_teacache", None)
    if coordinator is None:
        coordinator = LTX2TeaCacheCoordinator(config)
        setattr(transformer, "ltx2_teacache", coordinator)
        logger.info("Installed LTX2 TeaCache residual replay hook: %s", config)
    return coordinator


__all__ = [
    "LTX2TeaCacheConfig",
    "LTX2TeaCacheCoordinator",
    "LTX2TeaCacheDecision",
    "get_ltx2_teacache_coordinator",
    "make_ltx2_teacache_config_from_env",
]
