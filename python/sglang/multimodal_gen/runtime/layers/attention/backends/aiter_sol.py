# SPDX-License-Identifier: Apache-2.0
#
# AITER Sol-Attn attention backend (ROCm, gfx950).
#
# One backend, several quantization recipes selected through
# --attention-backend-config (e.g. `recipe=f8f6`). The recipes differ only in
# how Q/K/V are quantized; the routing, the pooled correction and the geometry
# are the same for all of them.
#
# Unrelated to `sol_attn.py`, which wraps the NVIDIA `sol_attn` package.

from __future__ import annotations

from numbers import Integral

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sol_dispatch import (
    SOL_ATTN_HEAD_DIM,
    SOL_ATTN_RECIPES,
    SolAttnUnsupported,
    check_sol_attn_build,
    check_sol_attn_values,
    sol_attn_bshd,
    sol_attn_routing_mode,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Every Sol-Attn manifest row is head_dim 128. Selecting this backend is
# explicit, so unmet constraints raise rather than silently falling back.

_DEFAULT_RECIPE = "fp8"
_DEFAULT_BETA = 0.5


class _BetaSchedule(msgspec.Struct, frozen=True):
    """The routing threshold per denoising step.

    Either two endpoints to ramp between or one value per step. A larger beta
    raises the threshold, so it keeps fewer blocks exact and leans harder on
    the pooled correction.
    """

    endpoints: tuple[float, float] | None = None
    values: tuple[float, ...] = ()

    @classmethod
    def from_spec(cls, spec: str) -> _BetaSchedule:
        text = spec.strip()
        if ":" in text:
            first, last = text.split(":", 1)
            return cls(endpoints=(float(first), float(last)))
        values = tuple(float(part) for part in text.split(",") if part.strip())
        if not values:
            raise ValueError(
                "beta_schedule must be 'first:last' for a linear ramp or one "
                f"comma-separated value per step, got {spec!r}."
            )
        if len(values) == 1:
            return cls(endpoints=(values[0], values[0]))
        return cls(values=values)

    def beta_at(self, step: int, total_steps: int) -> float:
        if self.values:
            return self.values[min(step, len(self.values) - 1)]
        first, last = self.endpoints
        if total_steps <= 1:
            return first
        position = min(step, total_steps - 1) / (total_steps - 1)
        return first + (last - first) * position


class _SolConfig(msgspec.Struct, frozen=True):
    recipe: str
    beta: float
    beta_schedule: _BetaSchedule | None
    block_tile: tuple[int, int] | None


def _parse_block_tile(value: str | None) -> tuple[int, int] | None:
    """Read a `block_size` of the form 128x128 (or 128,128)."""
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", "x")
    parts = text.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"block_size must be <q_tile>x<kv_tile>, e.g. 128x128, got {value!r}."
        )
    return (int(parts[0]), int(parts[1]))


def _resolve_config() -> _SolConfig:
    """Read the aiter_sol keys from --attention-backend-config."""
    cfg = get_global_server_args().attention_backend_config or {}
    recipe = str(cfg.get("recipe", _DEFAULT_RECIPE)).strip().lower()
    if recipe not in SOL_ATTN_RECIPES:
        raise ValueError(
            f"Unknown aiter_sol recipe {recipe!r}. Set "
            "--attention-backend-config recipe=<name> to one of: "
            f"{', '.join(SOL_ATTN_RECIPES)}."
        )
    schedule = cfg.get("beta_schedule")
    return _SolConfig(
        recipe=recipe,
        beta=float(cfg.get("beta", _DEFAULT_BETA)),
        beta_schedule=(
            None if schedule is None else _BetaSchedule.from_spec(str(schedule))
        ),
        block_tile=_parse_block_tile(cfg.get("block_size")),
    )


class AITERSolBackend(AttentionBackend):
    """AITER Sol-Attn block-sparse attention backend (ROCm)."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [SOL_ATTN_HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_SOL

    @staticmethod
    def get_impl_cls() -> type[AITERSolImpl]:
        return AITERSolImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        # Sol-Attn routes from Q/K alone and needs no spatial metadata.
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError(
            "AITER Sol-Attn backend does not have a metadata builder."
        )


class AITERSolImpl(AttentionImpl):
    """Sol-Attn via aiter's mha_v4 mode-2 rows, with the recipe selected by the
    `recipe` key of --attention-backend-config (default fp8).

    Other keys: `beta` (routing threshold, default 0.5), `beta_schedule`
    (`first:last` or one value per denoising step) and `block_size`
    (`<q_tile>x<kv_tile>`, default the recipe's own geometry).
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        check_sol_attn_values(head_dim=head_size, is_causal=causal)
        config = _resolve_config()
        check_sol_attn_build(config.recipe, block_tile=config.block_tile)
        self.recipe = config.recipe
        self.block_tile = config.block_tile
        self.beta = config.beta
        self.beta_schedule = config.beta_schedule
        self.softmax_scale = softmax_scale

        logger.info_once(
            "aiter_sol attention backend using recipe=%s beta=%s block_size=%s "
            "routing=%s.",
            self.recipe,
            self.beta if self.beta_schedule is None else "scheduled",
            "default" if self.block_tile is None else self.block_tile,
            sol_attn_routing_mode(self.recipe),
        )

    def _total_steps(self) -> int | None:
        """The denoising step count this request runs, or None when no request
        is in scope. Many stages set a forward context without one."""
        context = get_forward_context_or_none()
        if context is None or context.forward_batch is None:
            return None
        sampling_params = context.forward_batch.sampling_params
        if sampling_params is None or sampling_params.num_inference_steps is None:
            return None
        return int(sampling_params.num_inference_steps)

    def _current_beta(self) -> float:
        """This forward's routing threshold.

        A scheduled beta is a different Python float on every step, so it is
        baked as a new constant into each traced graph; keep the schedule for
        eager runs until the step index is published as a device tensor.
        """
        if self.beta_schedule is None:
            return self.beta
        context = get_forward_context_or_none()
        if context is None:
            return self.beta
        step = context.current_timestep
        if not isinstance(step, Integral):
            return self.beta
        total_steps = self._total_steps()
        if total_steps is None:
            # A per-step list indexes without knowing the count; a ramp cannot
            # place a step without one, so it keeps the constant.
            if not self.beta_schedule.values:
                return self.beta
            total_steps = len(self.beta_schedule.values)
        return self.beta_schedule.beta_at(int(step), total_steps)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Route the KV blocks, compute the selected ones exactly, and recover
        the rest from pooled K/V.

        Q/K/V are [batch_size, seq_len, num_heads, head_dim] bf16. Key and value
        may carry a different seq_len than query (cross attention) and fewer
        heads than query (GQA). Returns the query shape.
        """
        del attn_metadata
        output, _ = sol_attn_bshd(
            query,
            key,
            value,
            beta=self._current_beta(),
            softmax_scale=self.softmax_scale,
            recipe=self.recipe,
            block_tile=self.block_tile,
        )
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int | None = None,
    ) -> torch.Tensor:
        """Serve a packed call that is one sequence followed by a trailing pad.

        Sol-Attn takes no key-padding mask, and its pooled blocks would straddle
        any boundary other than the end, so a pack of several sequences is
        refused rather than attended over its own padding.
        """
        del max_seqlen_k
        used = (
            None
            if cu_seqlens_host is None
            else trailing_padding_used_len(
                total_tokens=query.shape[0],
                max_seqlen=max_seqlen,
                bounds=cu_seqlens_host,
            )
        )
        if used is None or cu_seqlens_k is not None:
            raise SolAttnUnsupported(
                "AITER Sol-Attn backend serves a packed call only as a single "
                "sequence with a trailing pad; it has no key-padding mask."
            )

        live_output, _ = sol_attn_bshd(
            query[:used].unsqueeze(0),
            key[:used].unsqueeze(0),
            value[:used].unsqueeze(0),
            beta=self._current_beta(),
            softmax_scale=self.softmax_scale,
            recipe=self.recipe,
            block_tile=self.block_tile,
        )
        live_output = live_output.squeeze(0)
        if used == query.shape[0]:
            return live_output
        output = live_output.new_zeros(query.shape[0], query.shape[1], value.shape[2])
        output[:used] = live_output
        return output


__all__ = [
    "AITERSolBackend",
    "AITERSolImpl",
    "SolAttnUnsupported",
]
