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
    SOL_MIN_USEFUL_KV_BLOCKS,
    SolAttnUnsupported,
    check_sol_attn_build,
    check_sol_attn_values,
    check_sol_tilt,
    sol_attn_block_tile,
    sol_attn_bshd,
    sol_attn_routing_mode,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sol_schedule import (
    BetaSchedule,
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
_DEFAULT_FALLBACK_BACKEND = AttentionBackendEnum.AITER


class AITERSolMetadata(msgspec.Struct, frozen=True):
    """Per-forward state a model hands Sol-Attn through the forward context.

    exact_kv_tokens is a (seq_len,) bool mask over the packed sequence: every KV
    block holding a flagged key is computed exactly for every query.
    """

    exact_kv_tokens: torch.Tensor | None = None


class _SolConfig(msgspec.Struct, frozen=True):
    recipe: str
    beta: float
    beta_schedule: BetaSchedule | None
    block_tile: tuple[int, int] | None
    tilt_rank: int
    fallback_backend: AttentionBackendEnum = _DEFAULT_FALLBACK_BACKEND


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


def _parse_fallback_backend(value: str | None) -> AttentionBackendEnum:
    """Read `fallback_backend`: a dense backend for the calls Sol-Attn cannot serve."""
    if value is None:
        return _DEFAULT_FALLBACK_BACKEND
    name = str(value).strip().upper().replace("-", "_")
    try:
        backend = AttentionBackendEnum[name]
    except KeyError:
        raise ValueError(
            f"Unknown aiter_sol fallback_backend {value!r}. Set "
            "--attention-backend-config fallback_backend=<name> to a dense "
            "attention backend, e.g. aiter."
        ) from None
    if backend is AttentionBackendEnum.AITER_SOL or backend.is_sparse:
        raise ValueError(
            f"aiter_sol fallback_backend must be a dense backend, got {value!r}."
        )
    return backend


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
            None if schedule is None else BetaSchedule.from_spec(str(schedule))
        ),
        block_tile=_parse_block_tile(cfg.get("block_size")),
        tilt_rank=int(cfg.get("tilt_rank", 0)),
        fallback_backend=_parse_fallback_backend(cfg.get("fallback_backend")),
    )


class AITERSolBackend(AttentionBackend):
    """AITER Sol-Attn block-sparse attention backend (ROCm)."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [SOL_ATTN_HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_SOL

    @classmethod
    def dense_fallback_backend(cls) -> AttentionBackendEnum:
        return _resolve_config().fallback_backend

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
    (`first:last` or one value per denoising step), `block_size`
    (`<q_tile>x<kv_tile>`, default the recipe's own geometry), `tilt_rank`
    (pooled value tilt rank, default 0, which keeps the block means) and
    `fallback_backend` (default aiter).

    Calls Sol-Attn cannot serve run on `fallback_backend`: every call of a
    causal layer or of a head size other than 128, and single calls that are
    cross attention, not BF16, too short to route, or packed from several
    sequences.
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
        config = _resolve_config()
        self.fallback_backend = config.fallback_backend
        self._fallback_args = dict(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        self._fallback_impl: AttentionImpl | None = None
        self.serves_layer = head_size == SOL_ATTN_HEAD_DIM and not causal
        self._layer_fallback_reason = (
            "causal attention" if causal else f"head size {head_size}"
        )
        if not self.serves_layer:
            self._fallback(self._layer_fallback_reason)
            return
        check_sol_attn_values(head_dim=head_size, is_causal=causal)
        check_sol_attn_build(config.recipe, block_tile=config.block_tile)
        check_sol_tilt(config.recipe, config.tilt_rank)
        self.kv_tile = sol_attn_block_tile(config.recipe, config.block_tile)[1]
        self.recipe = config.recipe
        self.block_tile = config.block_tile
        self.beta = config.beta
        self.beta_schedule = config.beta_schedule
        self.tilt_rank = config.tilt_rank
        self.softmax_scale = softmax_scale

        logger.info_once(
            "aiter_sol attention backend using recipe=%s beta=%s block_size=%s "
            "tilt_rank=%d routing=%s.",
            self.recipe,
            self.beta if self.beta_schedule is None else "scheduled",
            "default" if self.block_tile is None else self.block_tile,
            self.tilt_rank,
            sol_attn_routing_mode(self.recipe, self.tilt_rank),
        )

    def _fallback(self, reason: str) -> AttentionImpl:
        """The fallback backend's impl for this layer, built on first use."""
        if self._fallback_impl is None:
            from sglang.multimodal_gen.runtime.layers.attention.selector import (
                _cached_get_attn_backend,
            )

            backend_cls = _cached_get_attn_backend(
                self._fallback_args["head_size"],
                torch.bfloat16,
                (),
                self.fallback_backend,
            )
            self._fallback_impl = backend_cls.get_impl_cls()(**self._fallback_args)
        logger.info_once(
            "aiter_sol runs %s on %s.", reason, self.fallback_backend.name.lower()
        )
        return self._fallback_impl

    def _fallback_reason(
        self, query: torch.Tensor, key: torch.Tensor, seqlen_q: int, seqlen_k: int
    ) -> str | None:
        """Why Sol-Attn cannot serve this call, or None when it can."""
        if not self.serves_layer:
            return self._layer_fallback_reason
        if seqlen_q != seqlen_k:
            return "cross attention"
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
            return f"{query.dtype} attention"
        if -(-seqlen_k // self.kv_tile) < SOL_MIN_USEFUL_KV_BLOCKS:
            return "sequences too short to route"
        return None

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

    def _current_beta(self) -> float | torch.Tensor:
        """Read the threshold prepared before the compiled model forward."""
        if self.beta_schedule is None:
            return self.beta
        context = get_forward_context_or_none()
        if context is None:
            return self.beta
        if context.sol_beta is not None:
            return context.sol_beta
        if torch.compiler.is_compiling():
            raise SolAttnUnsupported(
                "Compiled beta_schedule requires a denoising context with sol_beta."
            )
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

        Q/K/V are [batch_size, seq_len, num_heads, head_dim]. Key and value may
        carry fewer heads than query (GQA). A call Sol-Attn cannot serve runs on
        the fallback backend. Returns the query shape.
        """
        reason = self._fallback_reason(query, key, query.shape[1], key.shape[1])
        if reason is not None:
            return self._fallback(reason).forward(query, key, value, attn_metadata)
        del attn_metadata
        output, _ = sol_attn_bshd(
            query,
            key,
            value,
            beta=self._current_beta(),
            softmax_scale=self.softmax_scale,
            recipe=self.recipe,
            block_tile=self.block_tile,
            tilt_rank=self.tilt_rank,
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
        exact_kv_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Serve one sequence with optional trailing padding.

        exact_kv_tokens is an optional bool mask over the packed keys; see
        AITERSolMetadata.
        """
        del max_seqlen_k
        total_tokens = query.shape[0]
        if max_seqlen == total_tokens and cu_seqlens_host in (
            (0, total_tokens),
            (0, total_tokens, total_tokens),
        ):
            used = total_tokens
        else:
            used = (
                None
                if cu_seqlens_host is None
                else trailing_padding_used_len(
                    total_tokens=total_tokens,
                    max_seqlen=max_seqlen,
                    bounds=cu_seqlens_host,
                )
            )
        if cu_seqlens_k is not None:
            raise SolAttnUnsupported(
                "AITER Sol-Attn backend serves a packed call only as self "
                "attention over one sequence; it has no key-padding mask."
            )
        reason = (
            "packed calls of several sequences"
            if used is None
            else self._fallback_reason(query, key, used, used)
        )
        if reason is not None:
            return self._fallback(reason).forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        live_output, _ = sol_attn_bshd(
            query[:used].unsqueeze(0),
            key[:used].unsqueeze(0),
            value[:used].unsqueeze(0),
            beta=self._current_beta(),
            softmax_scale=self.softmax_scale,
            recipe=self.recipe,
            block_tile=self.block_tile,
            tilt_rank=self.tilt_rank,
            exact_kv_tokens=None if exact_kv_tokens is None else exact_kv_tokens[:used],
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
    "AITERSolMetadata",
    "SolAttnUnsupported",
]
