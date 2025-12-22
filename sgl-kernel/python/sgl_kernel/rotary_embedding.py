from __future__ import annotations

from typing import Literal, Optional

import torch

_HAS_COS_SIN = hasattr(torch.ops.sgl_kernel, "rotary_embedding_cos_sin")
_HAS_POSITION = hasattr(torch.ops.sgl_kernel, "rotary_embedding")


def _resolve_interleaved(interleaved: Optional[bool], is_neox: Optional[bool]) -> bool:
    if interleaved is not None and is_neox is not None and interleaved != is_neox:
        raise ValueError(
            f"is_neox({is_neox}) and interleaved({interleaved}) mismatch; keep only one or make them equal."
        )
    if is_neox is not None:
        return is_neox
    if interleaved is not None:
        return interleaved
    # Historical default in this wrapper: assume NeoX-style if not specified.
    return True


def apply_rotary_embedding(
    *,
    mode: Literal["cos_sin", "positions"],
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
    is_neox: Optional[bool] = None,
    # cos/sin mode
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    # positions mode
    positions: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
) -> None:
    effective_interleaved = _resolve_interleaved(interleaved, is_neox)

    if mode == "cos_sin":
        if cos is None or sin is None:
            raise ValueError("mode='cos_sin' requires cos and sin")

        if _HAS_COS_SIN:
            torch.ops.sgl_kernel.rotary_embedding_cos_sin(
                cos,
                sin,
                query,
                key if key is not None else None,
                head_size,
                effective_interleaved,
            )
            return

        if _HAS_POSITION:
            torch.ops.sgl_kernel.rotary_embedding(
                cos,
                sin,
                query,
                key if key is not None else None,
                head_size,
                effective_interleaved,
            )
            return

        raise RuntimeError(
            "No cos/sin rotary embedding kernel is available in torch.ops.sgl_kernel"
        )

    if mode == "positions":
        if positions is None or cos_sin_cache is None:
            raise ValueError("mode='positions' requires positions and cos_sin_cache")

        if not _HAS_POSITION:
            raise RuntimeError(
                "No positions rotary embedding kernel is available in torch.ops.sgl_kernel"
            )

        # Use positional args for maximum compatibility across builds.
        torch.ops.sgl_kernel.rotary_embedding(
            positions,
            query,
            key if key is not None else None,
            head_size,
            cos_sin_cache,
            effective_interleaved,
        )
        return

def rotary_embedding_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
    *,
    is_neox: Optional[bool] = None,
) -> None:
    apply_rotary_embedding(
        mode="cos_sin",
        cos=cos,
        sin=sin,
        query=query,
        key=key,
        head_size=head_size,
        interleaved=interleaved,
        is_neox=is_neox,
    )


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
    *,
    is_neox: Optional[bool] = None,
    cos_sin_cache: torch.Tensor,
) -> None:
    apply_rotary_embedding(
        mode="positions",
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        query=query,
        key=key,
        head_size=head_size,
        interleaved=interleaved,
        is_neox=is_neox,
    )
