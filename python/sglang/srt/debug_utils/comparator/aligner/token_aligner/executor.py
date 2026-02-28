from __future__ import annotations

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.dims import (
    BATCH_DIM_NAME,
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
    TokenLayout,
    resolve_dim_by_name,
    strip_dim_names,
)
from sglang.srt.debug_utils.comparator.utils import Pair

_UNNAMED_TOKEN_DIM_FALLBACK: int = 0


def execute_token_aligner(
    plan: TokenAlignerPlan,
    tensor_of_step_pair: Pair[dict[int, torch.Tensor]],
) -> Pair[torch.Tensor]:
    flat_pair: Pair[dict[int, torch.Tensor]] = Pair(
        x=_collapse_bs_to_t(
            tensor_of_step=tensor_of_step_pair.x, layout=plan.layouts.x
        ),
        y=_collapse_bs_to_t(
            tensor_of_step=tensor_of_step_pair.y, layout=plan.layouts.y
        ),
    )

    if not plan.locators.x.steps:
        return Pair(
            x=_make_empty(tensor_of_step=flat_pair.x),
            y=_make_empty(tensor_of_step=flat_pair.y),
        )

    return Pair(
        x=_extract_and_stack_tokens(
            tensor_of_step=flat_pair.x, locator=plan.locators.x
        ),
        y=_extract_and_stack_tokens(
            tensor_of_step=flat_pair.y, locator=plan.locators.y
        ),
    )


# ── BS → T preprocessing ─────────────────────────────────────────


def _collapse_bs_to_t(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    layout: TokenLayout,
) -> dict[int, torch.Tensor]:
    """Collapse B and S dims into a single flat token dim (always batch-major).

    Handles both ``b s`` and ``s b`` orderings correctly via einops rearrange.
    Returns the original tensors unchanged if layout is T.
    """
    if layout != TokenLayout.BS:
        return tensor_of_step

    some_tensor: torch.Tensor = next(iter(tensor_of_step.values()))
    batch_dim: int = _resolve_dim_or_fallback(some_tensor, BATCH_DIM_NAME)
    seq_dim: int = _resolve_dim_or_fallback(some_tensor, SEQ_DIM_NAME)

    if abs(batch_dim - seq_dim) != 1:
        raise ValueError(
            f"BS dims must be adjacent: "
            f"{BATCH_DIM_NAME}={batch_dim}, "
            f"{SEQ_DIM_NAME}={seq_dim}"
        )

    lhs_pattern, rhs_pattern, new_names = _build_bs_collapse_pattern(
        names=list(some_tensor.names),
        batch_dim=batch_dim,
        seq_dim=seq_dim,
    )

    result: dict[int, torch.Tensor] = {}
    for step, tensor in tensor_of_step.items():
        plain: torch.Tensor = strip_dim_names(tensor)
        collapsed: torch.Tensor = rearrange(plain, f"{lhs_pattern} -> {rhs_pattern}")
        result[step] = collapsed.refine_names(*new_names)

    return result


def _build_bs_collapse_pattern(
    *,
    names: list[str | None],
    batch_dim: int,
    seq_dim: int,
) -> tuple[str, str, list[str | None]]:
    """Build einops lhs/rhs patterns and output dim names for BS→T collapse.

    Always produces batch-major order ``(b s)`` regardless of input ordering.
    Uses the tensor's own dim names as einops axis names.
    """
    lo: int = min(batch_dim, seq_dim)
    hi: int = max(batch_dim, seq_dim)

    lhs: str = " ".join(names)  # type: ignore[arg-type]

    rhs_names: list[str] = list(names[:lo]) + [f"({BATCH_DIM_NAME} {SEQ_DIM_NAME})"] + list(names[hi + 1 :])  # type: ignore[misc]
    rhs: str = " ".join(rhs_names)

    new_names: list[str | None] = (
        list(names[:lo]) + [TOKEN_DIM_NAME] + list(names[hi + 1 :])
    )

    return lhs, rhs, new_names


# ── core logic (T layout only) ───────────────────────────────────


def _resolve_dim_or_fallback(tensor: torch.Tensor, name: str) -> int:
    if tensor.names[0] is None:
        return _UNNAMED_TOKEN_DIM_FALLBACK
    return resolve_dim_by_name(tensor, name)


def _make_empty(*, tensor_of_step: dict[int, torch.Tensor]) -> torch.Tensor:
    dummy: torch.Tensor = next(iter(tensor_of_step.values()))
    token_dim: int = _resolve_dim_or_fallback(dummy, TOKEN_DIM_NAME)
    shape: list[int] = list(dummy.shape)
    shape[token_dim] = 0
    return torch.empty(shape, dtype=dummy.dtype)


def _extract_and_stack_tokens(
    *,
    tensor_of_step: dict[int, torch.Tensor],
    locator: TokenLocator,
) -> torch.Tensor:
    some_tensor: torch.Tensor = next(iter(tensor_of_step.values()))
    token_dim: int = _resolve_dim_or_fallback(some_tensor, TOKEN_DIM_NAME)

    tokens: list[torch.Tensor] = [
        strip_dim_names(tensor_of_step[s]).select(dim=token_dim, index=i)
        for s, i in zip(locator.steps, locator.token_index_in_step)
    ]
    return torch.stack(tokens, dim=token_dim)
