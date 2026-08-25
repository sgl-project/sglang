import contextlib
from typing import Iterator, Optional

import msgspec
import torch


class ZeroCopyContext(msgspec.Struct, frozen=True):
    moe_output: Optional[torch.Tensor] = None


ctx = ZeroCopyContext()


@contextlib.contextmanager
def set_moe_output(out: torch.Tensor) -> Iterator[None]:
    """Publish `out` as the MoE runner's output destination for the block."""
    old_output = ctx.moe_output
    msgspec.structs.force_setattr(ctx, "moe_output", out)
    try:
        yield
    finally:
        msgspec.structs.force_setattr(ctx, "moe_output", old_output)


def get_moe_output(ref: torch.Tensor) -> Optional[torch.Tensor]:
    """The published destination iff it can stand in for empty_like(ref)."""
    return get_moe_output_spec(ref.shape, ref.dtype, ref.device)


def get_moe_output_spec(
    shape: torch.Size, dtype: torch.dtype, device: torch.device
) -> Optional[torch.Tensor]:
    """Spec form of get_moe_output for callers that would otherwise have to
    materialize a reference tensor just for the match (e.g. the trtllm-gen
    runner, whose activation input is fp4-packed and shaped differently
    from its output)."""
    out = ctx.moe_output
    if (
        out is not None
        and out.shape == shape
        and out.dtype == dtype
        and out.device == device
        and out.is_contiguous()
    ):
        return out
    return None


__all__ = [
    "ZeroCopyContext",
    "ctx",
    "set_moe_output",
    "get_moe_output",
    "get_moe_output_spec",
]
