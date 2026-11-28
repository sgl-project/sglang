from __future__ import annotations

import torch

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op


class LocalTokenCountOp(BaseFusedOp):
    """Clamp a global token count to one rank's padded interval.

    ``global_count_cpu`` must mirror the scalar or one-element integer tensor
    when provided. Overrides preserve the native value, shape, dtype and device.
    Eager overrides use ``forward_native`` during tracing or graph capture.
    Registered compile overrides require ``enter_torch_compile()``.
    """

    op = "elementwise.local_token_count"

    def forward_native(
        self,
        global_count: torch.Tensor,
        rank_offset: int,
        tokens_per_rank: int,
        *,
        global_count_cpu: int | None = None,
    ) -> torch.Tensor:
        return torch.clamp(global_count - rank_offset, 0, tokens_per_rank)


_LOCAL_TOKEN_COUNT = register_fused_op(
    LocalTokenCountOp(), __name__, "_LOCAL_TOKEN_COUNT"
)


def local_token_count(
    global_count: torch.Tensor,
    rank_offset: int,
    tokens_per_rank: int,
    *,
    global_count_cpu: int | None = None,
) -> torch.Tensor:
    return _LOCAL_TOKEN_COUNT(
        global_count=global_count,
        rank_offset=rank_offset,
        tokens_per_rank=tokens_per_rank,
        global_count_cpu=global_count_cpu,
    )


def local_token_count_cpu(
    global_count: int, rank_offset: int, tokens_per_rank: int
) -> int:
    return min(max(global_count - rank_offset, 0), tokens_per_rank)


__all__ = ["LocalTokenCountOp", "local_token_count", "local_token_count_cpu"]
