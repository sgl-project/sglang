"""Double Sparsity selector — replaces NSA Indexer.forward selection role.

This is the **placeholder** implementation. It returns deterministic
sequence-order-ascending logical page IDs so that the FlashMLA block-table
plumbing in ``DeepseekV2AttentionMLA.forward_core`` can be wired and tested
end-to-end before real selection kernels land. The placeholder guard refuses
to serve real traffic unless ``SGLANG_DS_ALLOW_PLACEHOLDER=1`` is set.

The class does NOT inherit from any HiSparse base and is NOT registered in
``_ALGORITHM_REGISTRY``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.double_sparsity.config import (
        DoubleSparsityConfig,
    )


class DoubleSparsitySelector:
    """Sequence-order-ascending top-K logical-page selector.

    The real implementation (selection kernels + page signature projection)
    lands later. The placeholder picks the first ``top_k`` logical pages of
    each sequence so that downstream FlashMLA wiring is exercisable.
    """

    IS_PLACEHOLDER: bool = True

    def __init__(
        self,
        config: "DoubleSparsityConfig",
        num_local_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.num_local_heads = num_local_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_top_k = int(config.top_k)
        self.page_size = int(config.page_size)
        if self.max_top_k <= 0:
            raise ValueError(
                f"Double Sparsity max_top_k must be positive, got {self.max_top_k}."
            )

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(selected_indices, valid_lengths)``.

        ``selected_indices``: int32 ``[bs, max_top_k]``, logical page IDs in
            **sequence-order ascending**, ``-1`` padded.
        ``valid_lengths``: int32 ``[bs]``, the unpadded length of each row.

        Placeholder: pick the first ``min(num_pages, max_top_k)`` logical pages
        of each request. The active in-fill page is the last entry of the
        unpadded prefix by construction, satisfying the hot-page rule for the
        placeholder path.
        """

        if queries.dim() < 2:
            raise ValueError(
                f"Double Sparsity expects queries with at least 2 dims, got shape {tuple(queries.shape)}."
            )

        batch_size = req_pool_indices.shape[0]
        device = req_pool_indices.device

        if seq_lens is None:
            if sparse_mask is None or sparse_mask.dim() < 2:
                raise ValueError(
                    "Double Sparsity placeholder requires either explicit seq_lens or a "
                    "2-D sparse_mask of shape [bs, max_seq_pages]."
                )
            page_lens = sparse_mask.to(torch.int32).sum(dim=-1)
        else:
            page_lens = (
                (seq_lens.to(torch.int64) + self.page_size - 1) // self.page_size
            ).to(torch.int32)

        if page_lens.shape[0] != batch_size:
            raise ValueError(
                f"Double Sparsity placeholder: page_lens batch size {page_lens.shape[0]} "
                f"does not match req_pool_indices batch size {batch_size}."
            )

        valid_lengths = torch.minimum(
            page_lens,
            torch.full_like(page_lens, self.max_top_k),
        ).to(torch.int32)

        selected_indices = torch.full(
            (batch_size, self.max_top_k),
            -1,
            dtype=torch.int32,
            device=device,
        )
        position_grid = torch.arange(self.max_top_k, device=device, dtype=torch.int32)
        keep = position_grid.unsqueeze(0) < valid_lengths.unsqueeze(1)
        broadcast_positions = position_grid.unsqueeze(0).expand(batch_size, -1)
        selected_indices = torch.where(keep, broadcast_positions, selected_indices)

        return selected_indices, valid_lengths


def assert_real_selector_or_placeholder_allowed(selector: DoubleSparsitySelector) -> None:
    """Refuse to serve real traffic when the placeholder selector is live.

    Set ``SGLANG_DS_ALLOW_PLACEHOLDER=1`` to bypass — intended only for unit
    tests and smoke tests that exercise the wiring without depending on real
    selection kernels.
    """

    if not getattr(selector, "IS_PLACEHOLDER", False):
        return
    if os.environ.get("SGLANG_DS_ALLOW_PLACEHOLDER") == "1":
        return
    raise RuntimeError(
        "Double Sparsity is built with the placeholder selector. Refusing to serve "
        "production traffic. Land the real selection kernels and channel-mask "
        "projection before enabling this in production, or set "
        "SGLANG_DS_ALLOW_PLACEHOLDER=1 if this is an explicit test invocation."
    )
