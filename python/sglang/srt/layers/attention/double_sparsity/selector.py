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
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.double_sparsity.channel_mask import ChannelMask
    from sglang.srt.layers.attention.double_sparsity.config import (
        DoubleSparsityConfig,
    )
    from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
        PageSignatureTable,
    )


class DoubleSparsitySelector:
    """Sequence-order-ascending top-K logical-page selector.

    Two modes:

    * **Placeholder** (default after construction). ``retrieve_topk`` returns
      deterministic ascending logical-page IDs so the downstream FlashMLA
      wiring is exercisable in unit tests before real selection kernels and
      a real channel mask are wired. Production serving is refused by the
      placeholder guard unless ``SGLANG_DS_ALLOW_PLACEHOLDER=1`` is set.

    * **Real** — after :meth:`bind_runtime_data` is called with a populated
      :class:`PageSignatureTable` and a loaded :class:`ChannelMask`, the
      selector switches to the real score → all-reduce → top-K flow from
      :mod:`selection_kernel`. ``IS_PLACEHOLDER`` flips to ``False``.

    The class does NOT inherit from any HiSparse base and is NOT registered
    in ``_ALGORITHM_REGISTRY``.
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

        self.page_signature_table: Optional["PageSignatureTable"] = None
        self.channel_mask: Optional["ChannelMask"] = None
        self.process_group = None
        self.IS_PLACEHOLDER = True

    def bind_runtime_data(
        self,
        page_signature_table: "PageSignatureTable",
        channel_mask: "ChannelMask",
        *,
        process_group=None,
    ) -> None:
        """Switch the selector from placeholder to real mode.

        Both arguments must be non-None and shape-compatible. Subsequent
        calls to ``retrieve_topk`` use the real score → all-reduce → top-K
        flow from :mod:`selection_kernel`.
        """

        if page_signature_table is None:
            raise ValueError("page_signature_table is required for real selection.")
        if channel_mask is None:
            raise ValueError("channel_mask is required for real selection.")
        if channel_mask.label_dim != page_signature_table.label_dim:
            raise ValueError(
                f"channel_mask.label_dim={channel_mask.label_dim} does not match "
                f"page_signature_table.label_dim={page_signature_table.label_dim}."
            )
        if page_signature_table.num_heads_local != self.num_local_heads:
            raise ValueError(
                f"page_signature_table.num_heads_local={page_signature_table.num_heads_local} "
                f"does not match selector.num_local_heads={self.num_local_heads}."
            )
        mask_num_heads = int(channel_mask.channel_selection.shape[1])
        if mask_num_heads != self.num_local_heads:
            raise ValueError(
                f"channel_mask num_heads={mask_num_heads} does not match "
                f"selector.num_local_heads={self.num_local_heads}. The calibration "
                "artifact is TP-agnostic (H_full); call "
                "channel_mask.slice_per_rank(mask, num_local_heads=..., rank=..., "
                "tp_size=...) before bind_runtime_data."
            )

        # The mask is typically loaded from disk on CPU while the page
        # signature table + queries live on the selector's device. Align the
        # mask tensors to the table's device so retrieve_topk's torch.gather
        # and weight-multiply don't trip a device mismatch.
        target_device = page_signature_table.signatures.device
        if channel_mask.channel_selection.device != target_device:
            channel_mask = replace(
                channel_mask,
                channel_selection=channel_mask.channel_selection.to(target_device),
                channel_weights=channel_mask.channel_weights.to(target_device),
            )

        self.page_signature_table = page_signature_table
        self.channel_mask = channel_mask
        self.process_group = process_group
        self.IS_PLACEHOLDER = False

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        hot_pages: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(selected_indices, valid_lengths)``.

        ``selected_indices``: int32 ``[bs, max_top_k]``, logical page IDs in
            **sequence-order ascending**, ``-1`` padded.
        ``valid_lengths``: int32 ``[bs]``, the unpadded length of each row.

        Dispatches to the real :mod:`selection_kernel` pipeline once
        :meth:`bind_runtime_data` has installed a populated page signature
        table and channel mask; otherwise runs the placeholder ascending-IDs
        scheme so downstream wiring is exercisable in unit tests.
        """

        if queries.dim() < 2:
            raise ValueError(
                f"Double Sparsity expects queries with at least 2 dims, got shape {tuple(queries.shape)}."
            )

        if self.page_signature_table is not None and self.channel_mask is not None:
            from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
                retrieve_topk_via_signatures,
            )

            return retrieve_topk_via_signatures(
                queries=queries,
                page_signatures=self.page_signature_table.signatures,
                valid_mask=self.page_signature_table.valid_mask,
                channel_selection=self.channel_mask.channel_selection,
                channel_weights=self.channel_mask.channel_weights,
                layer_id=layer_id,
                max_top_k=self.max_top_k,
                hot_pages=hot_pages,
                process_group=self.process_group,
                per_request_valid=sparse_mask,
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
