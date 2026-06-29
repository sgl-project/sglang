"""Double Sparsity selector — holds the bound runtime selection state.

The selector owns the per-rank bound state (channel mask, absorbed projection,
process group) installed by ``bind_runtime_data`` at startup. Until then it is
"unbound" (``IS_PLACEHOLDER``), and ``assert_real_selector_or_placeholder_allowed``
refuses to serve traffic — production must bind the real ``ChannelMask`` first.

Production selection runs through ``selection_kernel.retrieve_topk_graph_safe``
(called directly from ``deepseek_v2``); the eager :meth:`retrieve_topk` here is a
unit-test reference path. The class does NOT inherit from any HiSparse base and
is NOT registered in ``_ALGORITHM_REGISTRY``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.double_sparsity.channel_mask import ChannelMask
    from sglang.srt.layers.attention.double_sparsity.config import (
        DoubleSparsityConfig,
    )


class DoubleSparsityRebindError(RuntimeError):
    """Raised when ``bind_runtime_data`` is called a second time with
    different objects than the first call. Re-binding with the SAME
    objects (identity check) is a no-op; binding with different objects
    is a contract violation because CUDA-graph capture and TP setup
    cache references to the first set.
    """


class DoubleSparsityTPMisconfigured(RuntimeError):
    """Raised at startup or at ``bind_runtime_data`` time when a TP world
    size > 1 is detected with no ``process_group`` provided. A missing
    process group would silently turn the page-score all-reduce into a
    no-op and produce divergent ``selected_indices`` across ranks.
    """


class DoubleSparsitySelector:
    """Holds the bound DS selection state; ``IS_PLACEHOLDER`` until bound.

    Constructed unbound (``IS_PLACEHOLDER=True``); :meth:`bind_runtime_data`
    installs the loaded :class:`ChannelMask` + absorbed projection + process
    group and flips ``IS_PLACEHOLDER`` to ``False``. The eager :meth:`retrieve_topk`
    is a unit-test reference (production uses ``retrieve_topk_graph_safe``).
    """

    IS_PLACEHOLDER: bool = True

    def __init__(
        self,
        config: DoubleSparsityConfig,
        num_local_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.num_local_heads = num_local_heads
        self.head_dim = head_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_top_k = int(config.top_k)
        self.page_size = int(config.page_size)
        if self.max_top_k <= 0:
            raise ValueError(
                f"Double Sparsity max_top_k must be positive, got {self.max_top_k}."
            )

        self.channel_mask: Optional[ChannelMask] = None
        # Bind-time absorbed-latent projection [H, label_dim, kv_lora_rank], built
        # by deepseek_v2's bind path. Selection scores the resident MLA latent
        # through it.
        self.absorbed_w_sel: Optional[torch.Tensor] = None
        self.process_group = None
        # Custom all-reduce communicator for the score reduce (the attention-TP
        # coordinator's ca_comm under plain TP); None falls back to NCCL on the
        # raw process group.
        self.reduce_ca = None
        self.IS_PLACEHOLDER = True

    def bind_runtime_data(
        self,
        channel_mask: ChannelMask,
        *,
        process_group=None,
        reduce_ca=None,
    ) -> None:
        """Switch the selector from placeholder to real mode.

        Selection scores the resident MLA latent through the
        bind-time-installed ``absorbed_w_sel`` (the caller installs it before
        this call), so the only resident selection state is the channel mask.
        Subsequent calls to ``retrieve_topk`` use the real score → all-reduce →
        top-K flow from :mod:`selection_kernel`.

        Idempotence contract: a second call with the SAME object
        identities for ``channel_mask`` and ``process_group`` is a no-op. A
        second call with any different object raises
        :class:`DoubleSparsityRebindError` to prevent silent invalidation of
        CUDA-graph buffers and TP state captured on the first bind.
        """

        if not self.IS_PLACEHOLDER:
            same_objects = (
                self.channel_mask is channel_mask
                and self.process_group is process_group
                and self.reduce_ca is reduce_ca
            )
            if same_objects:
                return
            raise DoubleSparsityRebindError(
                "bind_runtime_data was called a second time with different "
                "objects. CUDA-graph capture and TP setup may have cached "
                "references to the first bind; swapping silently would "
                "corrupt those references. To rebind, construct a new "
                "DoubleSparsitySelector."
            )

        if channel_mask is None:
            raise ValueError("channel_mask is required for real selection.")
        if self.absorbed_w_sel is None:
            raise ValueError(
                "bind_runtime_data requires selector.absorbed_w_sel to be set "
                "first (the bind-time K-noPE W_UK projection). The caller installs "
                "it before bind_runtime_data."
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

        if channel_mask.channel_selection.device != self.device:
            channel_mask = replace(
                channel_mask,
                channel_selection=channel_mask.channel_selection.to(self.device),
                channel_weights=channel_mask.channel_weights.to(self.device),
            )
        if channel_mask.channel_selection.dtype != torch.int32:
            raise ValueError(
                "channel_mask.channel_selection must be int32, got "
                f"{channel_mask.channel_selection.dtype}."
            )
        if channel_mask.channel_weights.dtype != torch.float32:
            raise ValueError(
                "channel_mask.channel_weights must be float32, got "
                f"{channel_mask.channel_weights.dtype}."
            )

        self.channel_mask = channel_mask
        self.process_group = process_group
        self.reduce_ca = reduce_ca
        self.IS_PLACEHOLDER = False

        # Structured bind-time INFO log: operators rely on this to confirm
        # every DS-enabled layer has been bound on every TP rank.
        pg_size = 0
        pg_rank = 0
        if process_group is not None:
            try:
                pg_size = int(torch.distributed.get_world_size(group=process_group))
                pg_rank = int(torch.distributed.get_rank(group=process_group))
            except Exception:
                pg_size = 0
                pg_rank = 0
        logger.info(
            "double_sparsity bind_runtime_data completed: "
            "selector_id=%s num_local_heads=%d label_dim=%d page_size=%d "
            "process_group_size=%d process_group_rank=%d",
            id(self),
            self.num_local_heads,
            int(channel_mask.label_dim),
            self.page_size,
            pg_size,
            pg_rank,
            extra={
                "ds_selector_id": id(self),
                "ds_num_local_heads": self.num_local_heads,
                "ds_label_dim": int(channel_mask.label_dim),
                "ds_page_size": self.page_size,
                "ds_process_group_size": pg_size,
                "ds_process_group_rank": pg_rank,
            },
        )

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        req_to_token: Optional[torch.Tensor] = None,
        max_seq_len: int = 0,
        absorbed_latent_fp8: Optional[torch.Tensor] = None,
        absorbed_latent_scales: Optional[torch.Tensor] = None,
        absorbed_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(selected_indices, valid_lengths)``.

        ``selected_indices``: int32 ``[bs, max_top_k]``, logical token positions in
            **sequence-order ascending**, ``-1`` padded.
        ``valid_lengths``: int32 ``[bs]``, the unpadded length of each row.

        Dispatches to the real :mod:`selection_kernel` pipeline once
        :meth:`bind_runtime_data` has installed real runtime data: selection
        scores the absorbed-latent score off the resident MLA latent. The
        placeholder ascending-positions scheme runs until a bind, so downstream
        wiring is exercisable in unit tests.
        """

        if queries.dim() < 2:
            raise ValueError(
                f"Double Sparsity expects queries with at least 2 dims, got shape {tuple(queries.shape)}."
            )

        if self.channel_mask is not None:
            from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
                absorbed_topk_select,
            )

            return absorbed_topk_select(
                queries=queries,
                absorbed_w_sel=self.absorbed_w_sel,
                channel_selection_layer=self.channel_mask.channel_selection[layer_id],
                channel_weights_layer=self.channel_mask.channel_weights[layer_id],
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                max_top_k=self.max_top_k,
                absorbed_latent_fp8=absorbed_latent_fp8,
                absorbed_latent_scales=absorbed_latent_scales,
                absorbed_latent=absorbed_latent,
                per_request_valid=sparse_mask,
                process_group=self.process_group,
                reduce_ca=self.reduce_ca,
                score_reduce_bf16=(
                    getattr(self.config, "score_reduce_dtype", "bf16") == "bf16"
                ),
                head_agg=getattr(self.config, "head_agg", "max"),
                scorer_norm=getattr(self.config, "scorer_norm", "off"),
            )

        batch_size = req_pool_indices.shape[0]
        device = req_pool_indices.device

        if seq_lens is None:
            if sparse_mask is None or sparse_mask.dim() < 2:
                raise ValueError(
                    "Double Sparsity placeholder requires either explicit seq_lens or a "
                    "2-D sparse_mask of shape [bs, max_seq_tokens]."
                )
            token_lens = sparse_mask.to(torch.int32).sum(dim=-1)
        else:
            token_lens = seq_lens.to(torch.int32)

        if token_lens.shape[0] != batch_size:
            raise ValueError(
                f"Double Sparsity placeholder: token_lens batch size {token_lens.shape[0]} "
                f"does not match req_pool_indices batch size {batch_size}."
            )

        valid_lengths = torch.minimum(
            token_lens,
            torch.full_like(token_lens, self.max_top_k),
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


def assert_tp_configured(
    selector: DoubleSparsitySelector, *, tp_world_size: int
) -> None:
    """Fail fast at startup if TP world size > 1 but the selector has no
    process group.

    A missing process group would silently no-op the page-score
    all-reduce inside :mod:`selection_kernel` and produce divergent
    ``selected_indices`` across ranks. Refuse to serve.
    """
    if tp_world_size <= 1:
        return
    if getattr(selector, "process_group", None) is None:
        raise DoubleSparsityTPMisconfigured(
            f"Double Sparsity selector at tp_world_size={tp_world_size} "
            "must be bound with a non-None process_group. Without it the "
            "page-score all-reduce becomes a no-op and ranks diverge."
        )


def assert_real_selector_or_placeholder_allowed(
    selector: DoubleSparsitySelector,
) -> None:
    """Refuse to serve real traffic when the placeholder selector is live.

    ``bind_runtime_data`` is the only sanctioned way to flip a selector
    out of placeholder mode in production; tests that want to exercise a
    real selector without binding real runtime data can either call
    ``bind_runtime_data`` with synthetic page signatures and channel
    mask, or build a selector instance via ``object.__new__`` and toggle
    ``IS_PLACEHOLDER`` directly.
    """

    if not getattr(selector, "IS_PLACEHOLDER", False):
        return
    raise RuntimeError(
        "Double Sparsity is built with the placeholder selector. Refusing to serve "
        "production traffic. Call DoubleSparsitySelector.bind_runtime_data with the "
        "real ChannelMask before serving."
    )
