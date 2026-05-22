from __future__ import annotations

import warnings
from abc import ABC
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpecInput


class AttentionBackend(ABC):
    """The base class of attention backends.

    Init lifecycle:

        construct        : __init__(model_runner)            — bind pool refs (step 02)
        capture-state    : init_cuda_graph_state(max_bs, max_num_tokens)
                                                             — alloc backend-private buffers
        per-forward init : init_forward_data(fb)             — eager wrapper
                           init_forward_data_out_graph(fb)   — host prep, NOT graph-safe
                           init_forward_data_in_graph(fb)    — GPU ops only, graph-safe (future opt)
        warmup hook      : on_after_cuda_graph_warmup()      — reset dirtied state

        forward kernel   : forward(q, k, v, layer, fb, ...)
    """

    # ------------------------------------------------------------------
    # Per-forward init — three-method contract (A3 invariant)
    # ------------------------------------------------------------------

    def init_forward_data_out_graph(self, forward_batch: "ForwardBatch") -> None:
        """Per-iter metadata prep — called outside cuda graph capture/replay scope.

        Runs at eager time (via init_forward_data wrapper), at graph capture
        time (before ``with graph.capture():``), and at replay time (before
        ``graph.replay()``).  All three paths run the same body.

        This is where the entire body goes in the initial-stage migration;
        the _in_graph slot below is reserved for a future per-backend
        graph-recording optimization.

        Default: no-op.
        """

    def init_forward_data_in_graph(self, forward_batch: "ForwardBatch") -> None:
        """Graph-recordable metadata prep — called inside ``with graph.capture():``
        at capture time; recorded ops auto-replay via ``graph.replay()``.

        Default: no-op.  Most backends do NOT override this in the initial
        stage — all prep stays in init_forward_data_out_graph (correct but
        slower: per-iter Python dispatch instead of recorded once).  A
        follow-up per-backend PR moves graph-safe static-shape ops here.

        Override contract: body must NOT call ``.item()`` / ``.cpu()`` /
        ``.tolist()`` / dynamic-shape ``torch.empty()``.  Those ops cannot
        be recorded into a cuda graph and belong in init_forward_data_out_graph.
        """

    def init_forward_data(self, forward_batch: "ForwardBatch") -> None:
        """Eager path wrapper — runs both phases in sequence."""
        self.init_forward_data_out_graph(forward_batch)
        self.init_forward_data_in_graph(forward_batch)

    # ------------------------------------------------------------------
    # Deprecation shims — one-release backward-compat window
    #
    # init_forward_metadata        → use init_forward_data
    # init_forward_metadata_capture_cuda_graph  \
    # init_forward_metadata_replay_cuda_graph    → override init_forward_data_out_graph
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Deprecated: use init_forward_data."""
        warnings.warn(
            "init_forward_metadata is deprecated; use init_forward_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.init_forward_data(forward_batch)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs) -> None:
        """Removed. Override init_forward_data_out_graph instead."""
        raise NotImplementedError(
            "init_forward_metadata_capture_cuda_graph is removed. "
            "Override init_forward_data_out_graph(self, forward_batch) instead."
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs) -> None:
        """Removed. Override init_forward_data_out_graph instead."""
        raise NotImplementedError(
            "init_forward_metadata_replay_cuda_graph is removed. "
            "Override init_forward_data_out_graph(self, forward_batch) instead."
        )

    # ------------------------------------------------------------------
    # Graph lifecycle hooks (unchanged)
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        raise NotImplementedError()

    def on_after_cuda_graph_warmup(self):
        """Hook between cuda graph warmup pass and the actual capture.

        Override to undo state that warmup mutated or eagerly advanced
        (e.g. dirty metadata buffers, raw->full upgrades) before capture
        freezes the kernel pointers.
        """

    # ------------------------------------------------------------------
    # Spec-runner helpers (unchanged)
    # ------------------------------------------------------------------

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers of verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: "SpecInput", cuda_graph_bs: Optional[int]
    ):
        """
        Update the buffers returned by get_verify_fill_after_draft_buffers if needed.

        Here, we need to redo the computation of all metadata of the attention backend
        that depends on tree mask and position buffers.
        """
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # Forward kernel dispatch (unchanged)
    # ------------------------------------------------------------------

    @debug_kernel_api
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode.is_mixed() and is_npu():
            return self.forward_mixed(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
    ):
        """Run a forward for mix."""
        raise NotImplementedError()

    def support_triton(self):
        """Check if the current backend supports triton."""
        return True

    def get_indexer_metadata(
        self,
        layer_id: int,
        forward_batch: "ForwardBatch",
    ) -> Optional["BaseIndexerMetadata"]:
        """Get the indexer metadata. None means don't support indexer."""
        return None
