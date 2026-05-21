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

    Per-forward init three-method contract (initial stage of step 03):

        init_forward_data(fb)            -- eager wrapper = out_graph + in_graph
        init_forward_data_out_graph(fb)  -- per-iter metadata prep run OUTSIDE
                                            any cuda graph capture/replay scope.
                                            All three paths (eager / capture /
                                            replay) call this; capture and
                                            replay run the same body.
        init_forward_data_in_graph(fb)   -- graph-recordable metadata prep run
                                            INSIDE `with graph.capture():`.
                                            Recorded ops auto-execute at
                                            replay via `graph.replay()`.
                                            Default no-op; most backends do
                                            NOT override this in the initial
                                            stage (deferred optimization slot).

    The five backend init paths (eager / full-graph capture / full-graph
    replay / PCG capture / PCG replay) all flow through these methods. Only
    the caller varies which subset it invokes:

        eager        : init_forward_data(fb)         == out_graph + in_graph
        full capture : out_graph(fb) outside; in_graph(fb) inside graph.
        full replay  : out_graph(fb_view) outside; graph.replay() auto-runs
                       recorded in_graph ops.
        PCG capture  : out_graph(fb) outside; in_graph(fb) inside.
        PCG replay   : out_graph(fb) outside; graph.replay() auto-runs.

    Initial stage: backends override only ``init_forward_data_out_graph``
    with the merged body of the old ``init_forward_metadata`` (eager) +
    ``init_forward_metadata_capture_cuda_graph`` (capture) +
    ``init_forward_metadata_replay_cuda_graph`` (replay) variants. The
    ``_in_graph`` slot is reserved for a follow-up per-backend optimization
    PR that moves graph-safe static-shape ops into the captured graph for
    Python-dispatch savings at replay.
    """

    # -----------------------------------------------------------------
    # Forward-data init -- three-method contract
    # -----------------------------------------------------------------

    def init_forward_data_out_graph(self, forward_batch: ForwardBatch) -> None:
        """Per-iter metadata prep run OUTSIDE any cuda graph capture/replay scope.

        Called at eager (via the wrapper), capture (before the graph capture
        context), and replay (before ``graph.replay()``). All three paths run
        the same body; this is where the unified body lives in the initial
        stage of step 03.

        Default: no-op. Override this with the backend's per-iter prep.
        """

    def init_forward_data_in_graph(self, forward_batch: ForwardBatch) -> None:
        """Graph-recordable metadata prep run INSIDE ``with graph.capture():``.

        Recorded ops auto-execute at replay via ``graph.replay()`` -- no
        Python dispatch at replay time.

        Default: no-op. Most backends do NOT override this in the initial
        stage; all prep stays in ``_out_graph``. A follow-up per-backend PR
        moves graph-safe static-shape ops here for replay-time speedup.

        Override contract: body must NOT call ``.item()`` / ``.cpu()`` /
        ``.tolist()`` / dynamic-shape ``torch.empty()`` -- those are not
        recordable and belong in ``_out_graph``.
        """

    def init_forward_data(self, forward_batch: ForwardBatch) -> None:
        """Eager-path wrapper -- runs both phases in order."""
        self.init_forward_data_out_graph(forward_batch)
        self.init_forward_data_in_graph(forward_batch)

    # -----------------------------------------------------------------
    # Deprecation shims for out-of-tree backends (one release window)
    # -----------------------------------------------------------------
    #
    # Out-of-tree subclasses that still override the old method names continue
    # to work via the forwarders below -- at the cost of a DeprecationWarning
    # per call site. The capture/replay variants had positional-arg signatures
    # incompatible with fb-only, so they raise NotImplementedError instead:
    # out-of-tree code that uses them must migrate to ``_out_graph``.

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        """[DEPRECATED] Old eager init entry point.

        Default forwards to the new three-method contract. Subclasses should
        override ``init_forward_data_out_graph`` (and optionally
        ``init_forward_data_in_graph``) instead of this method.
        """
        warnings.warn(
            "AttentionBackend.init_forward_metadata is deprecated; "
            "override init_forward_data_out_graph instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.init_forward_data(forward_batch)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        """[DEPRECATED] Old capture-time init entry point -- not auto-forwarded.

        Signature differed from the eager method, so callers must migrate to
        ``init_forward_data_out_graph(fb)`` explicitly. Out-of-tree backends
        overriding this method must move the body into
        ``init_forward_data_out_graph``.
        """
        raise NotImplementedError(
            "init_forward_metadata_capture_cuda_graph is deprecated; "
            "override init_forward_data_out_graph instead."
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        """[DEPRECATED] Old replay-time init entry point -- not auto-forwarded.

        Signature differed from the eager method, so callers must migrate to
        ``init_forward_data_out_graph(fb_view)`` explicitly. Out-of-tree
        backends overriding this method must move the body into
        ``init_forward_data_out_graph``.
        """
        raise NotImplementedError(
            "init_forward_metadata_replay_cuda_graph is deprecated; "
            "override init_forward_data_out_graph instead."
        )

    # -----------------------------------------------------------------
    # Cuda-graph buffer alloc + warmup hook (unchanged by step 03)
    # -----------------------------------------------------------------

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
        pass

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers of verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        """
        Update the buffers returned by get_verify_fill_after_draft_buffers if needed.

        Here, we need to redo the computation of all metadata of the attention backend
        that depends on tree mask and position buffers.
        """
        raise NotImplementedError()

    @debug_kernel_api
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
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
        layer: RadixAttention,
        forward_batch: ForwardBatch,
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
        layer: RadixAttention,
        forward_batch: ForwardBatch,
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
        layer: RadixAttention,
        forward_batch: ForwardBatch,
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
        forward_batch: ForwardBatch,
    ) -> Optional[BaseIndexerMetadata]:
        """Get the indexer metadata. None means don't support indexer."""
        return None
