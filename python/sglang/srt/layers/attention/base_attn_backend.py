from __future__ import annotations

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

    Forward-data init contract (3 methods):

      - ``init_forward_metadata(fb)`` — eager entry point. Default is a wrapper
        that calls ``_out_graph(fb)`` then ``_in_graph(fb)``. Backends may
        override to keep an independent eager body.
      - ``init_forward_metadata_out_graph(fb, in_capture=False)`` — per-iter
        metadata prep, runs outside ``with graph.capture():``. Capture
        sites pass ``in_capture=True``; replay/eager use the default
        ``False``. Backends read ``in_capture`` only when capture / replay
        bodies diverge.
      - ``init_forward_metadata_in_graph(fb)`` — graph-recordable static-shape
        GPU op, runs inside ``with graph.capture():`` at capture time and
        is auto-replayed by ``graph.replay()``. Default is no-op.

    The legacy ``init_forward_metadata_capture_cuda_graph`` and
    ``init_forward_metadata_replay_cuda_graph`` overrides are fully
    deprecated and removed from the ABC: out-of-tree backends overriding
    those must migrate to ``init_forward_metadata_out_graph(fb, in_capture)``.
    """

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Eager entry point. Default = ``_out_graph(fb) + _in_graph(fb)``.

        Backends may override to keep an independent eager body.
        """
        self.init_forward_metadata_out_graph(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """Per-iter metadata prep — runs outside ``with graph.capture():``.

        Called at:
          * capture: before ``with graph.capture():`` (caller passes
            ``in_capture=True``).
          * replay: before ``graph.replay()`` (``in_capture=False``).
          * eager: via :py:meth:`init_forward_metadata` default wrapper
            (``in_capture=False``).

        Backends read ``in_capture`` only when capture / replay bodies
        diverge (e.g., snapshot metadata, swap buffer pointers, install
        temp workspace). Host op / dynamic-shape / non-graph-recordable
        logic lives here.

        Default: no-op.
        """

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable static-shape GPU op.

        Runs inside ``with graph.capture():`` at capture time; recorded
        ops auto-execute at replay via ``graph.replay()``.

        Lint contract for overrides: body must NOT call ``.item()`` /
        ``.cpu()`` / ``.tolist()`` / dynamic-shape ``torch.empty()``.
        Such ops belong in :py:meth:`init_forward_metadata_out_graph`; they
        cannot be recorded into a cuda graph.

        Default: no-op.
        """

    # Opt out only when this backend never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = True

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
        """Run forward on an attention layer.

        Dispatches on the shape class of ``forward_batch.forward_mode``:

        * ``IDLE`` — short-circuits to an empty tensor. This branch will
          be removed once :py:class:`RadixAttention.forward` takes over
          the IDLE short-circuit (see attention refactor step 09.d).
        * ``SINGLE_TOKEN`` (== ``DECODE``) — :py:meth:`forward_single_token`.
        * ``UNIFORM_LEN`` (NPU only, dispatched via the legacy
          ``is_mixed()`` predicate today) — :py:meth:`forward_uniform_len`.
        * everything else — :py:meth:`forward_var_len`.
        """
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_single_token():
            return self.forward_single_token(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        elif forward_batch.forward_mode.is_mixed() and is_npu():
            return self.forward_uniform_len(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_var_len(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    # ------------------------------------------------------------------
    # Canonical forward methods (attention refactor step 09)
    # ------------------------------------------------------------------
    # Backends should override the canonical names. The deprecated
    # ``forward_extend`` / ``forward_decode`` / ``forward_mixed`` overrides
    # below remain dispatchable for one release window: if a backend
    # overrides only the old name, the default canonical method forwards
    # to it.

    def forward_var_len(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for a variable-length batch (extend / chunked
        prefill / draft extend / split prefill / dllm extend /
        target verify on non-NPU backends)."""
        return self.forward_extend(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_single_token(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for a single-token batch (decode)."""
        return self.forward_decode(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_uniform_len(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for a uniform-length batch (NPU mixed dispatch).

        Default forwards to :py:meth:`forward_mixed` for back-compat with
        backends that still override the old name.
        """
        return self.forward_mixed(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Deprecated forward methods (one release window)
    # ------------------------------------------------------------------
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
        """Deprecated; override :py:meth:`forward_single_token` instead."""
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
        """Deprecated; override :py:meth:`forward_var_len` instead."""
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
        """Deprecated; override :py:meth:`forward_uniform_len` instead."""
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
