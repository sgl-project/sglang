from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Optional

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import (
        BaseIndexerMetadata,
    )
    from sglang.srt.layers.attention.verify_mask import VerifyMask
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


class SharedReadEnds(Enum):
    """Where an attention backend finishes reading the shared data"""

    PRE_REPLAY = 1  # After the init_forward_metadata_out_graph
    IN_REPLAY = 2  # After the init_forward_metadata_in_graph
    POST_REPLAY = 3  # Metadata snapshot not implemented
    UNKNOWN = 4  # not audited -> coarse whole-forward fence

    @staticmethod
    def max_of(items: Iterable[SharedReadEnds]) -> SharedReadEnds:
        # Ordered by lateness: the latest end covers every child.
        return max(items, key=lambda x: x.value)


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

    # Resolved per-mode backend names, stamped by ModelRunner.init_attention_backend
    prefill_attention_backend_str: Optional[str] = None
    decode_attention_backend_str: Optional[str] = None

    supports_ragged_verify_graph: bool = False
    # Compute / KV-cache dtype. Only backends that need them (MLA/MHA fp8
    # fuse-rope checks) set these in __init__; declared here as None so callers
    # can read them off ANY backend — including hybrid wrappers that don't set
    # them — without defensive getattr. See trtllm_mla fuse-rope path.
    data_type: Optional[torch.dtype] = None
    kv_cache_dtype: Optional[torch.dtype] = None

    # Wrapper backends (e.g. HybridLinearAttnBackend) set this to their child
    # backends; leaves keep None. Lets generic code (metadata glue graph)
    # enumerate every backend whose python-side forward_metadata must be
    # snapshotted/restored around a captured metadata-prep replay.
    attn_backend_list: Optional[list] = None

    # Per-iter metadata produced by init_forward_metadata*; backends that use
    # it assign their own type. Declared here so generic snapshot/restore code
    # (metadata glue graph) can read it off any backend without hasattr.
    forward_metadata: Optional[object] = None

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

    def draft_extend_metadata_captured_in_graph(self) -> bool:
        """True when :py:meth:`init_forward_metadata_in_graph` fully rebuilds
        this backend's DRAFT_EXTEND_V2 replay metadata inside the captured
        graph, so a replaying runner may skip the eager
        :py:meth:`init_forward_metadata_out_graph` call."""
        return False

    # Opt out only when this backend never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = True

    # True for backends that preallocate per-seq extend metadata at req-pool
    # size (e.g. triton's kv_indptr): dummy extend batches must then keep
    # batch_size <= req_to_token_pool.size.
    extend_dummy_seqs_capped_by_req_pool: bool = False

    # Most attention backends can rebuild and replace forward metadata before
    # every forward. BCG capture is different: some backends expose metadata
    # tensors to kernels across graph breaks, so the captured graph depends on
    # those tensor addresses. Such backends opt in here, create the metadata
    # object during capture, and refresh its dynamic fields before each replay.
    use_captured_forward_metadata_for_breakable_cuda_graph: bool = False

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        """Declare where this backend's scheduler-shared reads end per mode.
        Override only for audited deviations from this conservative default."""
        if fm.is_decode() or fm.is_target_verify():
            return SharedReadEnds.IN_REPLAY
        return SharedReadEnds.UNKNOWN

    def prepare_prefill_shared_read_snapshot(
        self, forward_batch: ForwardBatch, *, num_qo_tokens: int
    ) -> None:
        """Snapshot late prefill reads before a PRE_REPLAY event is published.

        Runners call this only after the actual eager/replay query geometry is
        known. Backends that retain scheduler-shared reads into the model
        forward keep the default no-op and must not declare PRE_REPLAY.
        """

    # Chunked-prefix FullCG capture has a second model topology and stable
    # prefix buffers. Backends must opt in explicitly so the runner does not
    # assume that generic ForwardBatch metadata is sufficient for every
    # attention implementation.
    supports_full_cuda_graph_chunked_prefix: bool = False

    def prepare_full_cuda_graph_chunked_prefix(
        self,
        forward_batch: ForwardBatch,
        *,
        in_capture: bool,
    ) -> None:
        """Prepare backend-private metadata for chunked-prefix FullCG.

        Only called for backends that set
        ``supports_full_cuda_graph_chunked_prefix``; the runner validates the
        flag up front. The runner owns and refreshes the shared ForwardBatch
        prefix buffers. Backends that need wrappers or other derived metadata
        should override this hook for both capture and replay.
        """

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self,
        forward_batch: ForwardBatch,
    ):
        """Create forward metadata whose tensor addresses will be graph-captured."""
        raise NotImplementedError()

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:
        """Refresh captured metadata for the current batch before BCG replay.

        Implementations should update ``capture_metadata`` in place where graph
        address stability is required, assign any safe per-replay objects, and
        make the backend's active ``forward_metadata`` point to the captured
        metadata object.
        """
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

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        """The mask the draft stage fills in place, if this backend has one."""
        return None

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
