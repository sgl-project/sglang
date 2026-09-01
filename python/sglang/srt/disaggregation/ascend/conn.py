import concurrent.futures
import enum
import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import build_transfer_entry_pairs
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


class AscendStateType(str, enum.Enum):
    """DSV4-on-NPU per-pool PD components, kept out of the cross-hardware
    StateType enum. Sent via the same page-indexed path as SWA."""

    DSV4_SWA = "dsv4_swa"
    DSV4_C4 = "dsv4_c4"
    DSV4_C128 = "dsv4_c128"
    DSV4_INDEXER = "dsv4_indexer"
    DSV4_C4_STATE = "dsv4_c4_state"
    DSV4_C128_STATE = "dsv4_c128_state"


_DSV4_KVCACHE_STATE_TYPES = tuple(AscendStateType)


class AscendKVManager(MooncakeKVManager):
    def _requires_exact_state_index_match(self, st: StateType) -> bool:
        return (
            super()._requires_exact_state_index_match(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )

    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        # MemFabric aligns registered buffers to 2 MiB. Register everything in
        # one batch so overlapping aligned ranges from small tensors are merged
        # before they are published to the peer.
        ptrs = list(self.kv_args.kv_data_ptrs)
        lens = list(self.kv_args.kv_data_lens)
        ptrs.extend(self.kv_args.aux_data_ptrs)
        lens.extend(self.kv_args.aux_data_lens)
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            ptrs.extend(component_ptrs)
            lens.extend(component_lens)
        if ptrs:
            self.engine.batch_register(ptrs, lens)

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int], state_type=None
    ) -> Tuple[List[int], List[int], int]:
        # src_kv_ptrs: k_data, v_data, index_k_data(optional)
        # dst_kv_ptrs: k_data, v_data, index_k_data(optional)
        # state_type is accepted for parity with the common disaggregation path;
        # the NPU kv_buf_groups slicing below is state-type agnostic.
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        hidden_kv_layers = getattr(self.kv_args, "hidden_kv_layers", 0)
        draft_kv_layers = getattr(self.kv_args, "draft_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        dst_layers = len(dst_kv_ptrs) // kv_buf_groups
        if src_layers == dst_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            start_layer = self.kv_args.prefill_start_layer
            transfer_draft_kv = get_pp_group().is_last_rank and draft_kv_layers
            if transfer_draft_kv:
                end_layer = start_layer + src_layers - draft_kv_layers
            else:
                end_layer = start_layer + src_layers

            # target kv
            for i in range(kv_buf_groups):
                layer_offset = i * hidden_kv_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
                )
            # draft kv
            if transfer_draft_kv:
                for i in range(kv_buf_groups):
                    layer_offset = (
                        i * draft_kv_layers + kv_buf_groups * hidden_kv_layers
                    )
                    sliced_dst_kv_ptrs.extend(
                        dst_kv_ptrs[layer_offset : layer_offset + draft_kv_layers]
                    )
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_kv_item_len: Optional[int] = None,
        dst_attn_tp_size: Optional[int] = None,
    ):
        if dst_device_kv_indices is not None:
            raise NotImplementedError(
                "Ascend PD transfer does not support HiSparse "
                "destination device KV indices"
            )
        self._validate_envelope_kv_layout(
            dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size
        )
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        if self.pp_size > 1:
            if self.is_mla_backend:
                # Prefer layer-id pairing so heterogeneous per-layer buffer
                # groups (e.g. partial-layer DSA indexer + scale) are matched
                # correctly even when their group length != hidden_kv_layers.
                src_layer_ids = self.kv_args.kv_layer_ids
                if src_layer_ids and dst_layer_ids:
                    pairs = build_transfer_entry_pairs(
                        src_layer_ids,
                        dst_layer_ids,
                        len(self.kv_args.kv_data_ptrs),
                        len(dst_kv_ptrs),
                    )
                    layers_params = [
                        (
                            self.kv_args.kv_data_ptrs[i],
                            dst_kv_ptrs[j],
                            self.kv_args.kv_item_lens[i],
                        )
                        for i, j in pairs
                    ]
                else:
                    src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                        self.get_mla_kv_ptrs_with_pp(
                            self.kv_args.kv_data_ptrs, dst_kv_ptrs
                        )
                    )
                    layers_params = [
                        (
                            src_kv_ptrs[layer_id],
                            sliced_dst_kv_ptrs[layer_id],
                            self.kv_args.kv_item_lens[layer_id],
                        )
                        for layer_id in range(layers_current_pp_stage)
                    ]
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def _is_generic_kvcache_state_type(self, st) -> bool:
        # DSV4 per-pool components also use the page-indexed send path.
        return (
            super()._is_generic_kvcache_state_type(st)
            or st in _DSV4_KVCACHE_STATE_TYPES
        )

    def init_layerwise(self) -> None:
        """Lazily create the dedicated NPU transfer stream.  Called once on
        the first layerwise forward; idempotent."""
        if getattr(self, "_layerwise_initialized", False):
            return

        self._transfer_stream = torch.npu.Stream()
        self._layerwise_initialized = True
        logger.info(
            "Layerwise PD KV transfer enabled (NPU stream overlap) "
            "on AscendKVManager (pp_size=%d, is_mla=%s).",
            self.pp_size,
            self.is_mla_backend,
        )

    def build_layer_transfer_blocks(
        self,
        layer_id: int,
        src_ptr: int,
        dst_kv_ptrs_for_layer: int,
        item_len: int,
        precomputed_layout: Optional[Tuple[List, List]] = None,
    ) -> List[Tuple[int, int, int]]:
        """Build the ``(src_addr, dst_addr, length)`` tuples for a single
        layer, mirroring ``set_transfer_blocks`` inside ``send_kvcache``.

        ``precomputed_layout`` is the result of a prior
        ``group_concurrent_contiguous`` call for the same page indices.
        When provided, the NumPy diff/split work is skipped — the layout
        is identical for every layer in one forward."""
        if precomputed_layout is not None:
            prefill_kv_blocks, dst_kv_blocks = precomputed_layout
        else:
            raise RuntimeError(
                "build_layer_transfer_blocks requires precomputed_layout "
                "(the sender caches group_concurrent_contiguous per-forward)"
            )
        transfer_blocks: List[Tuple[int, int, int]] = []
        for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
            src_addr = src_ptr + int(prefill_index[0]) * item_len
            dst_addr = dst_kv_ptrs_for_layer + int(decode_index[0]) * item_len
            length = item_len * len(prefill_index)
            transfer_blocks.append((src_addr, dst_addr, length))
        return transfer_blocks

    def submit_layerwise_transfer(
        self,
        session_id: str,
        transfer_blocks: List[Tuple[int, int, int]],
        compute_event: Optional[object],
    ) -> None:
        """Submit one layer's RDMA write on the transfer stream.

        ``compute_event`` is recorded on the compute stream right after the
        layer's attention op.  ``stream.wait_event`` inserts a *non-blocking*
        dependency: the transfer stream will not execute the RDMA until the
        compute stream has reached that event, but the calling (compute)
        thread is **not** blocked — it returns immediately and can continue
        submitting the next layer's compute kernels.
        """
        stream = self._transfer_stream
        if compute_event is not None:
            stream.wait_event(compute_event)
        with torch.npu.stream(stream):
            self._transfer_data(session_id, transfer_blocks)

    def wait_compute_on_transfer(self) -> None:
        """Make the compute stream wait for all queued transfer-stream work.

        Inserts a non-blocking dependency on the NPU: the compute stream will
        not execute subsequent kernels until the transfer stream has drained.
        Called before EP combine communication to avoid RDMA / HCCL network
        resource contention.
        """
        if hasattr(self, "_transfer_stream"):
            torch.npu.current_stream().wait_stream(self._transfer_stream)

    def finish_layerwise(self) -> None:
        """Record completion event on the transfer stream (non-blocking).

        Called right after the model forward loop ends.  Records an event on
        the transfer stream so the actual CPU-side wait can be deferred to
        :meth:`wait_layerwise_done`, which the scheduler calls at a single
        sync point — after PP batchSendRecv but before any ``KVPoll.Success``
        is set.  This lets PP communication overlap with the KV RDMA tail.
        """
        if hasattr(self, "_transfer_stream"):
            if not hasattr(self, "_layerwise_done_event"):
                self._layerwise_done_event = torch.npu.Event()
            self._layerwise_done_event.record(self._transfer_stream)

    def wait_layerwise_done(self) -> None:
        """Block the CPU until all transfer-stream RDMA writes have completed.

        Called from the scheduler at a single sync point before
        ``_pp_process_batch_result`` so that all PP ranks wait at the same
        pipeline stage, keeping the consensus consistent.
        """
        event = getattr(self, "_layerwise_done_event", None)
        if event is not None:
            event.synchronize()


class AscendKVSender(MooncakeKVSender):
    """Ascend KV sender with optional layerwise transfer (NPU stream overlap).

    When ``SGLANG_DISAGG_LAYERWISE`` is enabled, the model forward loop calls
    :meth:`start_layerwise_send` / :meth:`save_kv_layer` /
    :meth:`finalize_layerwise_send` instead of the single post-forward
    :meth:`send`.  Each ``save_kv_layer`` records a compute-stream event and
    submits the RDMA write on a dedicated transfer stream, so layer N's
    transfer overlaps with layer N+1's compute on the NPU hardware.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dummy CP ranks (CP1-7 with attn-cp-size > 1) have no real KV data
        # and must go through the standard send() path every chunk so that
        # curr_idx advances correctly and is_last_chunk=True triggers
        # KVPoll.Success on the final chunk.  Enabling layerwise for them
        # causes non-last chunks to skip send_kv_chunk (because the overlap
        # mode early-send guard checks is_layerwise_enabled), so curr_idx
        # never accumulates and Success is never signalled.
        self._layerwise_enabled = bool(envs.SGLANG_DISAGG_LAYERWISE.get()) and (
            not self.kv_mgr.is_dummy_cp_rank
        )
        self._layerwise_num_layers = 0
        # Page indices shared across all layers of one forward (KV layout is
        # identical across layers for a given page).  Populated by the
        # scheduler via set_layerwise_indices before the forward begins.
        self._layerwise_prefill_kv_indices: Optional[npt.NDArray[np.int32]] = None
        self._layerwise_dst_kv_indices: Optional[npt.NDArray[np.int32]] = None
        # Per-sender cache for the contiguous-block layout shared across all
        # layers of one forward.  Unlike the manager-level cache this is
        # isolated per sender so concurrent senders with different page
        # indices do not pollute each other.
        self._layer_layout_cache: Optional[Tuple[List, List]] = None
        # Layers whose KV couldn't be dispatched because bootstrap hadn't
        # completed yet when their save_kv_layer was called.  Flushed the
        # moment transfer_infos becomes available (in a later save_kv_layer
        # call or in finalize_layerwise_send).
        self._pending_layerwise_layers: List[int] = []
        # Number of layers actually dispatched via the layerwise path.
        self._layerwise_dispatched_count = 0
        # Tracks whether the transfer-stream done event has been waited on
        # for the current forward.  Reset in start_layerwise_send, consumed
        # by the scheduler before _pp_process_batch_result.
        self._layerwise_send_waited = True
        self._deferred_mode = self._layerwise_enabled and self.kv_mgr.pp_size > 1
        self._deferred_transfer_blocks: List[Tuple[str, List[Tuple[int, int, int]]]] = []

    @property
    def is_layerwise_enabled(self) -> bool:
        return self._layerwise_enabled

    @property
    def layerwise_kv_fully_dispatched(self) -> bool:
        """True only if every expected layer was dispatched via save_kv_layer.
        When False, the post-forward send path must fall back to a full
        KV send so no layer's cache is lost."""
        return (
            self._layerwise_enabled
            and self._layerwise_num_layers > 0
            and self._layerwise_dispatched_count >= self._layerwise_num_layers
        )

    def set_layerwise_indices(
        self,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
    ) -> None:
        """Cache the page indices for the upcoming layerwise forward.  Called
        by the scheduler right before model.forward so every save_kv_layer
        reuses the same indices."""
        self._layerwise_prefill_kv_indices = prefill_kv_indices
        self._layerwise_dst_kv_indices = dst_kv_indices

    def start_layerwise_send(self, num_layers: int) -> None:
        """Prepare the manager's transfer stream for the upcoming forward."""
        if not self._layerwise_enabled:
            return
        self._layerwise_num_layers = num_layers
        self._layer_layout_cache = None
        self._pending_layerwise_layers = []
        self._layerwise_dispatched_count = 0
        self._layerwise_send_waited = False
        self._deferred_transfer_blocks = []
        self.kv_mgr.init_layerwise()

    def save_kv_layer(self, layer_id: int) -> None:
        """Submit one layer's KV transfer on the NPU transfer stream.

        1. Record an event on the compute stream (marks KV write complete).
        2. Submit the RDMA write on the transfer stream with a ``wait_event``
           dependency — non-blocking for the CPU, the NPU handles overlap.

        If bootstrap hasn't completed yet (transfer_infos empty), the layer
        is buffered in ``_pending_layerwise_layers`` and flushed once
        bootstrap becomes available in a later call or in
        finalize_layerwise_send.
        """
        if not self._layerwise_enabled:
            return
        if self._layerwise_prefill_kv_indices is None:
            return
        mgr = self.kv_mgr
        transfer_infos = getattr(mgr, "transfer_infos", {})
        room = self.bootstrap_room
        # Snapshot the room's transfer infos in one GIL-atomic step; the
        # transfer_worker thread may pop this room concurrently once the
        # post-forward send path concludes.  Iterating over a list copy
        # keeps the loop safe even if the underlying dict is modified.
        room_infos = transfer_infos.get(room)
        if not room_infos:
            # Bootstrap not yet complete — buffer this layer for later
            # dispatch.  Its KV data is already written by the attention
            # op, so we can safely transfer it once the destination
            # pointers are available.
            self._pending_layerwise_layers.append(layer_id)
            return
        # Bootstrap is ready.  Record a compute-stream event once; it
        # covers both the current layer and any previously buffered layers
        # (their attention ops have already completed on the compute stream).
        # In deferred mode, no compute event is needed — transfers are
        # submitted after the entire forward completes.
        compute_event = (
            None if self._deferred_mode else self._record_compute_event()
        )
        # Flush layers that were buffered while bootstrap was pending.
        if self._pending_layerwise_layers:
            for pending_id in self._pending_layerwise_layers:
                self._dispatch_single_layer(
                    pending_id, mgr, room_infos, compute_event
                )
                self._layerwise_dispatched_count += 1
            self._pending_layerwise_layers = []
        # Dispatch the current layer.
        self._dispatch_single_layer(layer_id, mgr, room_infos, compute_event)
        self._layerwise_dispatched_count += 1

    def _dispatch_single_layer(
        self,
        layer_id: int,
        mgr: "AscendKVManager",
        room_infos: dict,
        compute_event: Optional[object],
    ) -> None:
        """Build and submit one layer's RDMA write on the transfer stream.

        Shared by save_kv_layer (live layers) and the pending-flush path
        (layers buffered while bootstrap was incomplete)."""
        for tinfo in list(room_infos.values()):
            if tinfo.is_dummy:
                continue
            decode_kv_args = mgr.decode_kv_args_table.get(tinfo.mooncake_session_id)
            if decode_kv_args is None:
                continue
            # Build the per-layer transfer blocks.  For pp_size == 1 the
            # layer id indexes directly into kv_data_ptrs / dst_kv_ptrs.
            # For pp_size > 1, kv_data_ptrs / kv_item_lens only contain the
            # current PP stage's layers, so offset by prefill_start_layer.
            if mgr.pp_size > 1:
                rel_layer = layer_id - mgr.kv_args.prefill_start_layer
                src_kv_ptrs, sliced_dst_kv_ptrs, _ = mgr.get_mla_kv_ptrs_with_pp(
                    mgr.kv_args.kv_data_ptrs, decode_kv_args.dst_kv_ptrs
                )
                src_ptr = src_kv_ptrs[rel_layer]
                dst_ptr = sliced_dst_kv_ptrs[rel_layer]
                item_len = mgr.kv_args.kv_item_lens[rel_layer]
            else:
                src_ptr = mgr.kv_args.kv_data_ptrs[layer_id]
                dst_ptr = decode_kv_args.dst_kv_ptrs[layer_id]
                item_len = mgr.kv_args.kv_item_lens[layer_id]
            if self._layer_layout_cache is None:
                self._layer_layout_cache = group_concurrent_contiguous(
                    self._layerwise_prefill_kv_indices,
                    self._layerwise_dst_kv_indices,
                )
            blocks = mgr.build_layer_transfer_blocks(
                layer_id,
                src_ptr,
                dst_ptr,
                item_len,
                precomputed_layout=self._layer_layout_cache,
            )
            if self._deferred_mode:
                self._deferred_transfer_blocks.append(
                    (tinfo.mooncake_session_id, blocks)
                )
            else:
                mgr.submit_layerwise_transfer(
                    tinfo.mooncake_session_id, blocks, compute_event
                )

    def wait_compute_on_transfer(self) -> None:
        """Make the compute stream wait for all queued transfer-stream work.

        Inserts a non-blocking dependency on the NPU: the compute stream will
        not execute subsequent kernels (e.g. EP combine all-reduce) until the
        transfer stream has drained, avoiding RDMA / HCCL network resource
        contention.  No-op if layerwise is disabled or no transfer was
        submitted.
        """
        if not self._layerwise_enabled:
            return
        if self._deferred_mode:
            return
        if self._layerwise_dispatched_count == 0 and not self._pending_layerwise_layers:
            return
        self.kv_mgr.wait_compute_on_transfer()

    def finalize_layerwise_send(self) -> None:
        """Prepare for post-forward KV transfer completion.  Aux/state
        components are small and stay on the standard post-forward path.

        If bootstrap completed by end of forward, any layers buffered while
        it was pending are flushed here.  If bootstrap is still not complete,
        the buffered layers remain undispatched and
        ``layerwise_kv_fully_dispatched`` returns False, causing the
        post-forward send path to fall back to a full KV send.

        In deferred mode (PP > 1), transfer blocks are only cached here;
        the actual RDMA submission is deferred to ``flush_deferred_transfers``
        which the scheduler calls after PP_send completes."""
        if not self._layerwise_enabled:
            return
        # Last chance: if bootstrap completed by end of forward, flush any
        # layers that were buffered while it was pending.
        if self._pending_layerwise_layers:
            mgr = self.kv_mgr
            transfer_infos = getattr(mgr, "transfer_infos", {})
            room_infos = transfer_infos.get(self.bootstrap_room)
            if room_infos:
                compute_event = (
                    None
                    if self._deferred_mode
                    else self._record_compute_event()
                )
                for pending_id in self._pending_layerwise_layers:
                    self._dispatch_single_layer(
                        pending_id, mgr, room_infos, compute_event
                    )
                    self._layerwise_dispatched_count += 1
                self._pending_layerwise_layers = []
        if not self._deferred_mode:
            self.kv_mgr.finish_layerwise()

    def _record_compute_event(self):
        """Record an NPU event on the current compute stream so the transfer
        stream can wait for the KV write to land before issuing RDMA."""
        if torch_npu is None:
            return None
        event = torch.npu.Event()
        event.record()
        return event

    def wait_layerwise_send_done(self) -> None:
        """Synchronize the transfer-stream event (deferred from
        finalize_layerwise_send).

        Called by the scheduler at a single sync point — after PP
        batchSendRecv, before ``_pp_process_batch_result`` — so all PP
        ranks wait at the same pipeline stage.  Idempotent within one
        forward (guarded by ``_layerwise_send_waited``).
        """
        if not self._layerwise_enabled or self._layerwise_send_waited:
            return
        self._layerwise_send_waited = True
        self.kv_mgr.wait_layerwise_done()


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
