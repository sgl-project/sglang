from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.async_kv_utils import (
    AsyncInfo,
    StreamAsyncSubmitter,
    TransferKVChunkSet,
    cached_group_concurrent_contiguous,
    env_int,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, kv_to_page_indices
from sglang.srt.mem_cache.memory_pool import get_mamba_pool_state_tensor_counts

logger = logging.getLogger(__name__)


class MooncakeKVAsyncMixin:
    """Mixin that implements layerwise async KV transfer for MooncakeKVManager.

    This mixin is intentionally self-contained and only relies on a small set
    of attributes/methods provided by the concrete manager:

    Required attrs:
    - `disaggregation_mode`, `is_mla_backend`, `attn_tp_size`
    - `kv_args` (with kv/state ptrs and state_type)
    - `transfer_infos`, `request_status`, `decode_kv_args_table`

    Required methods:
    - `_transfer_data(session_id, blocks)`
    """

    # -------------------------
    # Lifecycle / state
    # -------------------------

    def _async_kv_init_state(self) -> None:
        self._async_kv_enabled: bool = False
        self._async_submitter: Optional[StreamAsyncSubmitter] = None
        self._notify_queue: Optional[deque[AsyncInfo]] = None
        self._waiting_rooms: Optional[deque[Optional[TransferKVChunkSet]]] = None
        self._current_kv_chunk_infos: Optional[TransferKVChunkSet] = None
        self._req_begin_count: Dict[int, deque[int]] = {}
        self._req_bids: Dict[int, deque[int]] = {}
        self._req_tensor_seen: Dict[int, set[int]] = {}
        self._room_to_kv_chunk_info: Dict[int, tuple[TransferKVChunkSet, int]] = {}
        self._lock: Optional[threading.Lock] = None
        self._bids_cond: Optional[threading.Condition] = None
        self._queue_lock: Optional[threading.Lock] = None
        self._kv_tensor_ntensors: int = 0
        self._state_tensor_ntensors: int = 0
        self._tensor_ntensors_total: int = 0
        self._kv_cache_nlayers: int = 0
        self._async_kv_missing_wait_ms: int = 0
        self._layer_ready_events: Dict[Tuple[int, int], Any] = {}
        self._mamba_num_layers_debug: int = 0
        self._mamba_state_tensors_per_layer_debug: int = 0

    @property
    def async_kv_enabled(self) -> bool:
        return bool(self._async_kv_enabled)

    def _async_kv_enable(self) -> None:
        """Enable layerwise async KV transfers for PREFILL."""

        # Only supported in PREFILL.
        if getattr(self, "disaggregation_mode", None) is None:
            return
        from sglang.srt.disaggregation.utils import DisaggregationMode

        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return

        self._async_kv_enabled = True
        self._notify_queue = deque()
        self._waiting_rooms = deque()
        self._current_kv_chunk_infos = None
        self._req_begin_count = {}
        self._req_bids = {}
        self._req_tensor_seen = {}
        self._room_to_kv_chunk_info = {}
        # _lock protects per-room bookkeeping (_req_*, _room_to_kv_chunk_info, _layer_ready_events).
        self._lock = threading.Lock()
        self._bids_cond = threading.Condition(self._lock)
        self._queue_lock = threading.Lock()
        self._kv_tensor_ntensors = len(self.kv_args.kv_data_ptrs)
        self._state_tensor_ntensors = len(self.kv_args.state_data_ptrs)
        self._tensor_ntensors_total = self._kv_tensor_ntensors + self._state_tensor_ntensors
        self._kv_cache_nlayers = (
            self._kv_tensor_ntensors
            if self.is_mla_backend
            else (self._kv_tensor_ntensors // 2)
        )
        self._async_kv_missing_wait_ms = env_int(
            "SGLANG_ASYNC_KV_MISSING_WAIT_MS", "20"
        )
        self._layer_ready_events = {}
        self._mamba_num_layers_debug = 0
        self._mamba_state_tensors_per_layer_debug = 0

        self._async_submitter = StreamAsyncSubmitter(self._async_put_kvcache_func)

    # -------------------------
    # Scheduler hook
    # -------------------------

    def maybe_prepare_async_kv(self, sch: Any, batch: Any) -> Optional[Callable[[int], None]]:
        """Optionally prepare async KV transfer for a scheduler batch.

        Returns a `layer_ready_callback` if async is enabled and the batch is eligible.
        """

        if not self._async_kv_enabled:
            return None

        eligible_reqs = [
            req
            for req in batch.reqs
            if getattr(req, "bootstrap_host", None) != FAKE_BOOTSTRAP_HOST
        ]
        if not eligible_reqs:
            return None

        # Current async path only supports non-chunked, full-send (start_send_idx=0) requests.
        if not all(
            getattr(req, "start_send_idx", None) == 0
            and getattr(req, "is_chunked", 0) <= 0
            for req in eligible_reqs
        ):
            return None

        self._async_prepare_batch(sch, batch)
        setattr(batch, "async_kv_prepared", True)
        return self._async_mark_layer_ready

    # -------------------------
    # Internal helpers
    # -------------------------

    def _async_put_kvcache_func(self) -> None:
        try:
            if self._notify_queue is None or self._queue_lock is None:
                return
            with self._queue_lock:
                if not self._notify_queue:
                    return
                info = self._notify_queue.pop()
            self._async_put_kv_cache_internal(info)
        except Exception:
            logger.exception("Unhandled exception in async KV submitter.")

    def _async_try_sync_ready_event(self, *, room_id: int, tensor_id: int, reason: str) -> None:
        if not self._async_kv_enabled or self._lock is None:
            return
        event_key = (int(room_id), int(tensor_id))
        with self._lock:
            event = self._layer_ready_events.pop(event_key, None)
        if event is None:
            return

        try:
            import torch

            if torch.cuda.is_available():
                event.synchronize()
        except Exception:
            logger.warning(
                "Failed to synchronize CUDA event (%s): room=%s tensor=%s",
                reason,
                room_id,
                tensor_id,
                exc_info=True,
            )

    def _async_try_record_ready_event_for_rooms(
        self, *, rooms: Tuple[int, ...], tensor_id: int, reason: str
    ) -> None:
        if not self._async_kv_enabled or self._lock is None:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            event = torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False)
            event.record()
            with self._lock:
                for rid in rooms:
                    self._layer_ready_events[(int(rid), int(tensor_id))] = event
        except Exception:
            logger.debug(
                "Failed to record CUDA event (%s): tensor=%s",
                reason,
                tensor_id,
                exc_info=True,
            )

    def _async_maybe_start_next_kv_chunk(self) -> None:
        if not self._async_kv_enabled or self._queue_lock is None or self._lock is None:
            return
        assert self._async_submitter is not None
        begin_count = self._async_submitter.get_step_count()
        with self._queue_lock:
            assert self._waiting_rooms is not None
            current = self._waiting_rooms.pop() if self._waiting_rooms else None
            self._current_kv_chunk_infos = current

        if not current:
            logger.warning("async kv layer0: no waiting rooms")
            return

        # Keep lock ordering consistent: _queue_lock -> _lock.
        with self._lock:
            for idx, rid in enumerate(current.rooms):
                if rid not in self._req_begin_count:
                    self._req_begin_count[rid] = deque()
                self._req_begin_count[rid].appendleft(begin_count)
                self._room_to_kv_chunk_info[rid] = (current, idx)

    def _async_filter_current_kv_chunk_infos(self) -> None:
        if not self._async_kv_enabled or self._queue_lock is None or self._lock is None:
            return
        with self._queue_lock:
            current = self._current_kv_chunk_infos
            if not current or not current.rooms:
                return
            rooms = current.rooms

        keep_indices = [
            idx
            for idx, rid in enumerate(rooms)
            if rid in self.transfer_infos and self.request_status.get(rid) != KVPoll.Success
        ]
        if not keep_indices or len(keep_indices) == len(rooms):
            return

        filtered = TransferKVChunkSet(
            rooms=tuple(rooms[i] for i in keep_indices),
            prefill_kv_indices=tuple(current.prefill_kv_indices[i] for i in keep_indices),
            index_slices=tuple(current.index_slices[i] for i in keep_indices),
            prefill_state_indices=tuple(current.prefill_state_indices[i] for i in keep_indices),
        )
        with self._queue_lock:
            self._current_kv_chunk_infos = filtered

        with self._lock:
            for rid in rooms:
                if rid not in filtered.rooms:
                    self._room_to_kv_chunk_info.pop(rid, None)
            for idx, rid in enumerate(filtered.rooms):
                self._room_to_kv_chunk_info[rid] = (filtered, idx)

    def _async_get_info_with_risk(self, room: int) -> dict:
        if room not in self.transfer_infos:
            status = self.request_status.get(room)
            if status != KVPoll.Success:
                logger.warning(
                    "async kv skip: room=%s not in transfer_infos status=%s",
                    room,
                    status,
                )
            return {}
        return self.transfer_infos[room]

    def _async_submit_layer(
        self,
        session_id: str,
        src_ptr: int,
        dst_ptr: int,
        prefill_kv_blocks: npt.NDArray[np.int64],
        dst_kv_blocks: npt.NDArray[np.int64],
        item_len: int,
    ) -> int:
        prefill_kv_blocks_tmp, dst_kv_blocks_tmp = cached_group_concurrent_contiguous(
            prefill_kv_blocks, dst_kv_blocks
        )
        if not prefill_kv_blocks_tmp:
            return 0
        transfer_blocks = []
        for prefill_index, decode_index in zip(prefill_kv_blocks_tmp, dst_kv_blocks_tmp):
            src_addr = src_ptr + int(prefill_index[0]) * item_len
            dst_addr = dst_ptr + int(decode_index[0]) * item_len
            transfer_blocks.append((src_addr, dst_addr, item_len * len(prefill_index)))
        return int(self._transfer_data(session_id, transfer_blocks))

    def _async_put_kv_cache_internal(self, async_info: AsyncInfo) -> None:
        kv_chunk_info = async_info.kv_chunk_info
        if not kv_chunk_info.rooms:
            return
        infos = [self._async_get_info_with_risk(room) for room in kv_chunk_info.rooms]
        for layer_id in async_info.layer_ids:
            for room_id, transfer_info_dict, kv_indice, index_slice, prefill_state_idx in zip(
                kv_chunk_info.rooms,
                infos,
                kv_chunk_info.prefill_kv_indices,
                kv_chunk_info.index_slices,
                kv_chunk_info.prefill_state_indices,
            ):
                if not transfer_info_dict:
                    continue
                for transfer_info in transfer_info_dict.values():
                    if transfer_info.is_dummy:
                        continue
                    session_id = transfer_info.mooncake_session_id
                    registration = self.decode_kv_args_table.get(session_id)
                    if registration is None:
                        logger.warning(
                            "async kv skip: missing registration room=%s session=%s layer=%s",
                            room_id,
                            session_id,
                            layer_id,
                        )
                        continue

                    is_state_tensor = layer_id >= self._kv_tensor_ntensors
                    if not is_state_tensor:
                        self._async_try_sync_ready_event(
                            room_id=int(room_id), tensor_id=int(layer_id), reason="kv"
                        )
                        dst = transfer_info.dst_kv_indices[index_slice]
                        if len(dst) < len(kv_indice):
                            kv_indice = kv_indice[: len(dst)]
                        item_len = int(registration.dst_kv_item_len)
                        src_ptr = int(self.kv_args.kv_data_ptrs[int(layer_id)])
                        dst_ptr = int(registration.dst_kv_ptrs[int(layer_id)])
                        status = self._async_submit_layer(
                            session_id, src_ptr, dst_ptr, kv_indice, dst, item_len
                        )
                    else:
                        # State tensors (mamba only).
                        state_tensor_id = int(layer_id) - self._kv_tensor_ntensors
                        if self.kv_args.state_type != "mamba" or not transfer_info.dst_state_indices:
                            status = 0
                        else:
                            self._async_try_sync_ready_event(
                                room_id=int(room_id),
                                tensor_id=int(layer_id),
                                reason="mamba_state",
                            )
                            dst_state_idx = int(transfer_info.dst_state_indices[0])
                            src_state_ptrs = self.kv_args.state_data_ptrs
                            src_state_item_lens = self.kv_args.state_item_lens
                            dst_state_ptrs = getattr(registration, "dst_state_data_ptrs", [])
                            if (
                                state_tensor_id >= len(src_state_ptrs)
                                or state_tensor_id >= len(dst_state_ptrs)
                                or state_tensor_id >= len(src_state_item_lens)
                            ):
                                status = 0
                            else:
                                item_len = int(src_state_item_lens[state_tensor_id])
                                src_addr = int(src_state_ptrs[state_tensor_id]) + item_len * int(
                                    prefill_state_idx
                                )
                                dst_addr = int(dst_state_ptrs[state_tensor_id]) + item_len * int(
                                    dst_state_idx
                                )
                                status = int(
                                    self._transfer_data(session_id, [(src_addr, dst_addr, item_len)])
                                )

                    assert self._bids_cond is not None
                    with self._bids_cond:
                        self._req_tensor_seen.setdefault(room_id, set()).add(int(layer_id))
                        self._req_bids.setdefault(room_id, deque()).appendleft(int(status))
                        self._bids_cond.notify_all()

    def _async_mark_layer_ready(self, layer_id: int) -> None:
        if not self._async_kv_enabled:
            return
        tensor_id = int(layer_id)
        if tensor_id == -1:
            self._async_maybe_start_next_kv_chunk()
            return

        assert self._queue_lock is not None
        with self._queue_lock:
            current = self._current_kv_chunk_infos

        if current is None:
            self._async_maybe_start_next_kv_chunk()
            with self._queue_lock:
                current = self._current_kv_chunk_infos

        if tensor_id < 0 or tensor_id >= self._tensor_ntensors_total:
            logger.warning(
                "async kv layer ready skipped: tensor=%s total=%s",
                tensor_id,
                self._tensor_ntensors_total,
            )
            return

        if current:
            self._async_filter_current_kv_chunk_infos()
            with self._queue_lock:
                current = self._current_kv_chunk_infos

        if not current or not current.rooms:
            return

        if tensor_id < self._kv_tensor_ntensors:
            self._async_try_record_ready_event_for_rooms(
                rooms=current.rooms, tensor_id=int(tensor_id), reason="kv"
            )
        elif self.kv_args.state_type == "mamba":
            self._async_try_record_ready_event_for_rooms(
                rooms=current.rooms, tensor_id=int(tensor_id), reason="mamba_state"
            )

        with self._queue_lock:
            assert self._notify_queue is not None
            self._notify_queue.appendleft(
                AsyncInfo(layer_ids=(int(tensor_id),), kv_chunk_info=current)
            )
        assert self._async_submitter is not None
        self._async_submitter.step_async()

    def _async_wait_for_bids(self, rid: int, *, timeout_s: Optional[float] = None) -> bool:
        if self._bids_cond is None:
            return False
        deadline = None if timeout_s is None else (time.time() + float(timeout_s))
        with self._bids_cond:
            while True:
                q = self._req_bids.get(rid)
                if q is not None and len(q) >= self._tensor_ntensors_total:
                    return True
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return False
                    self._bids_cond.wait(timeout=remaining)
                else:
                    self._bids_cond.wait()

    def _async_resend_missing_state_tensors(self, room: int, missing_state_tensor_ids: list[int]) -> None:
        if not missing_state_tensor_ids:
            return
        info = self._room_to_kv_chunk_info.get(room)
        if info is None:
            return
        kv_chunk_info, idx = info
        if idx >= len(kv_chunk_info.prefill_state_indices):
            return
        prefill_state_idx = kv_chunk_info.prefill_state_indices[idx]
        if prefill_state_idx is None or prefill_state_idx < 0:
            return
        transfer_info_dict = self.transfer_infos.get(room)
        if not transfer_info_dict:
            return
        src_state_ptrs = self.kv_args.state_data_ptrs
        src_state_item_lens = self.kv_args.state_item_lens
        for transfer_info in transfer_info_dict.values():
            if transfer_info.is_dummy:
                continue
            session_id = transfer_info.mooncake_session_id
            registration = self.decode_kv_args_table.get(session_id)
            if registration is None or not transfer_info.dst_state_indices:
                continue
            dst_state_ptrs = getattr(registration, "dst_state_data_ptrs", [])
            dst_state_idx = int(transfer_info.dst_state_indices[0])
            transfer_blocks = []
            for state_tensor_id in missing_state_tensor_ids:
                if (
                    state_tensor_id >= len(src_state_ptrs)
                    or state_tensor_id >= len(dst_state_ptrs)
                    or state_tensor_id >= len(src_state_item_lens)
                ):
                    continue
                item_len = int(src_state_item_lens[state_tensor_id])
                src_addr = int(src_state_ptrs[state_tensor_id]) + item_len * int(prefill_state_idx)
                dst_addr = int(dst_state_ptrs[state_tensor_id]) + item_len * int(dst_state_idx)
                transfer_blocks.append((src_addr, dst_addr, item_len))
            if transfer_blocks:
                self._transfer_data(session_id, transfer_blocks)

    def _async_pop_req_bids(self, rid: int, is_remove: bool):
        assert self._bids_cond is not None
        with self._bids_cond:
            q = self._req_bids.pop(rid) if is_remove else self._req_bids[rid]
            return [q.pop() for _ in range(self._tensor_ntensors_total)]

    def _async_flush_all_layers(self, rid: int) -> None:
        if self._lock is None:
            return
        with self._lock:
            if rid not in self._req_begin_count:
                return

        assert self._async_submitter is not None
        while True:
            with self._lock:
                if not self._req_begin_count.get(rid):
                    break
                begin_count = self._req_begin_count[rid].pop()

            self._async_submitter.wait_sent_finish(begin_count + self._tensor_ntensors_total)
            self._async_wait_for_bids(rid)

            with self._lock:
                current_last = len(self._req_begin_count.get(rid, ())) == 0

            statuses = self._async_pop_req_bids(rid, current_last)
            if current_last:
                with self._lock:
                    seen = set(self._req_tensor_seen.get(rid, set()))
                missing_kv = [i for i in range(self._kv_tensor_ntensors) if i not in seen]
                missing_state = [
                    i
                    for i in range(self._kv_tensor_ntensors, self._tensor_ntensors_total)
                    if i not in seen
                ]
                if missing_state:
                    self._async_resend_missing_state_tensors(
                        rid, [i - self._kv_tensor_ntensors for i in missing_state]
                    )
                if any(s != 0 for s in statuses) or missing_kv or missing_state:
                    logger.warning(
                        "async kv flush: room=%s nonzero=%s missing=%s",
                        rid,
                        sum(1 for s in statuses if s != 0),
                        len(missing_kv) + len(missing_state),
                    )
                with self._lock:
                    self._req_tensor_seen.pop(rid, None)
                    self._room_to_kv_chunk_info.pop(rid, None)
                    if self._layer_ready_events:
                        keys = [k for k in self._layer_ready_events.keys() if k[0] == rid]
                        for k in keys:
                            self._layer_ready_events.pop(k, None)
                    self._req_begin_count.pop(rid, None)
                break

        with self._lock:
            self._req_begin_count.pop(rid, None)

    def _async_prepare_batch(self, sch: Any, batch: Any) -> None:
        if not self._async_kv_enabled:
            return

        rooms = []
        prefill_kv_indices = []
        index_slices = []
        prefill_state_indices = []

        if self.kv_args.state_type == "mamba" and self._mamba_num_layers_debug == 0:
            mamba_pool = getattr(getattr(sch, "req_to_token_pool", None), "mamba_pool", None)
            (
                self._mamba_num_layers_debug,
                self._mamba_state_tensors_per_layer_debug,
            ) = get_mamba_pool_state_tensor_counts(mamba_pool)

        for req in batch.reqs:
            if getattr(req, "is_chunked", 0) > 0:
                continue
            if getattr(req, "bootstrap_host", None) == FAKE_BOOTSTRAP_HOST:
                continue
            room = int(req.bootstrap_room)
            if not self._async_is_eligible_room(room):
                continue
            page_size = sch.token_to_kv_pool_allocator.page_size
            start_idx = req.start_send_idx
            end_idx = min(len(req.fill_ids), len(req.origin_input_ids))
            if end_idx <= start_idx:
                continue
            kv_indices = (
                sch.req_to_token_pool.req_to_token[req.req_pool_idx, start_idx:end_idx]
                .cpu()
                .numpy()
            )
            page_indices = kv_to_page_indices(kv_indices, page_size)
            if len(page_indices) == 0:
                continue
            index_slice = slice(
                req.disagg_kv_sender.curr_idx,
                req.disagg_kv_sender.curr_idx + len(page_indices),
            )
            rooms.append(room)
            prefill_kv_indices.append(page_indices)
            index_slices.append(index_slice)
            state_idx = -1
            if self.kv_args.state_type == "mamba":
                try:
                    state_idx = int(
                        sch.req_to_token_pool.req_index_to_mamba_index_mapping[req.req_pool_idx].item()
                    )
                except Exception:
                    state_idx = -1
            prefill_state_indices.append(state_idx)

        kv_chunk_info_set = (
            TransferKVChunkSet(
                rooms=tuple(rooms),
                prefill_kv_indices=tuple(prefill_kv_indices),
                index_slices=tuple(index_slices),
                prefill_state_indices=tuple(prefill_state_indices),
            )
            if rooms
            else None
        )

        assert self._queue_lock is not None
        with self._queue_lock:
            assert self._waiting_rooms is not None
            self._waiting_rooms.appendleft(kv_chunk_info_set)

    def _async_is_eligible_room(self, room: int) -> bool:
        transfer_info_dict = self.transfer_infos.get(room)
        if transfer_info_dict is None or not transfer_info_dict:
            return False
        for transfer_info in transfer_info_dict.values():
            if transfer_info.is_dummy:
                continue
            registration = self.decode_kv_args_table.get(transfer_info.mooncake_session_id)
            if registration is None:
                return False
            if registration.dst_attn_tp_size != self.attn_tp_size:
                return False
        return not all(t.is_dummy for t in transfer_info_dict.values())

    def _async_use_for_room(self, room: int) -> bool:
        if not self._async_kv_enabled or self._lock is None:
            return False
        with self._lock:
            return room in self._req_begin_count

