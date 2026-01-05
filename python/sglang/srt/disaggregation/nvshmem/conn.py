from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch
import zmq
import json

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
)
import time
from sglang.srt.disaggregation.utils import DisaggregationMode, NVSHMEM_PWRITE_MODE
from sglang.srt.server_args import ServerArgs
from sglang.srt.disaggregation.nvshmem import nvshmem_utils
from sglang.srt.utils import (
    format_tcp_address,
    get_local_ip_auto,
    is_valid_ipv6_address,
)
import os

logger = logging.getLogger(__name__)

PD_DEBUG = os.getenv("SGLANG_PD_DEBUG", "0").lower() in ("1", "true")

GUARD = "NVSHMEMMsgGuard".encode("ascii")


class NVSHMEMKVArgs(KVArgs):
    """Placeholder subclass for potential NVSHMEM-specific extensions."""

    pass


@dataclass
class TransferInfo:
    room: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

class NVSHMEMKVManager(CommonKVManager):
    def __init__(
        self,
        args: NVSHMEMKVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        if not nvshmem_utils.ensure_initialized():
            raise RuntimeError("Failed to initialize NVSHMEM runtime.")
        self.peer_tensor_fn = nvshmem_utils.get_peer_tensor_fn()
        if self.peer_tensor_fn is None:
            raise RuntimeError("NVSHMEM torch interop is unavailable.")
        self.peer_rank = nvshmem_utils.get_peer_rank()
        if self.peer_rank is None:
            raise RuntimeError("NVSHMEM peer rank is not configured.")
        self.status_tensor = (
            self.kv_args.aux_nvshmem_buffers[-1]
            if getattr(self.kv_args, "aux_nvshmem_buffers", [])
            else None
        )
        if self.status_tensor is None:
            raise RuntimeError("NVSHMEM backend requires aux NVSHMEM buffers for status flags.")
        self.peer_status_tensor = (
            self.peer_tensor_fn(self.status_tensor, self.peer_rank)
            if self.peer_tensor_fn is not None and self.peer_rank is not None
            else None
        )
        # Cache peer views of KV buffers to avoid repeated get_peer_tensor calls.
        self.peer_kv_views = (
            [
                self.peer_tensor_fn(tensor, self.peer_rank)
                for tensor in self.kv_args.nvshmem_buffers
            ]
            if self.peer_tensor_fn is not None and self.peer_rank is not None
            else []
        )
        if PD_DEBUG:
            logger.info(
                "NVSHMEM manager init role=%s rank=%s peer_rank=%s status_ptr=0x%x",
                disaggregation_mode.value,
                self.kv_args.engine_rank,
                self.peer_rank,
                self.status_tensor.data_ptr() if self.status_tensor is not None else 0,
            )
        self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
        self.local_ip = get_local_ip_auto()
        self.server_socket = zmq.Context().socket(zmq.PULL)
        if is_valid_ipv6_address(self.local_ip):
            self.server_socket.setsockopt(zmq.IPV6, 1)
        self.server_socket.bind(format_tcp_address(self.local_ip, self.rank_port))
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.request_status: Dict[int, KVPoll] = {}
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, bool] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )
        # Cache per-room peer views to avoid stale pointers if buffers change.
        self.peer_views_per_room: Dict[int, List[Optional[torch.Tensor]]] = {}
        logger.info("Initialized NVSHMEM manager in %s mode.", disaggregation_mode.value)

    def _connect_to_bootstrap_server(self, bootstrap_info: dict):
        """
        Mirror CommonKVReceiver._connect_to_bootstrap_server so the bootstrap thread can
        open PUSH sockets back to registering prefills/decode peers without importing
        receiver internals.
        """
        return CommonKVReceiver._connect_to_bootstrap_server(bootstrap_info)

    def update_status(self, bootstrap_room: int, status: KVPoll):
        self.request_status[bootstrap_room] = status

    def check_status(self, bootstrap_room: int) -> KVPoll:
        return self.request_status.get(bootstrap_room, KVPoll.WaitingForInput)

    def register_buffer_to_engine(self):
        return

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
    ):
        if bootstrap_room not in self.transfer_infos:
            logger.warning(
                "NVSHMEM transfer requested for unknown room %s. Ignoring chunk.",
                bootstrap_room,
            )
            return None

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        for req in reqs_to_be_processed:
            if req.is_dummy():
                continue
            dst_kv_indices = req.dst_kv_indices[index_slice]
            if PD_DEBUG:
                logger.info(
                    "NVSHMEM transfer room=%s chunk=%s size=%s last=%s aux=%s status_ptr=0x%x peer_rank=%s",
                    bootstrap_room,
                    chunk_id,
                    len(kv_indices),
                    is_last,
                    aux_index,
                    self.status_tensor.data_ptr() if self.status_tensor is not None else 0,
                    self.peer_rank,
                )
            if NVSHMEM_PWRITE_MODE:
                if PD_DEBUG:
                    logger.info(
                        "NVSHMEM pwrite mode: skipping copy for room=%s chunk=%s size=%s",
                        bootstrap_room,
                        chunk_id,
                        len(kv_indices),
                    )
            else:
                peer_views = self._get_peer_views(bootstrap_room)
                self._copy_pages(kv_indices, dst_kv_indices, peer_views)
            if is_last and self.status_tensor is not None:
                target_aux = req.dst_aux_index
                if target_aux is None or target_aux < 0:
                    target_aux = aux_index
                self._update_peer_status(target_aux, chunk_id + 1)

        if is_last:
            del self.transfer_infos[bootstrap_room]
            self.peer_views_per_room.pop(bootstrap_room, None)
        return None

    def mark_transfer_ready(
        self,
        bootstrap_room: int,
        aux_index: Optional[int],
        chunk_id: int = 0,
    ) -> None:
        if self.status_tensor is None:
            return
        target_aux = None
        if bootstrap_room in self.transfer_infos:
            infos = self.transfer_infos[bootstrap_room]
            for info in infos.values():
                if getattr(info, "dst_aux_index", None) is not None:
                    target_aux = info.dst_aux_index
                    break
        
        if target_aux is None or target_aux < 0:
            target_aux = aux_index
        if self.peer_status_tensor is None:
            return
        self._update_peer_status(target_aux, chunk_id + 1)
        if bootstrap_room in self.transfer_infos:
            del self.transfer_infos[bootstrap_room]
        self.peer_views_per_room.pop(bootstrap_room, None)
        #nvshmem_utils.quiet_current_stream()

    def _update_peer_status(
        self,
        target_aux: Optional[int],
        value: int,
    ) -> None:
        if self.peer_status_tensor is None:
            return
        if target_aux is None or target_aux < 0:
            raise ValueError("Missing aux index for NVSHMEM status flag.")
        self.peer_status_tensor[int(target_aux)] = value
        if PD_DEBUG:
            logger.info(
                "NVSHMEM status flag aux=%s marked=%s peer_rank=%s",
                target_aux,
                value,
                self.peer_rank,
            )

    def _get_peer_views(self, room_id: int) -> List[Optional[torch.Tensor]]:
        if room_id in self.peer_views_per_room:
            return self.peer_views_per_room[room_id]
        views: List[Optional[torch.Tensor]] = []
        for tensor in self.kv_args.nvshmem_buffers:
            view = (
                self.peer_tensor_fn(tensor, self.peer_rank)
                if self.peer_tensor_fn is not None and self.peer_rank is not None
                else None
            )
            views.append(view)
        self.peer_views_per_room[room_id] = views
        return views

    def _copy_pages(
        self,
        prefill_kv_indices: npt.NDArray[np.int32],
        decode_kv_indices: npt.NDArray[np.int32],
        peer_views: List[Optional[torch.Tensor]],
    ):
        if len(prefill_kv_indices) == 0:
            return
        if len(prefill_kv_indices) != len(decode_kv_indices):
            raise ValueError(
                f"Prefill/decode kv indices length mismatch: "
                f"{len(prefill_kv_indices)} vs {len(decode_kv_indices)}"
            )
        for tensor, peer_view in zip(self.kv_args.nvshmem_buffers, peer_views):
            if peer_view is None:
                continue
            # Vectorized copy into peer view to reduce Python looping overhead.
            src_idx_tensor = torch.as_tensor(
                prefill_kv_indices, device=tensor.device, dtype=torch.long
            )
            dst_idx_tensor = torch.as_tensor(
                decode_kv_indices, device=peer_view.device, dtype=torch.long
            )
            peer_view[dst_idx_tensor] = tensor[src_idx_tensor]
        if PD_DEBUG:
            logger.info(
                "NVSHMEM copied %s KV pages -> peer indices %s",
                len(prefill_kv_indices),
                decode_kv_indices.tolist()[:8],
            )


    def update_transfer_status(self):
        return

    def check_transfer_done(self, room: int):
        return False

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            while True:
                try:
                    waiting_req_bytes = self.server_socket.recv_multipart()
                    if waiting_req_bytes[0] != GUARD:
                        continue
                    msg = waiting_req_bytes[1:]
                    room = msg[0].decode("ascii")
                    if room == "None":
                        # This is a registration request from a prefill instance
                        # msg format: [b"None", json_bytes]
                        # self._handle_registration(msg[1:])
                        bootstrap_info = json.loads(msg[1].decode("ascii"))
                        self._connect_to_bootstrap_server(bootstrap_info)
                    else:
                        # This is a bootstrap request from a decode instance
                        # msg format: [room, local_ip, rank_port, receiver_id, kv_indices, aux_index, required_dst_info_num]
                        room_id = int(room)
                        agent_name = msg[3].decode("ascii")
                        dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
                        dst_aux_index = int(msg[5].decode("ascii"))
                        required_dst_info_num = int(msg[6].decode("ascii"))                        
                        if room_id not in self.transfer_infos:
                            self.transfer_infos[room_id] = {}
                        self.transfer_infos[room_id][agent_name] = TransferInfo(
                            room=room_id,
                            agent_name=agent_name,
                            dst_kv_indices=dst_kv_indices,
                            dst_aux_index=dst_aux_index,
                            required_dst_info_num=required_dst_info_num,
                        )
                        
                        if len(self.transfer_infos[room_id]) == required_dst_info_num:
                            self.update_status(room_id, KVPoll.WaitingForInput)
                except Exception as e:
                    logger.error(f"Bootstrap thread exception: {e}", exc_info=True)

        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def _handle_registration(self, _msg: List[bytes]):
        return


class NVSHMEMKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: NVSHMEMKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.dest_tp_ranks = dest_tp_ranks
        self.pp_rank = pp_rank
        self.num_kv_indices: Optional[int] = None
        self.aux_index: Optional[int] = None
        self.curr_idx = 0
        self.chunk_id = 0
        self.has_sent = False
        self._cached_dst_indices: Optional[np.ndarray] = None
        
        #start_time = time.perf_counter()
        # Check if the status is already WaitingForInput (race condition where bootstrap arrived first)
        #current_status = self.kv_mgr.request_status.get(self.bootstrap_room, None)
        #if current_status == KVPoll.WaitingForInput:
        #    pass
        #else:
        #    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        #duration = (time.perf_counter() - start_time) * 1000
        #if duration > 0.1:
        #    logger.warning(f"Slow NVSHMEM KVSender checking or updating status: {duration:.2f} ms")

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index
        if NVSHMEM_PWRITE_MODE:
            self.get_dst_kv_indices()

    def send(self, kv_indices: npt.NDArray[np.int32], state_indices: Optional[List[int]] = None):
        if self.num_kv_indices is None:
            raise RuntimeError("NVSHMEMKVSender.init must be called before send().")
        chunk_len = len(kv_indices)
        if NVSHMEM_PWRITE_MODE:
            self._notify_pwrite_chunk(chunk_len)
            return
        index_slice = slice(self.curr_idx, self.curr_idx + chunk_len)
        self.curr_idx += chunk_len
        is_last = self.curr_idx == self.num_kv_indices
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
        )
        self.chunk_id += 1
        if is_last:
            self.has_sent = True

    def notify_pwrite_chunk(self, chunk_len: int) -> None:
        self._notify_pwrite_chunk(chunk_len)

    def _notify_pwrite_chunk(self, chunk_len: int) -> None:
        self.curr_idx += chunk_len
        total = self.num_kv_indices or 0
        #is_complete = total == 0 or self.curr_idx >= total
        is_complete = True
        self.kv_mgr.mark_transfer_ready(
            self.bootstrap_room,
            self.aux_index,
            chunk_id=self.chunk_id,
        )
        self.chunk_id += 1
        if is_complete:
            self.has_sent = True

    def poll(self) -> KVPoll:
        if not self.has_sent:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status == KVPoll.Bootstrapping:
                # logger.info(f"Sender poll room {self.bootstrap_room} is Bootstrapping")
                pass
            time.sleep(0.001)
            return status
        return KVPoll.Success  # type: ignore

    def failure_exception(self):
        raise RuntimeError("NVSHMEM sender failure is not supported yet.")

    def get_dst_kv_indices(self):
        if self._cached_dst_indices is None:
            infos = self.kv_mgr.transfer_infos.get(self.bootstrap_room, {})
            for info in infos.values():
                if getattr(info, "dst_kv_indices", None) is not None:
                    self._cached_dst_indices = info.dst_kv_indices.copy()
                    break
        return self._cached_dst_indices


class NVSHMEMKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NVSHMEMKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state = None
        self.aux_index: Optional[int] = None
        self.receiver_id = f"nvshmem-{mgr.kv_args.engine_rank}"
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None, state_indices: Optional[List[int]] = None):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.receiver_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index if aux_index is not None else -1).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

        self.started_transfer = True
        self.aux_index = aux_index

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore
        if self.aux_index is None:
            return KVPoll.WaitingForInput  # type: ignore
        status_tensor = (
            self.kv_mgr.kv_args.aux_nvshmem_buffers[-1]
            if getattr(self.kv_mgr.kv_args, "aux_nvshmem_buffers", [])
            else None
        )
        if status_tensor is None:
            return KVPoll.WaitingForInput  # type: ignore
        if status_tensor is None:
            return KVPoll.WaitingForInput  # type: ignore
        # flag_val = int(status_tensor[self.aux_index].item())
        if PD_DEBUG:
            logger.info(
                "NVSHMEM receiver poll room=%s aux=%s flag=%s status_ptr=0x%x peer_rank=%s",
                self.bootstrap_room,
                self.aux_index,
                flag_val,
                status_tensor.data_ptr() if hasattr(status_tensor, "data_ptr") else 0,
                getattr(self.kv_mgr, "peer_rank", None),
            )
        
        flag_val = int(status_tensor[self.aux_index].item())

        if flag_val > 0:
            status_tensor[self.aux_index] = 0
            self.conclude_state = KVPoll.Success
            if PD_DEBUG:
                logger.info(
                    "NVSHMEM receiver room=%s aux=%s transfer complete",
                    self.bootstrap_room,
                    self.aux_index,
                )
            return KVPoll.Success  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        json.dumps(bootstrap_info).encode("ascii"),
                    ]
                )

    def failure_exception(self):
        raise RuntimeError("NVSHMEM receiver failure is not supported yet.")


class NVSHMEMKVBootstrapServer(CommonKVBootstrapServer):
    pass
