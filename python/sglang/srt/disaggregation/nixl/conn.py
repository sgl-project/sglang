from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import pickle
import queue
import socket
import struct
import threading
import uuid
from collections import defaultdict
from functools import cache
from typing import Dict, List, Optional, Set, Tuple, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import BaseKVSender, KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import DisaggregationMode, RemotePrefillReq
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import broadcast_pyobj, get_local_ip_by_remote

logger = logging.getLogger(__name__)

try:
    import os

    import etcd3
    import nats

    IS_REMOTE_PREFILL_SUPPORT = True
except ImportError:
    logger.info("etcd3 or nats is not installed, remote prefill will not be available.")
    IS_REMOTE_PREFILL_SUPPORT = False

GUARD = "NixlMsgGuard".encode("ascii")


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    gpu_id: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            gpu_id=int(msg[7].decode("ascii")),
        )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunk IDs that have been received.
    received_kvs: Set[int] = dataclasses.field(default_factory=set)
    # Number of kv chunks to expect, will know this after last chunk is received.
    num_kvs_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False

    def is_done(self):
        if self.num_kvs_expected is None:
            return False
        return self.num_kvs_expected == len(self.received_kvs) and self.received_aux


class NixlKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        is_remote_prefill: Optional[bool] = False,
    ):
        super().__init__(
            args,
            disaggregation_mode,
            server_args,
            is_mla_backend,
            is_remote_prefill and IS_REMOTE_PREFILL_SUPPORT,
        )
        try:
            from nixl._api import nixl_agent
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e
        self.agent = nixl_agent(str(uuid.uuid4()))
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()

        if self.is_remote_prefill:
            import base64
            import hashlib
            import json

            etcd_endpoint = os.environ.get("ETCD_ENDPOINT", "127.0.0.1:2379")
            self.model_name_hash = hashlib.sha256(
                server_args.served_model_name.encode("utf-8")
            ).hexdigest()[:8]
            self.etcd_client = etcd3.client(
                host=etcd_endpoint.split(":")[0], port=int(etcd_endpoint.split(":")[1])
            )
            # register to etcd
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.request_status = {}
                self.transfer_infos: Dict[int, TransferInfo] = {}
                self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
                self.peer_names: Dict[str, str] = {}
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                    TransferStatus
                )
                # broadcast engine_id to all ranks only in decode
                engine_id = None
                if self.engine_rank == 0:
                    engine_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:8]
                    broadcast_pyobj([engine_id], self.engine_rank, None, 0, False)
                    logger.debug(
                        f"Rank {self.engine_rank} broadcasted engine_id: {engine_id}"
                    )
                else:
                    engine_id = broadcast_pyobj([], self.engine_rank, None, 0, False)[0]
                    logger.debug(
                        f"Rank {self.engine_rank} received engine_id: {engine_id}"
                    )
                self.engine_id = engine_id

                agent_metadata = base64.b64encode(
                    self.agent.get_agent_metadata()
                ).decode("ascii")
                if self.engine_rank == 0:
                    self.etcd_client.put(
                        f"/decode/{self.model_name_hash}/{self.engine_id}",
                        json.dumps(
                            {
                                "tp_size": self.tp_size,
                                "dp_size": self.dp_size,
                            }
                        ),
                    )
                self.etcd_client.put(
                    f"/decode/{self.model_name_hash}/{self.engine_id}/{self.engine_rank}",
                    json.dumps(
                        {
                            "agent_metadata": agent_metadata,
                            "agent_name": self.agent.name,
                            "kv_data_ptrs": self.kv_args.kv_data_ptrs,
                            "aux_data_ptrs": self.kv_args.aux_data_ptrs,
                            "gpu_id": self.kv_args.gpu_id,
                            "tp_size": self.tp_size,
                            "dp_size": self.dp_size,
                        }
                    ),
                )
                logger.debug(
                    f"Register key: /decode/{self.model_name_hash}/{self.engine_id}/{self.engine_rank} to ETCD"
                )
            else:
                raise ValueError(
                    f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
                )
            return

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.request_status: Dict[int, KVPoll] = {}
            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: The prefill engine could recv bootstrapping first
            self.request_status[bootstrap_room] = max(
                self.request_status[bootstrap_room], status
            )

    def register_buffer_to_engine(self):
        kv_addrs = []
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            kv_addrs.append((kv_data_ptr, kv_data_len, self.kv_args.gpu_id, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM", is_sorted=False)
        logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM", is_sorted=False)
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        self.agent.add_remote_agent(decode_kv_args.agent_metadata)

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # Make descs
        num_layers = len(self.kv_args.kv_data_ptrs)
        src_addrs = []
        dst_addrs = []
        for layer_id in range(num_layers):
            src_ptr = self.kv_args.kv_data_ptrs[layer_id]
            dst_ptr = dst_kv_ptrs[layer_id]
            item_len = self.kv_args.kv_item_lens[layer_id]

            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                src_addrs.append((src_addr, length, self.kv_args.gpu_id))
                dst_addrs.append((dst_addr, length, dst_gpu_id))

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_kv_indices)}, after group: {len(src_addrs)}"
        )
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM", is_sorted=False)
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM", is_sorted=False)
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        # Make descs
        aux_item_len = self.kv_args.aux_item_lens[0]
        prefill_aux_addr = (
            self.kv_args.aux_data_ptrs[0] + prefill_aux_index * aux_item_len
        )
        decode_aux_addr = dst_aux_ptrs[0] + dst_aux_index * aux_item_len
        src_addrs = [(prefill_aux_addr, aux_item_len, 0)]
        dst_addrs = [(decode_aux_addr, aux_item_len, 0)]
        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM", is_sorted=False)
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM", is_sorted=False)
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        handles = []
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                continue

            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            assert len(chunked_dst_kv_indice) == len(kv_indices)
            assert req.agent_name in self.decode_kv_args_table

            notif = "_".join([str(req.room), "kv", str(chunk_id), str(int(is_last))])
            kv_xfer_handle = self.send_kvcache(
                req.agent_name,
                kv_indices,
                self.decode_kv_args_table[req.agent_name].dst_kv_ptrs,
                chunked_dst_kv_indice,
                self.decode_kv_args_table[req.agent_name].gpu_id,
                notif,
            )
            handles.append(kv_xfer_handle)
            # Only the last chunk we need to send the aux data.
            if is_last:
                assert aux_index is not None
                aux_xfer_handle = self.send_aux(
                    req.agent_name,
                    aux_index,
                    self.decode_kv_args_table[req.agent_name].dst_aux_ptrs,
                    req.dst_aux_index,
                    str(req.room) + "_aux",
                )
                handles.append(aux_xfer_handle)
        if is_last:
            del self.transfer_infos[bootstrap_room]
        return handles

    def update_transfer_status(self):
        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            # We could also check that self.bootstrap_info['agent_name'] matches
            # the message sender. But the bootstrap room alone should be
            # sufficient to map the status.
            for msg in messages:
                components = msg.decode("ascii").split("_")
                room = int(components[0])
                if components[1] == "kv":
                    chunk_id = int(components[2])
                    is_last = bool(int(components[3]))
                    self.transfer_statuses[room].received_kvs.add(chunk_id)
                    if is_last:
                        self.transfer_statuses[room].num_kvs_expected = chunk_id + 1
                elif components[1] == "aux":
                    self.transfer_statuses[room].received_aux = True

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _start_bootstrap_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # Register new peer and save KV base pointers.
                    self._add_remote_peer(
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(f"Register KVArgs from {agent_name} successfully")
                    continue
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(
                    waiting_req_bytes
                )
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()


class NixlKVSender(BaseKVSender):

    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.xfer_handles = []
        self.has_sent = False
        self.chunk_id = 0
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        # inner state
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        new_xfer_handles = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
        )
        self.xfer_handles.extend(new_xfer_handles)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True
            del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        states = [self.kv_mgr.agent.check_xfer_state(x) for x in self.xfer_handles]
        if all([x == "DONE" for x in states]):
            return KVPoll.Success  # type: ignore
        if any([x == "ERR" for x in states]):
            raise Exception("KVSender transfer encountered an error.")
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.is_remote_prefill = mgr.is_remote_prefill
        self.conclude_state = None
        if self.is_remote_prefill:
            # if remote prefill, don't get boostrap info from prefill server

            self.kv_mgr = mgr
            self.bootstrap_room = bootstrap_room
            self.send_to_queue_loop = asyncio.new_event_loop()
            self._start_send_to_queue_thread(self.send_to_queue_loop)
            return
        super().__init__(mgr, bootstrap_addr, bootstrap_room, data_parallel_rank)

    def _start_send_to_queue_thread(self, event_loop):
        """Start a thread to send requests to the queue."""
        per_dp_tp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        if self.kv_mgr.engine_rank % per_dp_tp_rank != 0:
            return

        def start_async_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        threading.Thread(target=start_async_loop, args=(event_loop,)).start()

    def _send_to_queue(
        self,
        req: Req,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
    ):
        # only send the request to the queue if the engine rank per dp rank is 0
        per_dp_tp_rank = self.kv_mgr.tp_size // self.kv_mgr.dp_size
        if self.kv_mgr.engine_rank % per_dp_tp_rank != 0:
            logger.debug(f"rank {self.kv_mgr.engine_rank} start to transfer")
            self.started_transfer = True
            return

        queue_name = f"{self.kv_mgr.model_name_hash}"
        kv_indices = base64.b64encode(kv_indices.tobytes()).decode("ascii")

        remote_prefill_req = RemotePrefillReq(
            rid=req.rid,
            origin_input_text="",
            origin_input_ids=req.origin_input_ids,
            sampling_params=req.sampling_params,
            kv_indices=kv_indices,
            rank_ip=self.kv_mgr.rank_ip,
            rank_port=self.kv_mgr.rank_port,
            engine_rank=self.kv_mgr.engine_rank,
            aux_index=aux_index,
            bootstrap_room=self.bootstrap_room,
            engine_id=self.kv_mgr.engine_id,
        )

        remote_prefill_req_data = pickle.dumps(remote_prefill_req)

        async def send_to_nats():
            from os import environ

            import nats

            logger.debug(
                f"Send request and kv indices to queue: {queue_name} with len: {len(remote_prefill_req_data)} and boostrap room: {req.bootstrap_room}"
            )
            nats_endpoint = environ.get("NATS_ENDPOINT", "nats://127.0.0.1:4222")
            nats_client = await nats.connect(nats_endpoint)
            js = nats_client.jetstream()
            await js.add_stream(
                name=queue_name,
                subjects=[queue_name],
                max_age=60 * 60 * 24,  # 1 day
                max_bytes=1024 * 1024 * 1024,  # 1 GB
                max_msgs=1000000,  # 1 million messages
            )

            ark = await js.publish(queue_name, remote_prefill_req_data)

            self.started_transfer = True
            logger.debug(f"Published request to NATS with ark: {ark}")

        asyncio.run_coroutine_threadsafe(send_to_nats(), self.send_to_queue_loop)
        logger.debug(f"Send request to queue succeed!")

    def init(
        self,
        req: Req,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
    ):
        if self.is_remote_prefill:
            self._send_to_queue(req, kv_indices, aux_index)
            return

        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to {self.prefill_server_url} with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

        self.started_transfer = True

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore

        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return KVPoll.Success  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                    ]
                )

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
