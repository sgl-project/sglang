from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Any

import numpy as np
import numpy.typing as npt
import uuid
import zmq
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from nixl._api import nixl_agent
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    nixl_agent = None

class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str


class DecodeMemDescMsg:
    room_number: int
    kv_descs_serialized: bytes
    aux_descs_serialized: bytes

    def __init__(
        self,
        room_number: int,
        kv_descs_serialized: bytes,
        aux_descs_serialized: bytes,
    ):
        # type check.
        assert isinstance(room_number, int)
        assert isinstance(kv_descs_serialized, bytes)
        assert isinstance(aux_descs_serialized, bytes)
        
        self.room_number = room_number
        self.kv_descs_serialized = kv_descs_serialized
        self.aux_descs_serialized = aux_descs_serialized

    def serialize(self) -> bytes:
        # layout:
        # room_number, 8 bytes
        # kv_descs_serialized_len, 4 bytes
        # kv_descs_serialized, M bytes
        # aux_desc_serialized_len, 4 bytes
        # aux_desc_serialized, K bytes

        return b"".join([
            self.room_number.to_bytes(8, "little"),
            len(self.kv_descs_serialized).to_bytes(4, "little"),
            self.kv_descs_serialized,
            len(self.aux_descs_serialized).to_bytes(4, "little"),
            self.aux_descs_serialized,
        ])

    @staticmethod
    def deserialize(data: bytes) -> DecodeMemDescMsg:
        def read_int_4B(data: bytes, offset: int) -> tuple[int, int]:
            return int.from_bytes(data[offset:offset+4], "little"), offset+4
        def read_int_8B(data: bytes, offset: int) -> tuple[int, int]:
            return int.from_bytes(data[offset:offset+8], "little"), offset+8
        
        offset = 0
        room_number, offset = read_int_8B(data, offset)
        kv_descs_serialized_len, offset = read_int_4B(data, offset)
        kv_descs_serialized = data[offset:offset+kv_descs_serialized_len]
        offset += kv_descs_serialized_len
        aux_descs_serialized_len, offset = read_int_4B(data, offset)
        aux_descs_serialized = data[offset:]
        return DecodeMemDescMsg(room_number, kv_descs_serialized, aux_descs_serialized)
        
    
    def send_to_memdesc_collector(self, addr: str):
        # addr is host:port
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.connect(f"tcp://{addr}")
        sock.send(self.serialize())

        sock.close()
        ctx.term()

    
class MemDescCollector:
    def __init__(self, port: int, deserializer): 
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.bind(f"tcp://*:{port}")
        self.buffer = {} # room number -> (remote_kv_descs, remote_aux_descs)
        self.deserializer = deserializer
        pass

    def recv_msgs(self):
        while True:
            try:
                msg= self.socket.recv(flags=zmq.NOBLOCK)
                msg = DecodeMemDescMsg.deserialize(msg)
                logger.debug(f'[NIXL PD disagg] memdesc collector recv {msg.room_number}, len(kv_descs): {len(msg.kv_descs_serialized)}, len(aux_descs): {len(msg.aux_descs_serialized)}')
                assert self.buffer.get(msg.room_number) is None, f"Duplicate message for {msg.room_number}"
                self.buffer[msg.room_number] = (
                    self.deserializer(msg.kv_descs_serialized), 
                    self.deserializer(msg.aux_descs_serialized)
                )
            except zmq.ZMQError:
                break
    
    def try_fetch(self, room_number: int) -> tuple[Any, Any]:
        if self.buffer.get(room_number) is None:
            return (None, None)
        logger.debug(f'[NIXL PD disagg] memdesc collector pop out {room_number}')
        return self.buffer[room_number]

class KVManager:
    def __init__(self, args: KVArgs, mode: str):
        if mode not in ["prefill", "decode"]:
            raise Exception("Mode must be prefill or decode")
        self.args = args
        if nixl_agent is None:
            raise Exception("NIXL is not available")
        self.agent = nixl_agent(str(uuid.uuid4()))

        # Register buffers.
        kv_addrs = []
        for data_ptr, data_len in zip(self.args.kv_data_ptrs, self.args.kv_data_lens):
            kv_addrs.append((data_ptr, data_len, self.args.engine_rank, ""))
            # kv_addrs.append((tmp.data_ptr(), data_len, self.args.engine_rank, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM", is_sorted=True)
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = [(self.args.aux_data_ptrs[0], self.args.aux_data_lens[0], 0, "")]
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM", is_sorted=True)
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

        # Create socket connection between decode/prefill
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PAIR)

        # TODO(wyt) For 1P1D we just let D connect P now, and exchange metadata immediately.
        # Possible future design:
        # we may have many P and Ds, 
        # every P will have an unique addr and port to serve a boostrap server
        # Once D recv a req, it will:
        # 1. connect to the corresponding P bootstrap server
        # 2. exchange metadata, such as room number, memory desc, etc.
        # 3. poll and wait until kvcache is ready.

        if mode == "prefill":
            self.socket.bind("tcp://127.0.0.1:8998")
            self.memdesc_collector = MemDescCollector(9000, self.agent.deserialize_descs)
        elif mode == "decode":
            self.socket.connect("tcp://127.0.0.1:8998")

        # Metadata exchange
        if mode == "prefill":
            remote_metadata = self.socket.recv()
            self.socket.send_string(self.agent.name)
            self.peer_name = self.agent.add_remote_agent(remote_metadata)
            if type(self.peer_name) is bytes:
                self.peer_name = self.peer_name.decode("utf-8")

            if not self.peer_name:
                raise Exception("KVSender failed to add KVReceiver's remote agent metadata")
        elif mode == "decode":
            metadata = self.agent.get_agent_metadata()
            if not metadata:
                raise Exception("KVSender failed to get metadata")
            self.socket.send(metadata)
            self.peer_name = self.socket.recv_string()
            if type(self.peer_name) is bytes:
                self.peer_name = self.peer_name.decode("utf-8")
        
        self.peer_name_bytes = self.peer_name.encode("utf-8")

class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4

    @staticmethod
    def str(x) -> str:
        if x == KVPoll.Failed:
            return "Failed"
        elif x == KVPoll.Bootstrapping:
            return "Bootstrapping"
        elif x == KVPoll.WaitingForInput:
            return "WaitingForInput"
        elif x == KVPoll.Transferring:
            return "Transferring"
        elif x == KVPoll.Success:
            return "Success"
        else:
            raise Exception("Unknown KVPoll state")

import torch
tmp=torch.zeros(64 * 1024 * 1024, dtype=torch.int32, device="cpu").contiguous()


def init_xfer_names(prefill_agent_name: str, decode_agent_name: str, bootstrap_room: int) -> tuple[bytes, bytes]:
    assert isinstance(prefill_agent_name, str)
    assert isinstance(decode_agent_name, str)
    assert isinstance(bootstrap_room, int)
    return (
        (f"{prefill_agent_name}_{decode_agent_name}_[{bootstrap_room}]_kv").encode("utf-8"),
        (f"{prefill_agent_name}_{decode_agent_name}_[{bootstrap_room}]_aux").encode("utf-8")
    )

class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        """
        bootstrap_addr: prefill: host address, decode: corresponding prefill address
        bootstrap_room: unique id per request
        """

        assert isinstance(bootstrap_room, int)
        
        self.has_sent = False
        self.mgr = mgr
        self.bootstrap_room = bootstrap_room

        self.remote_kv_descs = None
        self.remote_aux_descs = None

        # To adapt the strange design of NIXL xfer lookup mechanism. Check https://github.com/ai-dynamo/nixl/pull/138
        self.kv_xfer_name, self.aux_xfer_name = init_xfer_names(mgr.agent.name, mgr.peer_name, bootstrap_room)

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index

    def send(self, kv_indices: npt.NDArray[np.int32]):
        assert self.remote_kv_descs is not None and self.remote_aux_descs is not None, "Have not received remote mem desc yet when try sending kv"

        # Get descs
        kv_addrs = []
        for data_ptr, item_len in zip(self.mgr.args.kv_data_ptrs, self.mgr.args.kv_item_lens):
            for i in kv_indices:
                # kv_addrs.append((tmp.data_ptr(), item_len, self.mgr.args.engine_rank))
                kv_addrs.append((data_ptr + i * item_len , item_len, self.mgr.args.engine_rank))
        kv_descs = self.mgr.agent.get_xfer_descs(kv_addrs, "VRAM", is_sorted=True)
        aux_addrs = [(self.mgr.args.aux_data_ptrs[0] + self.aux_index * self.mgr.args.aux_item_lens[0], self.mgr.args.aux_item_lens[0], 0)]
        aux_descs = self.mgr.agent.get_xfer_descs(aux_addrs, "DRAM", is_sorted=True)
        logger.info(f"[NIXL PD disagg] KVSender: sending kv. self.bootstrap_room {self.bootstrap_room}") 

        # Send KV
        self.xfer_handle = self.mgr.agent.initialize_xfer(
            "WRITE", kv_descs, self.remote_kv_descs, self.mgr.peer_name, self.kv_xfer_name
        )
        if not self.xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.mgr.agent.transfer(self.xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        # Send aux
        self.xfer_handle_aux = self.mgr.agent.initialize_xfer(
            "WRITE", aux_descs, self.remote_aux_descs, self.mgr.peer_name, self.aux_xfer_name
        )
        if not self.xfer_handle_aux:
            raise Exception("KVSender failed to create transfer")
        state = self.mgr.agent.transfer(self.xfer_handle_aux)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        self.has_sent = True

    def poll(self) -> KVPoll:
        # TODO(wyt) We force recv remote mem desc first, and then forward. Actually they can take place in parallel.
        # Consider multiple TP worker synchronization here!

        if self.remote_kv_descs is None:
            assert self.remote_aux_descs is None
            # logger.debug(f'[NIXL PD disagg] try fetch {self.mgr.peer_name} {self.bootstrap_room}')
            self.remote_kv_descs, self.remote_aux_descs = \
                self.mgr.memdesc_collector.try_fetch( self.bootstrap_room)
        else:
            assert self.remote_aux_descs is not None

        if self.remote_kv_descs is None or self.remote_aux_descs is None:
            return KVPoll.Bootstrapping

        if self.has_sent is False:
            return KVPoll.WaitingForInput
        state = self.mgr.agent.check_xfer_state(self.xfer_handle)
        state2 = self.mgr.agent.check_xfer_state(self.xfer_handle_aux)
        if state == "ERR" or state2 == "ERR":
            raise Exception("KVSender transfer encountered an error.")
        elif state == "DONE" and state2 == "DONE":
            return KVPoll.Success
        else:
            # logger.info(f"[NIXL PD disagg] KVSender {self.bootstrap_room} poll result: {state}, {state2}")
            return KVPoll.Transferring

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:
    def __init__(
        self, mgr: KVManager, memdesc_collector_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.has_init = False
        self.mgr = mgr
        self.memdesc_collector_addr = memdesc_collector_addr

        if bootstrap_room is None:
            logger.warning("KVReceiver bootstrap room is None, using 0")
        self.bootstrap_room = bootstrap_room if bootstrap_room else 0

        self.kv_transfer_done = False
        self.aux_transfer_done = False
    

        # to adapt the strange design of NIXL xfer look up mechanism.
        self.kv_xfer_name, self.aux_xfer_name = init_xfer_names(mgr.peer_name, mgr.agent.name, self.bootstrap_room)

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        kv_addrs = []
        for data_ptr, item_len in zip(self.mgr.args.kv_data_ptrs, self.mgr.args.kv_item_lens):
            for i in kv_indices:
                kv_addrs.append((data_ptr + i * item_len , item_len, self.mgr.args.engine_rank))
                # kv_addrs.append((tmp.data_ptr() , item_len, self.mgr.args.engine_rank))
        kv_descs = self.mgr.agent.get_xfer_descs(kv_addrs, "VRAM", is_sorted=True)
        aux_addrs = [(self.mgr.args.aux_data_ptrs[0] + aux_index * self.mgr.args.aux_item_lens[0], self.mgr.args.aux_item_lens[0], 0)]
        aux_descs = self.mgr.agent.get_xfer_descs(aux_addrs, "DRAM", is_sorted=True)
        logger.debug(f'[NIXL PD disagg] KVReceiver: kv_descs: {kv_descs}, room: {self.bootstrap_room}')

        # Once D recv a req, it will:
        # 1. do preallocate 
        # 2. connect the bootstrap server and send metadata, such as room number, memory desc, etc.
        # 3. poll and wait until kvcache is ready.
        
        msg = DecodeMemDescMsg(
            room_number=self.bootstrap_room if self.bootstrap_room else 0,
            kv_descs_serialized=self.mgr.agent.get_serialized_descs(kv_descs),
            aux_descs_serialized=self.mgr.agent.get_serialized_descs(aux_descs)
        )
        logger.debug(f'[NIXL PD disagg] KVReceiver: sending to Prefill memdesc collector ({self.memdesc_collector_addr}): {msg.room_number}')
        msg.send_to_memdesc_collector(self.memdesc_collector_addr)

        self.has_init = True

    def poll(self) -> KVPoll:
        if self.has_init is False:
            return KVPoll.WaitingForInput
        if not self.kv_transfer_done:
            self.kv_transfer_done = self.mgr.agent.check_remote_xfer_done(self.mgr.peer_name, self.kv_xfer_name )
        if not self.aux_transfer_done:
            self.aux_transfer_done = self.mgr.agent.check_remote_xfer_done(self.mgr.peer_name, self.aux_xfer_name )
        if self.kv_transfer_done and self.aux_transfer_done:
            return KVPoll.Success
        
        # logging.info(f"[NIXL PD disagg] KVReceiver {self.bootstrap_room} poll result: {self.kv_transfer_done}, {self.aux_transfer_done}")
        return KVPoll.WaitingForInput

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int): ...

    def poll(self) -> KVPoll: ...
