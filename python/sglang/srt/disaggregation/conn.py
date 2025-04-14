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
    decode_agent_name: str
    room_number: int
    kv_descs_serialized: bytes
    aux_descs_serialized: bytes

    def __init__(
        self,
        decode_agent_name: str,
        room_number: int,
        kv_descs_serialized: bytes,
        aux_descs_serialized: bytes,
    ):
        # type check.
        assert isinstance(decode_agent_name, str)
        assert isinstance(room_number, int)
        assert isinstance(kv_descs_serialized, bytes)
        assert isinstance(aux_descs_serialized, bytes)
        
        self.decode_agent_name = decode_agent_name
        self.room_number = room_number
        self.kv_descs_serialized = kv_descs_serialized
        self.aux_descs_serialized = aux_descs_serialized

    def serialize(self) -> bytes:
        # layout:
        # decode_agent_name_len, 4 bytes
        # decode_agent_name, N bytes
        # room_number, 8 bytes
        # kv_descs_serialized_len, 4 bytes
        # kv_descs_serialized, M bytes
        # aux_desc_serialized_len, 4 bytes
        # aux_desc_serialized, K bytes

        return b"".join([
            len(self.decode_agent_name).to_bytes(4, "little"),
            self.decode_agent_name.encode("utf-8"),
            self.room_number.to_bytes(8, "little"),
            len(self.kv_descs_serialized).to_bytes(4, "little"),
            self.kv_descs_serialized,
            len(self.aux_descs_serialized).to_bytes(4, "little"),
            self.aux_descs_serialized
        ])

    @staticmethod
    def deserialize(data: bytes) -> DecodeMemDescMsg:
        def read_int(data: bytes, offset: int, size: int) -> tuple[int, int]:
            return int.from_bytes(data[offset:offset+size], "little"), offset+size
        def read_str_with_4B_len(data: bytes, offset: int) -> tuple[str, int]:
            length, offset = read_int(data, offset, 4)
            return data[offset:offset+length].decode("utf-8"), offset + length

        offset = 0
        decode_agent_name, offset = read_str_with_4B_len(data, offset)
        room_number, offset = read_int(data, offset, 8)
        kv_descs_serialized_len, offset = read_int(data, offset, 4)
        kv_descs_serialized = data[offset:offset+kv_descs_serialized_len]
        offset += kv_descs_serialized_len
        aux_descs_serialized_len, offset = read_int(data, offset, 4)
        aux_descs_serialized = data[offset:offset+aux_descs_serialized_len]
        return DecodeMemDescMsg(
            decode_agent_name=decode_agent_name,
            room_number=room_number,
            kv_descs_serialized=kv_descs_serialized,
            aux_descs_serialized=aux_descs_serialized
        )
    
    def send_to_memdesc_collector(self, addr: str):
        logger.info(f'[NIXL PD disagg] memdesc collector send memdesc with room {self.room_number} to {addr}. len(kv_descs): {len(self.kv_descs_serialized)}, len(aux_descs): {len(self.aux_descs_serialized)}')
        # addr is host:port
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.connect(f"tcp://{addr}")
        sock.send(self.serialize())

        sock.close()
        ctx.term()

    
class MemDescCollector:
    def __init__(self, addr: str): 
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.bind(f"tcp://{addr}")
        self.buffer = {} # room number -> DecodeMemDescMsg
        pass

    def recv_msgs(self):
        while True:
            try:
                msg= self.socket.recv(flags=zmq.NOBLOCK)
                msg = DecodeMemDescMsg.deserialize(msg)
                logger.info(f'[NIXL PD disagg] memdesc collector recv {msg.room_number}, len(kv_descs): {len(msg.kv_descs_serialized)}, len(aux_descs): {len(msg.aux_descs_serialized)}')
                assert self.buffer.get(msg.room_number) is None, f"Duplicate message for {msg.room_number}"
                self.buffer[msg.room_number] = msg
            except zmq.ZMQError:
                break
    
    def try_fetch(self, room_number: int) -> Optional[DecodeMemDescMsg]:
        if self.buffer.get(room_number) is None:
            return None
        logger.debug(f'[NIXL PD disagg] memdesc collector pop out {room_number}')
        return self.buffer[room_number]

# Connecting Stage:
# We may have many P tp workers and D tp workers, 
# every of them will have an unique addr.
# each pair of those P workers and D workers which need to communicate 
# will have a KVPeer for each other
#
# Request serving Stage:
# Once D worker recv a req, it will:
# * send metadata, such as room number, memory desc, etc, to the corresponding P memdesc collector
# * poll and wait until kvcache is ready.

'''
Used for world setup, not req serving.
'''
class D2PBootstrapMsg:
    agent_name: str
    agent_metadata: bytes
    
    def __init__(self, agent_name: str, agent_metadata: bytes):
        assert isinstance(agent_name, str)
        assert isinstance(agent_metadata, bytes)
        self.agent_name = agent_name
        self.agent_metadata = agent_metadata

    def serialize(self) -> bytes:
        return b"".join([
            len(self.agent_name.encode("utf-8")).to_bytes(4, "little"),
            self.agent_name.encode("utf-8"),
            len(self.agent_metadata).to_bytes(4, "little"),
            self.agent_metadata
        ])
    
    @staticmethod
    def deserialize(data: bytes) -> D2PBootstrapMsg:
        def read_int_4B(data: bytes, offset: int) -> tuple[int, int]:
            return int.from_bytes(data[offset:offset+4], "little"), offset+4
        def read_str(data: bytes, offset: int) -> tuple[str, int]:
            length, offset = read_int_4B(data, offset)
            return data[offset:offset+length].decode("utf-8"), offset + length

        offset = 0
        agent_name, offset = read_str(data, offset)
        agent_length, offset = read_int_4B(data, offset)
        agent_metadata = data[offset:offset+agent_length]
        return D2PBootstrapMsg(agent_name, agent_metadata)

class KVManager:
    def __init__(self, args: KVArgs, mode: str, *, 
                 # for prefill
                 bootstrap_port: int = 0,  
                 decode_instance_number: int = 0, 

                 # for decode
                 prefill_bootstrap_addrs: list[str] = []
                ):
        if mode not in ["prefill", "decode"]:
            raise Exception("Mode must be prefill or decode")
        
        if mode == 'prefill':
            assert bootstrap_port != 0
            assert decode_instance_number != 0
            assert prefill_bootstrap_addrs == []

        if mode == 'decode':
            assert bootstrap_port == 0
            assert len(prefill_bootstrap_addrs) > 0

        self.args = args
        if nixl_agent is None:
            raise Exception("NIXL is not available")
        
        self.agent=nixl_agent(str(uuid.uuid4()))

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
        

        # PD initial handshakes.
        if mode == "prefill":
            self.decode_agent_names: list[str] = []
            ctx = zmq.Context()
            socket = ctx.socket(zmq.ROUTER)
            socket.bind(f"tcp://*:{bootstrap_port}")

            for i in range(decode_instance_number):
                logger.info(f'[NIXL PD disagg] waiting for decode instance... Current: {i}/{decode_instance_number}')
                identity, msg = socket.recv_multipart()
                msg = D2PBootstrapMsg.deserialize(msg)
                logger.info(f'[NIXL PD disagg] decode instance connected. remote agent name: {msg.agent_name}')
                self.decode_agent_names.append(msg.agent_name)
                self.agent.add_remote_agent(msg.agent_metadata)
                socket.send_multipart([identity, self.agent.name.encode("utf-8")])

            logger.info(f'[NIXL PD disagg] bootstrapped all decode instance.')

            socket.close()
            ctx.term()
        else:
            # bootstrap_addr -> prefill agent_name
            # the boostrap addr will be also used as memdesc collector addr,
            # and we need remote_agent_name for later check_remote_xfer
            self.bootstrap_addr2_agent_names = {} 

            ctx = zmq.Context()
            for prefill_addr in prefill_bootstrap_addrs:
                logger.info(f'[NIXL PD disagg] connecting to prefill addr: {prefill_addr}')
                socket = ctx.socket(zmq.DEALER)
                socket.connect(f"tcp://{prefill_addr}")

                msg = D2PBootstrapMsg(
                    agent_name=self.agent.name,
                    agent_metadata=self.agent.get_agent_metadata()
                )
                socket.send(msg.serialize())
                remote_agent_name = socket.recv().decode("utf-8")
                self.bootstrap_addr2_agent_names[prefill_addr] = remote_agent_name
                socket.close()

            ctx.term()
                
            

        if mode == "prefill":
            logger.info(f'[NIXL PD disagg] bootstrap server started at port {bootstrap_port}')
            self.memdesc_collector = MemDescCollector(f"*:{bootstrap_port}")

class KVPoll():
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
    def __init__(self, mgr: KVManager, bootstrap_room: int):
        """
        bootstrap_addr: prefill: host address, decode: corresponding prefill address
        bootstrap_room: unique id per request
        """

        assert isinstance(bootstrap_room, int)
        
        self.has_sent = False
        self.mgr = mgr
        self.bootstrap_room = bootstrap_room


        self.aux_index = None
        self.decode_memdesc_msg: Optional[DecodeMemDescMsg] = None

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index

    def send(self, kv_indices: npt.NDArray[np.int32]):
        assert self.decode_memdesc_msg, "Have not received remote mem desc yet when try sending kv"

        # To adapt the strange design of NIXL xfer lookup mechanism. Check https://github.com/ai-dynamo/nixl/pull/138
        self.kv_xfer_name, self.aux_xfer_name = init_xfer_names(self.mgr.agent.name, self.decode_memdesc_msg.decode_agent_name, self.bootstrap_room)

        # Get descs
        kv_addrs = []
        for data_ptr, item_len in zip(self.mgr.args.kv_data_ptrs, self.mgr.args.kv_item_lens):
            for i in kv_indices:
                # kv_addrs.append((tmp.data_ptr(), item_len, self.mgr.args.engine_rank))
                kv_addrs.append((data_ptr + int(i) * item_len , item_len, self.mgr.args.engine_rank))
        kv_descs = self.mgr.agent.get_xfer_descs(kv_addrs, "VRAM", is_sorted=True)
        assert self.aux_index is not None
        aux_addrs = [(self.mgr.args.aux_data_ptrs[0] + self.aux_index * self.mgr.args.aux_item_lens[0], self.mgr.args.aux_item_lens[0], 0)]
        aux_descs = self.mgr.agent.get_xfer_descs(aux_addrs, "DRAM", is_sorted=True)
        logger.info(f"[NIXL PD disagg] KVSender: sending kv. self.bootstrap_room {self.bootstrap_room}") 

        # Send KV
        self.xfer_handle = self.mgr.agent.initialize_xfer(
            "WRITE", kv_descs, 
            self.mgr.agent.deserialize_descs(self.decode_memdesc_msg.kv_descs_serialized),
            self.decode_memdesc_msg.decode_agent_name, 
            self.kv_xfer_name # type: ignore
        )
        if not self.xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.mgr.agent.transfer(self.xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        # Send aux
        self.xfer_handle_aux = self.mgr.agent.initialize_xfer(
            "WRITE", aux_descs, 
            self.mgr.agent.deserialize_descs(self.decode_memdesc_msg.aux_descs_serialized),
            self.decode_memdesc_msg.decode_agent_name,
            self.aux_xfer_name # type: ignore
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

        if self.decode_memdesc_msg is None:
            # logger.debug(f'[NIXL PD disagg] try fetch {self.mgr.mgr} {self.bootstrap_room}')
            self.decode_memdesc_msg = \
                self.mgr.memdesc_collector.try_fetch( self.bootstrap_room)

        if self.decode_memdesc_msg is None:
            return KVPoll.Bootstrapping # type: ignore

        if self.has_sent is False:
            return KVPoll.WaitingForInput # type: ignore
        
        state = self.mgr.agent.check_xfer_state(self.xfer_handle)
        state2 = self.mgr.agent.check_xfer_state(self.xfer_handle_aux)

        if state == "ERR" or state2 == "ERR":
            raise Exception("KVSender transfer encountered an error.")
        elif state == "DONE" and state2 == "DONE":
            return KVPoll.Success # type: ignore
        else:
            # logger.info(f"[NIXL PD disagg] KVSender {self.bootstrap_room} poll result: {state}, {state2}")
            return KVPoll.Transferring # type: ignore

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
    

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: int):
        kv_addrs = []
        for data_ptr, item_len in zip(self.mgr.args.kv_data_ptrs, self.mgr.args.kv_item_lens):
            for i in kv_indices:
                kv_addrs.append((data_ptr + int(i) * item_len , item_len, self.mgr.args.engine_rank))
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
            decode_agent_name=self.mgr.agent.name,
            room_number=self.bootstrap_room if self.bootstrap_room else 0,
            kv_descs_serialized=self.mgr.agent.get_serialized_descs(kv_descs),
            aux_descs_serialized=self.mgr.agent.get_serialized_descs(aux_descs)
        )
        logger.debug(f'[NIXL PD disagg] KVReceiver: sending to Prefill memdesc collector ({self.memdesc_collector_addr}): {self.bootstrap_room}')
        msg.send_to_memdesc_collector(self.memdesc_collector_addr)

        self.prefill_agent_name = self.mgr.bootstrap_addr2_agent_names[self.memdesc_collector_addr]
        self.kv_xfer_name, self.aux_xfer_name = init_xfer_names(self.prefill_agent_name, self.mgr.agent.name, self.bootstrap_room)

        self.has_init = True

    def poll(self) -> KVPoll:
        if self.has_init is False:
            return KVPoll.WaitingForInput # type: ignore

        # to adapt the strange design of NIXL xfer look up mechanism.
        if not self.kv_transfer_done:
            self.kv_transfer_done = self.mgr.agent.check_remote_xfer_done(self.prefill_agent_name, self.kv_xfer_name )
        if not self.aux_transfer_done:
            self.aux_transfer_done = self.mgr.agent.check_remote_xfer_done(self.prefill_agent_name, self.aux_xfer_name )
        if self.kv_transfer_done and self.aux_transfer_done:
            return KVPoll.Success # type: ignore
        
        # logging.info(f"[NIXL PD disagg] KVReceiver {self.bootstrap_room} poll result: {self.kv_transfer_done}, {self.aux_transfer_done}")
        return KVPoll.WaitingForInput # type: ignore

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int): ...

    def poll(self) -> KVPoll: ...
