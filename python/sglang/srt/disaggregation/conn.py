from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt
import uuid
import zmq

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
            kv_addrs.append((tmp.data_ptr(), data_len, self.args.engine_rank, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "DRAM", is_sorted=True)
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = [(self.args.aux_data_ptrs[0], self.args.aux_data_lens[0], 0, "")]
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM", is_sorted=True)
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

        # Create socket connection between decode/prefill
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PAIR)
        if mode == "prefill":
            self.socket.connect("tcp://127.0.0.1:8998")
        elif mode == "decode":
            self.socket.bind("tcp://127.0.0.1:8998")

        # Metadata exchange
        if mode == "prefill":
            remote_metadata = self.socket.recv()
            self.socket.send_string(self.agent.name)
            self.peer_name = self.agent.add_remote_agent(remote_metadata)
            if not self.peer_name:
                raise Exception("KVSender failed to add KVReceiver's remote agent metadata")
        elif mode == "decode":
            metadata = self.agent.get_agent_metadata()
            if not metadata:
                raise Exception("KVSender failed to get metadata")
            self.socket.send(metadata)
            self.peer_name = self.socket.recv_string()

class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


import torch
tmp=torch.zeros(64 * 1024 * 1024, dtype=torch.int32, device="cpu").contiguous()


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        """
        bootstrap_addr: prefill: host address, decode: corresponding prefill address
        bootstrap_room: unique id per request
        """
        self.has_sent = False
        self.mgr = mgr
        self.bootstrap_room = bootstrap_room

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index

    def send(self, kv_indices: npt.NDArray[np.int32]):
        # Get descs
        logging.info(f"[wytdebug] recving descs for bootstrap_room {self.bootstrap_room}")
        remote_kv_descs = self.mgr.agent.deserialize_descs(self.mgr.socket.recv())
        logging.info(f"[wytdebug] recving aux descs for bootstrap_room {self.bootstrap_room}")
        remote_aux_descs = self.mgr.agent.deserialize_descs(self.mgr.socket.recv())
        kv_addrs = []
        for data_ptr, item_len in zip(self.mgr.args.kv_data_ptrs, self.mgr.args.kv_item_lens):
            for i in kv_indices:
                kv_addrs.append((tmp.data_ptr(), item_len, self.mgr.args.engine_rank))
                # kv_addrs.append((data_ptr + i * item_len , item_len, self.mgr.args.engine_rank))
        kv_descs = self.mgr.agent.get_xfer_descs(kv_addrs, "DRAM", is_sorted=True)
        aux_addrs = [(self.mgr.args.aux_data_ptrs[0] + self.aux_index * self.mgr.args.aux_item_lens[0], self.mgr.args.aux_item_lens[0], 0)]
        aux_descs = self.mgr.agent.get_xfer_descs(aux_addrs, "DRAM", is_sorted=True)
        
        logging.info("[wytdebug] KVSender: kv_descs: %s", kv_descs)
        logging.info("[wytdebug] KVSender: remote_kv_descs: %s", remote_kv_descs)
        logging.info("[wytdebug] KVSender: peer_name: %s", self.mgr.peer_name)
        logging.info("[wytdebug] KVSender: str(self.bootstrap_room) %s", str(self.bootstrap_room))

        # Send KV
        self.xfer_handle = self.mgr.agent.initialize_xfer(
            "WRITE", kv_descs, remote_kv_descs, self.mgr.peer_name, str(self.bootstrap_room)
        )
        if not self.xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.mgr.agent.transfer(self.xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        # Send aux
        self.xfer_handle_aux = self.mgr.agent.initialize_xfer(
            "WRITE", aux_descs, remote_aux_descs, self.mgr.peer_name, str(self.bootstrap_room) + "_aux"
        )
        if not self.xfer_handle_aux:
            raise Exception("KVSender failed to create transfer")
        state = self.mgr.agent.transfer(self.xfer_handle_aux)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        self.has_sent = True

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            return KVPoll.WaitingForInput
        state = self.mgr.agent.check_xfer_state(self.xfer_handle)
        state2 = self.mgr.agent.check_xfer_state(self.xfer_handle_aux)
        if state == "ERR" or state2 == "ERR":
            raise Exception("KVSender transfer encountered an error.")
        if state == "DONE" and state2 == "DONE":
            return KVPoll.Success
        return KVPoll.WaitingForInput

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:
    def __init__(
        self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.has_init = False
        self.mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_transfer_done = False
        self.aux_transfer_done = False

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        kv_addrs = []
        for data_ptr, item_len in zip(self.mgr.args.kv_data_ptrs, self.mgr.args.kv_item_lens):
            for i in kv_indices:
                kv_addrs.append((tmp.data_ptr() , item_len, self.mgr.args.engine_rank))
        kv_descs = self.mgr.agent.get_xfer_descs(kv_addrs, "DRAM", is_sorted=True)
        aux_addrs = [(self.mgr.args.aux_data_ptrs[0] + aux_index * self.mgr.args.aux_item_lens[0], self.mgr.args.aux_item_lens[0], 0)]
        aux_descs = self.mgr.agent.get_xfer_descs(aux_addrs, "DRAM", is_sorted=True)
        logging.info(f'[wytdebug] KVReceiver: kv_descs: {kv_descs}, room: {self.bootstrap_room}')
        self.mgr.socket.send(self.mgr.agent.get_serialized_descs(kv_descs))
        self.mgr.socket.send(self.mgr.agent.get_serialized_descs(aux_descs))
        self.has_init = True

    def poll(self) -> KVPoll:
        if self.has_init is False:
            return KVPoll.WaitingForInput
        if not self.kv_transfer_done:
            self.kv_transfer_done = self.mgr.agent.check_remote_xfer_done(self.mgr.peer_name, str(self.bootstrap_room))
        if not self.aux_transfer_done:
            self.aux_transfer_done = self.mgr.agent.check_remote_xfer_done(self.mgr.peer_name, str(self.bootstrap_room) + "_aux")
        if self.kv_transfer_done and self.aux_transfer_done:
            return KVPoll.Success
        return KVPoll.WaitingForInput

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int): ...

    def poll(self) -> KVPoll: ...
