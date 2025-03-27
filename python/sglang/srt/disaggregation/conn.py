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
    kv_tensors: List[torch.Tensor]
    aux_tensors: List[torch.Tensor]
    # kv_data_ptrs: list[int]
    # kv_data_lens: list[int]
    # kv_item_lens: list[int]
    # aux_data_ptrs: list[int]
    # aux_data_lens: list[int]
    # aux_item_lens: list[int]
    # ib_device: str


class KVManager:
    def __init__(self, args: KVArgs, mode: str):
        self.args = args
        if nixl_agent is None:
            raise Exception("NIXL is not available")
        self.agent = nixl_agent(str(uuid.uuid4()))
        print(f"Created NIXL agent for {self.args.engine_rank}")
        # TODO: do we also need to send aux_tensors?
        self.reg_descs = self.agent.register_memory(self.args.kv_tensors)
        if not self.reg_descs:
            raise Exception("NIXL memory registration failed")

        # Create socket connection between decode/prefill
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PAIR)
        if mode == "prefill":
            self.socket.connect("tcp://127.0.0.1:8998")
        elif mode == "decode":
            self.socket.bind("tcp://127.0.0.1:8998")
        print(f"{mode} socket connected")

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
        print(f"{mode} exchanged metadata")

class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        """
        bootstrap_addr: prefill: host address, decode: corresponding prefill address
        bootstrap_room: unique id per request
        """
        self.has_sent = False
        self.mgr = mgr

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        pass

    def send(self, kv_indices: npt.NDArray[np.int32]):
        print("KVSender begin send")
        # TODO: this currently include all buffers - use kv_indices
        target_descs = self.mgr.agent.deserialize_descs(self.mgr.socket.recv())
        self.xfer_handle = self.mgr.agent.initialize_xfer("READ", self.mgr.reg_descs.trim(), target_descs,
                                                     self.mgr.peer_name, "UUID") # TODO: change notification to bootstrap room?
        if not self.xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.mgr.agent.transfer(self.xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        self.has_sent = True

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            return KVPoll.WaitingForInput
        state = self.mgr.agent.check_xfer_state(self.xfer_handle)
        if state == "ERR":
            raise Exception("KVSender transfer encountered an error.")
        elif state == "DONE":
            return KVPoll.Success
        else:
            return KVPoll.WaitingForInput

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:
    def __init__(
        self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.has_init = False
        self.mgr = mgr

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        # Send descriptors of destination buffers
        # TODO: this currently include all buffers - use kv_indices
        self.mgr.socket.send(self.mgr.agent.get_serialized_descs(self.mgr.reg_descs.trim()))
        self.has_init = True
        print("KVReceiver sent descs")

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        # TODO: change notification to bootstrap room?
        if self.mgr.agent.check_remote_xfer_done(self.mgr.peer_name, "UUID"):
            return KVPoll.Success
        else:
            return KVPoll.WaitingForInput

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int): ...

    def poll(self) -> KVPoll: ...
