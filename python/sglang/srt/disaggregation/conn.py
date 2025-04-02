from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt
import uuid
import zmq
import pickle
import time

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
        if mode not in ["prefill", "decode"]:
            raise Exception("Mode must be prefill or decode")
        self.args = args
        if nixl_agent is None:
            raise Exception("NIXL is not available")
        self.agent = nixl_agent(str(uuid.uuid4()))
        print(f"Created NIXL agent for {self.args.engine_rank}")
        print("num kv tensors:", len(args.kv_tensors))
        for x in args.kv_tensors:
            print(x.shape, x.get_device())
        print(args.kv_tensors[0][0, :, :].get_device())
        print("num aux tensors:", len(args.aux_tensors))
        for x in args.aux_tensors:
            print(x.shape, x.get_device())
        print(args.aux_tensors[0][0, :].get_device())
        self.kv_descs = self.agent.register_memory(self.args.kv_tensors)
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        self.aux_descs = self.agent.register_memory(self.args.aux_tensors)
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

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

        # Create prepped xfer handle on sender
        num_blocks = self.args.kv_tensors[0].shape[0]
        start = time.time()
        if mode == "prefill":
            # KV
            # Make remote prepped xfer list
            remote_block_data = pickle.loads(self.socket.recv())
            self.remote_prep_handle = self.agent.prep_xfer_dlist(self.peer_name, remote_block_data, "VRAM")
            if not self.remote_prep_handle:
                raise Exception("KVSender failed to create remote prep_xfer_dlist")
            # Make local prepped xfer list
            blocks = []
            for kv_layer in self.args.kv_tensors:
                blocks.extend([kv_layer[i, :, :] for i in range(num_blocks)])
            self.local_prep_handle = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", blocks)
            if not self.local_prep_handle:
                raise Exception("KVSender failed to create local prep_xfer_dlist")
            
            # AUX
            remote_aux_data = pickle.loads(self.socket.recv())
            self.remote_prep_handle_aux = self.agent.prep_xfer_dlist(self.peer_name, remote_aux_data, "DRAM")
            if not self.remote_prep_handle_aux:
                raise Exception("KVSender failed to create remote prep_xfer_dlist")
            aux = []
            for i in range(self.args.aux_tensors[0].shape[0]):
                aux.append(self.args.aux_tensors[0][i, :])
            self.local_prep_handle_aux = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", aux)
            if not self.local_prep_handle_aux:
                raise Exception("KVSender failed to create local prep_xfer_dlist")
        elif mode == "decode":
            # kv
            blocks = []
            for kv_layer in self.args.kv_tensors:
                blocks.extend([(kv_layer[i, :, :].data_ptr(), kv_layer[i, :, :].nbytes, kv_layer.get_device()) for i in range(num_blocks)])
            self.socket.send(pickle.dumps(blocks))
            # aux
            aux = []
            print("aux base ", self.args.aux_tensors[0].data_ptr())
            print("aux size ", self.args.aux_tensors[0].nbytes)
            print("aux[0, :] base ", self.args.aux_tensors[0][0, :].data_ptr())
            print("aux[0, :] size ", self.args.aux_tensors[0][0, :].nbytes)
            for i in range(self.args.aux_tensors[0].shape[0]):
                # DRAM. use 0 for device_id
                aux.append((self.args.aux_tensors[0][i, :].data_ptr(), self.args.aux_tensors[0][i, :].nbytes, 0))
            self.socket.send(pickle.dumps(aux))
        print("total time for prep dlist ", time.time() - start)

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
        self.bootstrap_room = bootstrap_room
        self.num_layers = len(self.mgr.args.kv_tensors)
        self.num_blocks = self.mgr.args.kv_tensors[0].shape[0]

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        print(f"kvsender init() num_kv_indices={num_kv_indices} aux_index={aux_index}")
        self.aux_index = aux_index

    def send(self, kv_indices: npt.NDArray[np.int32]):
        print(f"KVSender begin send kv_indices={kv_indices}")
        remote_kv_indices, remote_aux_index = pickle.loads(self.mgr.socket.recv())
        print(f"KVSender got remote kv_indices={remote_kv_indices} remote_aux_index={remote_aux_index}")
        # Send kv
        self.xfer_handle = self.mgr.agent.make_prepped_xfer(
            "WRITE",
            self.mgr.local_prep_handle,
            self._kv_indices_to_dlist_indices(kv_indices),
            self.mgr.remote_prep_handle,
            self._kv_indices_to_dlist_indices(remote_kv_indices),
            str(self.bootstrap_room),
        )
        if not self.xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.mgr.agent.transfer(self.xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        # send aux
        self.xfer_handle_aux = self.mgr.agent.make_prepped_xfer(
            "WRITE",
            self.mgr.local_prep_handle_aux,
            [self.aux_index],
            self.mgr.remote_prep_handle_aux,
            [remote_aux_index],
            str(self.bootstrap_room) + "_aux",
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
    
    
    def _kv_indices_to_dlist_indices(self, kv_indices):
        result = []
        for i in range(self.num_layers):
            for idx in kv_indices:
                result.append(i * self.num_blocks + idx)
        return result


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
        print(f"kvreceiver init() kv_indices={kv_indices} aux_index={aux_index}")
        self.mgr.socket.send(pickle.dumps((kv_indices, aux_index)))
        self.has_init = True
        print("KVReceiver sent descs")

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
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
