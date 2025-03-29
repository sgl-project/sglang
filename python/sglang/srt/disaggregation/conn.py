from __future__ import annotations

import logging
from typing import Optional, Dict, Tuple
from sglang.srt.disaggregation.transfer_engine.mooncake import MooncakeTransferEngine
import threading
from functools import cache

import zmq
import struct

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str


RequestPoolType = Dict[int, Tuple[npt.NDArray[np.int32], Optional[int]]]
KVSENDER_POLLING_PORT = 17788
KVRECIVER_POLLING_PORT = 17789


class KVManager:
    # TODO: make it general and support multiple transfer backend before merging
    def __init__(self, args: KVArgs):
        self.engine = MooncakeTransferEngine()
        self.kv_args = args
        self.request_pool: RequestPoolType = {
            0: (np.array([0], dtype=np.int32), None)
        }
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        self.prefill_thread_started = False
        self.decode_thread_started = False

    def register_buffer_to_engine(self):
        for kv_data_ptr, kv_data_len in zip(self.kv_args.kv_data_ptrs,
                                            self.kv_args.kv_data_lens):
            self.engine.register(kv_data_ptr, kv_data_len)

        for aux_data_ptr, aux_data_len in zip(self.kv_args.aux_data_ptrs,
                                              self.kv_args.aux_data_lens):
            self.engine.register(aux_data_ptr, aux_data_len)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def send_kvcache(self, endpoint: str, bootstrap_room: int,
                     dst_ptrs: list[int],
                     dst_kv_indices: npt.NDArray[np.int32]):
        prefill_indices, _ = self.request_pool[bootstrap_room]
        layer_num = int(len(self.kv_args.kv_data_ptrs) / 2)
        for layer_id in range(layer_num):
            prefill_key_layer_ptr = self.kv_args.kv_data_ptrs[layer_id]
            key_item_len = self.kv_args.kv_item_lens[layer_id]
            prefill_value_layer_ptr = self.kv_args.kv_data_ptrs[layer_num +
                                                                layer_id]
            value_item_len = self.kv_args.kv_item_lens[layer_num + layer_id]

            decode_key_layer_ptr = dst_ptrs[layer_id]
            decode_value_layer_ptr = dst_ptrs[layer_num + layer_id]
            # TODO: Maybe combine multiple contiguous indices into one transfer_sync op
            for prefill_index, decode_index in zip(prefill_indices,
                                                   dst_kv_indices):
                prefill_key_addr = prefill_key_layer_ptr + prefill_index * key_item_len
                decode_key_addr = decode_key_layer_ptr + decode_index * key_item_len
                # TODO: mooncake transfer engine can do async transfer. Do async later
                self.engine.transfer_sync(endpoint, prefill_key_addr,
                                          decode_key_addr, key_item_len)

                prefill_value_addr = prefill_value_layer_ptr + prefill_index * value_item_len
                decode_value_addr = decode_value_layer_ptr + decode_index * value_item_len
                # TODO: mooncake transfer engine can do async transfer. Do async later
                self.engine.transfer_sync(endpoint, prefill_value_addr,
                                          decode_value_addr, value_item_len)

    def send_aux(self, endpoint: str, bootstrap_room: int,
                 dst_aux_ptrs: list[int], dst_aux_index: int):
        _, prefill_aux_index = self.request_pool[bootstrap_room]
        aux_item_len = self.kv_args.aux_data_lens[0]
        prefill_aux_addr = self.kv_args.aux_data_ptrs[
            0] + prefill_aux_index * aux_item_len
        decode_aux_addr = dst_aux_ptrs[0] + dst_aux_index * aux_item_len
        # TODO: mooncake transfer engine can do async transfer. Do async later
        # Not sure about the amount of aux data, maybe transfer it by zmq is more effective
        self.engine.transfer_sync(endpoint, prefill_aux_addr, decode_aux_addr,
                                  aux_item_len)

    def start_prefill_thread(self):
        if self.prefill_thread_started:
            return
        self.prefill_thread_started = True
        sender_rank_port = KVSENDER_POLLING_PORT + self.kv_args.engine_rank
        self.server_socket.bind("tcp://*:" + str(sender_rank_port))

        def prefill_thread():
            while True:
                (endpoint, bootstrap_room, dst_ptrs, dst_kv_indices,
                 dst_aux_ptrs,
                 dst_aux_index) = self.server_socket.recv_multipart()
                if bootstrap_room.decode('ascii') == 'None':
                    continue
                endpoint = endpoint.decode('ascii')
                bootstrap_room = int(bootstrap_room.decode('ascii'))
                dst_ptrs = list(struct.unpack(f'{len(dst_ptrs)//8}q',
                                              dst_ptrs))
                dst_kv_indices = np.frombuffer(dst_kv_indices, dtype=np.int32)
                dst_aux_ptrs = list(
                    struct.unpack(f'{len(dst_aux_ptrs)//8}q', dst_aux_ptrs))
                dst_aux_index = int(dst_aux_index.decode('ascii'))
                self.send_kvcache(endpoint, bootstrap_room, dst_ptrs,
                                  dst_kv_indices)
                self.send_aux(endpoint, bootstrap_room, dst_aux_ptrs,
                              dst_aux_index)
                self.request_pool.pop(bootstrap_room)
                self._connect("tcp://" + endpoint + ":" +
                              str(KVRECIVER_POLLING_PORT +
                                  self.kv_args.engine_rank)).send_multipart([
                                      str(bootstrap_room).encode('ascii'),
                                      "Done",
                                  ])

        threading.Thread(target=prefill_thread).start()

    def start_decode_thread(self):
        if self.decode_thread_started:
            return
        self.decode_thread_started = True
        reciver_rank_port = KVRECIVER_POLLING_PORT + self.kv_args.engine_rank
        self.server_socket.bind("tcp://*:" + str(reciver_rank_port))

        def decode_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                bootstrap_room = int(bootstrap_room.decode('ascii'))
                self.request_pool.pop(bootstrap_room)

        threading.Thread(target=decode_thread).start()

    def enqueue_request(self, bootstrap_room: int,
                        kv_indices: npt.NDArray[np.int32],
                        aux_index: Optional[int]):
        self.request_pool[bootstrap_room] = (kv_indices, aux_index)

    def has_finished(self, bootstrap_room: int):
        if bootstrap_room in self.request_pool:
            return False
        return True

    def get_localhost(self):
        return self.engine.get_localhost()


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:

    def __init__(self, mgr: KVManager, bootstrap_addr: str,
                 bootstrap_room: int):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.has_sent = False
        self.kv_mgr.start_prefill_thread()

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index
        self.num_kv_indices = num_kv_indices

    def send(self, kv_indices: npt.NDArray[np.int32]):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices,
                                    self.aux_index)

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            if self.kv_mgr.has_finished(self.bootstrap_room):
                self.has_sent = True
                return KVPoll.Success
            return KVPoll.WaitingForInput
        else:
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:

    def __init__(self,
                 mgr: KVManager,
                 bootstrap_addr: str,
                 bootstrap_room: Optional[int] = None):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self.prefill_server_url = (
            bootstrap_addr.split(":")[0] + ":" +
            str(KVSENDER_POLLING_PORT + self.kv_mgr.kv_args.engine_rank))
        self.decode_ip = self.kv_mgr.get_localhost()
        self.kv_mgr.start_decode_thread()
        self.has_init = False

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self,
             kv_indices: npt.NDArray[np.int32],
             aux_index: Optional[int] = None):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, aux_index)
        packed_kv_data_ptrs = b''.join(
            struct.pack('q', ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs)
        packed_aux_data_ptrs = b''.join(
            struct.pack('q', ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs)
        self._connect("tcp://" + self.prefill_server_url).send_multipart([
            self.decode_ip.encode('ascii'),
            str(self.bootstrap_room).encode('ascii'),
            packed_kv_data_ptrs,
            kv_indices.tobytes(),
            packed_aux_data_ptrs,
            str(aux_index).encode('ascii'),
        ])

    def poll(self) -> KVPoll:
        if self.has_init is False:
            if self.kv_mgr.has_finished(self.bootstrap_room):
                self.has_init = True
                return KVPoll.Success
            return KVPoll.WaitingForInput
        else:
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:

    def __init__(self, port: int):
        ...

    def poll(self) -> KVPoll:
        ...
