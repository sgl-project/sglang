from __future__ import annotations
import logging
import subprocess
import threading
import json
import asyncio
import os
import time

import requests

from sglang.srt.bootstrap.app import start_bootstrap_server

import uuid
from typing import Dict, Optional
import numpy as np
from sglang.srt.bootstrap.rdma_utils import RdmaServer, RdmaClient

from python.sglang.srt.disaggregation.group_indics import groups_by_continuity_numpy
from python.sglang.srt.utils import get_local_ip_by_remote

logger = logging.getLogger(__name__)

from sglang.srt.utils import get_open_port
from sglang.srt.disaggregation.ib_devices import find_best_roce_for_gpu


class KVBootstrapServer:
    def __init__(self, port: int):
        self.bootstrap_server_port = port
        self.start_server()

    def start_server(self):
        server = start_bootstrap_server("0.0.0.0", self.bootstrap_server_port)
        logger.info(" bootstrap server started")


class KVArgs:
    """Arguments for KV cache management"""
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str = "all"


class KVManager:
    def __init__(self, args: KVArgs, bootstrap_server: KVBootstrapServer = None):
        self.args = args
        self.engine_rank = args.engine_rank
        self.kv_data_ptrs = args.kv_data_ptrs
        self.kv_data_lens = args.kv_data_lens
        self.kv_item_lens = args.kv_item_lens
        self.aux_data_ptrs = args.aux_data_ptrs
        self.aux_data_lens = args.aux_data_lens
        self.aux_item_lens = args.aux_item_lens

        self.bootstrap_server = bootstrap_server
        self.args.ib_device, net_card = find_best_roce_for_gpu(self.args.gpu_id)
        if self.args.ib_device:
            logger.info(
                "Current Process Using the  gpu id: {}, ib_device: {} net:{}".format(self.args.gpu_id,
                                                                                     self.args.ib_device,
                                                                                     net_card))
        else:
            raise Exception("No ROCE IB device found...")

    def set_bootstrap_server(self, bootstrap_server):
        self.bootstrap_server = bootstrap_server

    def calculate_token_kv_address(self, layer_id: int, token_index: int):
        # Get base address - KV data pointer for each layer
        base_address = self.args.kv_data_ptrs[layer_id]
        # KV data size for each token
        token_kv_size = self.args.kv_item_lens[layer_id]
        # Calculate offset based on token index
        offset = token_kv_size * token_index
        # Final address = base address + offset
        token_kv_address = base_address + offset
        return token_kv_address, offset

    def caculate_layer_kv_addresses(self, token_indices: list[int]):
        addresses_base_and_len = []
        for layer_id in range(len(self.args.kv_data_ptrs)):
            # 每个token的KV数据大小
            token_kv_size = self.args.kv_item_lens[layer_id]
            # 计算偏移量
            offset = token_kv_size * token_indices[0]
            token_kv_layer_base_address = self.args.kv_data_ptrs[layer_id] + offset
            addresses_base_and_len.append((token_kv_layer_base_address,
                                           token_kv_size * len(token_indices)))
        return addresses_base_and_len


class KVPoll:
    """Status codes for KV operations"""
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.state = KVPoll.Bootstrapping

        logger.info(f"Sender registered with room {self.bootstrap_room}")

        # Network configuration
        self.target_ip = None

        # Memory management
        self.mrs_to_send = []  # Memory regions for data segments to be sent
        self.meta_has_sent = False  # Flag indicating if metadata has been sent

    def handshake(self):
        """Establish connection with the receiver through bootstrap server"""
        resp = requests.get(f"http://{self.bootstrap_addr}/get_room_info/{self.bootstrap_room}")

        if resp.status_code == 200:
            data = resp.json()
            return data
        return None

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):

        """Initialize sender with metadata only

        Args:
            num_tokens: Number of tokens to transfer
            aux_idx: Index for auxiliary data

        Returns:
            bool: True if metadata sent successfully
        """
        self.num_tokens = num_kv_indices
        self.aux_idx = aux_index
        metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        metadata_ptr_length = self.mgr.aux_item_lens[0]

        try:
            self.qp = RdmaClient(host_ip=self.target_ip, ib_device=self.mgr.args.ib_device,
                                 socket_port=self.target_port)
            if self.qp.init(metadata_ptr, metadata_ptr_length):
                logger.debug("Transferring...")
                self.state = KVPoll.Transferring
        except Exception as e:
            print(e)
            self.state = KVPoll.Bootstrapping

    def poll(self) -> KVPoll:
        """Poll transfer status"""
        if self.state == KVPoll.Bootstrapping:
            data = self.handshake()
            if not data:
                self.state = KVPoll.Bootstrapping
            else:
                logger.debug(data)
                self.target_ip = data.get(str(self.mgr.engine_rank))['ip']
                self.target_port = data.get(str(self.mgr.engine_rank))['port']

                self.state = KVPoll.WaitingForInput
        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        if self.state == KVPoll.Transferring:
            self.qp.check_send_complete()

            '''
            completed wrs + metadata wrs
            '''
            if self.qp.completed_wrs == len(self.qp.wrs_to_send) + 1 and self.meta_has_sent:
                print("Transferring complete")
                # 写入远端 metadata //todo
                self.state = KVPoll.Success
            elif self.qp.completed_wrs == len(self.qp.wrs_to_send) and not self.meta_has_sent:
                self.qp.send_metadata_wrs()
                self.meta_has_sent = True

        return self.state

    def send(self, kv_indices: np.ndarray[np.int32]):
        """Send actual data synchronously"""
        # 收集要传输的数据
        groups_mrs_info = []
        continous_indices = groups_by_continuity_numpy(kv_indices)
        for group_id, continue_kv_indices in enumerate(continous_indices):
            mrs_info = []
            address_lengths = self.mgr.caculate_layer_kv_addresses(continue_kv_indices)
            for layer_id, (address, length) in enumerate(address_lengths):
                mr = self.qp.create_mr(address, length)
                self.mrs_to_send.append(mr)
                mrs_info.append({
                    "address": address,
                    "length": length,
                    "rkey": mr.rkey,
                    'lkey': mr.lkey
                })
            groups_mrs_info.append(mrs_info)
        self.qp.send_wrs(groups_mrs_info)


class KVReceiver:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None):
        self.kv_layers_mrs = []
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.ep = None
        self.state = KVPoll.Bootstrapping
        self.num_tokens = 0
        self.aux_idx = -1
        self.kv_indices = None

        # Network setup
        self.rdma_port = get_open_port()

        # For metrics
        self.start_time = time.time()

        # todo get dynamic ip
        self.ip_address = get_local_ip_by_remote(self.bootstrap_addr)
        self.qp = RdmaServer(socket_port=self.rdma_port, ib_device=self.mgr.args.ib_device)

        # Initialize connection
        # todo can use other rapid method...
        self.handshake()
        self.mrs_to_receive = []  # Memory regions for receiving data segments

    def handshake(self):
        """Establish connection with the bootstrap server"""
        post_data = {
            "room_id": self.bootstrap_room,
            "session_id": self.session_id,
            "engine_rank": self.mgr.args.engine_rank,
            "ib_device": self.mgr.args.ib_device,
            "ip_addr": {
                "ip": self.ip_address,
                "port": self.rdma_port
            }
        }
        http_start = time.time()
        resp = requests.post(f"http://{self.bootstrap_addr}/handshake", json=post_data)
        http_end = time.time()
        print(f"HD Request time: {http_end - http_start}")
        if resp.status_code != 200:
            self.state = KVPoll.Failed
            print(resp.status_code)
        else:
            self.state = KVPoll.WaitingForInput
            self.initialized = True
            print("boostraped success..")

    def init(self, kv_indices: np.ndarray[np.int32], aux_index: Optional[int] = None):
        """Initialize receiver with KV indices and auxiliary data index

        Args:
            kv_indices: Array of KV indices to receive
            aux_index: Optional index for auxiliary data

        Returns:
            bool: True if initialization successful
        """

        metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        metadata_length = self.mgr.aux_item_lens[0]
        # 创建每一岑layer的mr 得到对应的key 传给客户端
        rkeys = []

        for layer_id, base_addr in enumerate(self.mgr.kv_data_ptrs):
            layer_mr = self.qp.create_mr(base_addr, self.mgr.kv_data_lens[layer_id])
            rkeys.append(layer_mr.rkey)
            self.kv_layers_mrs.append(layer_mr)
        # todo 根据kv_indics的连续性 来判断 地址连续性，借此可以动态创建 较大的 MR
        groups_mrs_info = []
        continous_indices = groups_by_continuity_numpy(kv_indices)
        for group_id, continue_kv_indices in enumerate(continous_indices):
            mrs_info = []
            address_lengths = self.mgr.caculate_layer_kv_addresses(continue_kv_indices)
            for layer_id, (address, length) in enumerate(address_lengths):
                mrs_info.append({
                    "address": address,
                    "length": length,
                    "rkey": rkeys[layer_id]
                })
            groups_mrs_info.append(mrs_info)

        try:
            self.qp.init(groups_mrs_info, metadata_ptr, metadata_length)
            self.state = KVPoll.Transferring
            self.qp.recv_metadata_mr()

        except Exception as e:
            self.state = KVPoll.Bootstrapping
            return

    def poll(self) -> KVPoll:
        """Poll receive status and handle state transitions"""
        if not self.initialized:
            return KVPoll.Bootstrapping

        if self.state == KVPoll.Transferring:
            self.qp.check_complete()
            # Check if metadata transfer is complete
            if self.qp.metadata_mr_complete_num == 1:
                logger.debug("Decode Transferring complete...")
                return KVPoll.Success

        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        return self.state

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'loop') and self.loop:
            self.loop.close()
