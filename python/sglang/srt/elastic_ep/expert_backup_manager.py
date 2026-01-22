import logging
import multiprocessing as mp
import torch
import zmq
import re
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.managers.io_struct import BackupDramReq
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_zmq_socket, get_local_ip_auto
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from mooncake.engine import TransferEngine


logger = logging.getLogger(__name__)

def extract_expert_id(param_name):
    pattern = r'\.experts\.(\d+)\.'
    match = re.search(pattern, param_name)
    if match:
        return int(match.group(1))
    return -1

class ExpertBackupManager:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args
        self.model_path = server_args.model_path
        self.load_format = server_args.load_format
        self.model_config = ModelConfig.from_server_args(server_args)
        self.continuous_buffer = None
        self.weight_pointer_map = {}
        self.transfer_engine = None
        self.session_id = None
        self.engine_num = server_args.nnodes
        self.engine_rank = server_args.node_rank
        self.expert_num = self.model_config.hf_config.n_routed_experts
        self.idmn = (self.expert_num // self.engine_num) * self.engine_rank
        self.idmx = (self.expert_num // self.engine_num) * (self.engine_rank + 1)
        context = zmq.Context(2)
        # TODO (stage 100): stop using localhost and extend to real multinode
        self.recv_from_expert_backup_client = get_zmq_socket(
            context, zmq.PULL, f"tcp://127.0.0.1:{10000 + server_args.node_rank * 2}", True
        )
        self.send_to_expert_backup_client = context.socket(zmq.PUB)
        self.send_to_expert_backup_client.bind(f"tcp://{get_local_ip_auto()}:{10000 + server_args.node_rank * 2 + 1}")

    def backup_weights_from_disk(self):
        load_config = LoadConfig(load_format=self.load_format)
        loader = get_model_loader(load_config, self.model_config)

        with set_default_torch_dtype(self.model_config.dtype):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(self.model_config, None)
            )

            total_bytes = 0
            weight_info_dict = {}

            for name, weight in iter:
                expert_id = extract_expert_id(name)
                if expert_id < self.idmx and expert_id >= self.idmn:
                    numel = weight.numel()
                    element_size = weight.element_size()
                    byte_size = numel * element_size
                    weight_info_dict[name] = {
                        'name': name,
                        'weight': weight,
                        'numel': numel,
                        'shape': weight.shape,
                        'dtype': weight.dtype,
                        'element_size': element_size,
                        'byte_size': byte_size
                    }
                    total_bytes += byte_size

            if total_bytes == 0:
                self.continuous_buffer = None
                self.weight_pointer_map = {}
                return

            self.continuous_buffer = torch.empty(total_bytes, dtype=torch.uint8, device='cpu')
            buffer_base_ptr = self.continuous_buffer.data_ptr()
            self.weight_pointer_map = {}
            current_byte_offset = 0

            for name in sorted(weight_info_dict.keys()):
                weight_info = weight_info_dict[name]
                weight = weight_info['weight']
                byte_size = weight_info['byte_size']
                weight_flat = weight.flatten().contiguous()
                weight_bytes = weight_flat.view(torch.uint8)
                start_byte = current_byte_offset
                end_byte = current_byte_offset + byte_size
                weight_ptr = buffer_base_ptr + current_byte_offset
                self.continuous_buffer[start_byte:end_byte].copy_(weight_bytes)
                self.weight_pointer_map[name] = {
                    'name': name,
                    'weight_ptr': weight_ptr,
                    'shape': weight_info['shape'],
                    'numel': weight_info['numel'],
                    'dtype': weight_info['dtype'],
                    'element_size': weight_info['element_size'],
                    'byte_size': byte_size,
                }

                current_byte_offset = end_byte

    def start_transfer_server(self):
        HOSTNAME = get_local_ip_auto()
        METADATA_SERVER = "P2PHANDSHAKE" # [ETCD_SERVER_URL, P2PHANDSHAKE, ...]
        PROTOCOL = "rdma"
        DEVICE_NAME = self.server_args.mooncake_ib_device

        self.transfer_engine = TransferEngine()
        self.transfer_engine.initialize(
            HOSTNAME,
            METADATA_SERVER,
            PROTOCOL,
            DEVICE_NAME
        )
        self.session_id = f"{HOSTNAME}:{self.transfer_engine.get_rpc_port()}"
        server_ptr = self.continuous_buffer.data_ptr()
        server_len = self.continuous_buffer.numel() * self.continuous_buffer.element_size()

        ret_value = self.transfer_engine.register_memory(server_ptr, server_len)
        if ret_value != 0:
            raise RuntimeError("Mooncake memory registration failed.")

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_expert_backup_client.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                logger.info("recv_req: %s", recv_req)
                self.backup_weights_from_disk()
                self.start_transfer_server()
                back_req = BackupDramReq(
                    _rank=self.engine_rank,
                    _map=self.weight_pointer_map,
                    session_id=self.session_id,
                    buffer_size=self.continuous_buffer.numel() * self.continuous_buffer.element_size()
                )
                self.send_to_expert_backup_client.send_pyobj(back_req)


def run_expert_backup_manager_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    set_global_server_args_for_scheduler(server_args)
    manager = ExpertBackupManager(server_args, port_args)
    manager.event_loop()


def run_expert_backup_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    proc = mp.Process(
        target=run_expert_backup_manager_process,
        args=(server_args, port_args),
    )
    proc.start()
    return proc
