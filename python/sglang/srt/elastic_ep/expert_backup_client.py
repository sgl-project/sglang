import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import re
import einops
import torch
import torch.distributed
from torch.distributed import P2POp
import zmq
import threading
import time
import numpy as np
from mooncake.engine import TransferEngine
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.distributed.parallel_state import (
    get_world_group,
    get_world_rank,
    get_world_size,
)

from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.managers.io_struct import UpdateExpertBackupReq
from sglang.srt.server_args import get_global_server_args, ServerArgs
from sglang.srt.utils import get_bool_env_var, get_local_ip_auto, get_zmq_socket
from mooncake.engine import TransferEngine

logger = logging.getLogger(__name__)

def extract_layer_and_expert_id(param_name):
    pattern = r'layers\.(\d+)\.mlp\.experts\.(\d+)\.'
    match = re.search(pattern, param_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1

class ExpertBackupClient:
    def __init__(self, server_args: ServerArgs, model_runner):
        context = zmq.Context(2)
        self.send_to_backup_manager = get_zmq_socket(
            context, zmq.PUSH, f"tcp://127.0.0.1:{10000 + server_args.node_rank * 2}", False
        )
        self.server_args = server_args
        self.engine_num = server_args.nnodes
        self.engine_rank = server_args.node_rank
        self.recv_list = [None] * self.engine_num
        self.model_runner = model_runner
        self.moe_ep_size = model_runner.moe_ep_size
        self.model_config = model_runner.model_config
        self.moe_ep_rank = model_runner.moe_ep_rank
        self.dram_map_list = [None] * self.engine_num
        self.session_id_list = [None] * self.engine_num
        self.transfer_engine = None
        self.gpu_buffer = None
        self.buffer_size = 0

        local_ip = get_local_ip_auto()
        all_ips = [None] * get_world_size()
        torch.distributed.all_gather_object(all_ips, local_ip, group=get_world_group().cpu_group)
        logger.info(f"all_ips: {all_ips}")

        for i in range(self.engine_num):
            self.recv_list[i] = context.socket(zmq.SUB)
            self.recv_list[i].connect(f"tcp://{all_ips[i * get_world_size() // server_args.nnodes]}:{10000 + i * 2 + 1}")
            self.recv_list[i].setsockopt(zmq.SUBSCRIBE, b"")

        if get_world_rank() % (get_world_size() // server_args.nnodes) == 0:
            self.send_to_backup_manager.send_pyobj(UpdateExpertBackupReq())

        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

    def _receive_loop(self):
        cnt = 0
        while True:
            for i in range(self.engine_num):
                try:
                    response = self.recv_list[i].recv_pyobj()
                except zmq.ZMQError:
                    continue
                self.dram_map_list[response._rank] = response._map
                self.session_id_list[response._rank] = response.session_id
                self.buffer_size = max(self.buffer_size, response.buffer_size)
                cnt += 1
                if cnt == self.engine_num:
                    self.model_runner.if_backup = True
                    self.start_transfer_client()
    
    def start_transfer_client(self):
        HOSTNAME = get_local_ip_auto()
        METADATA_SERVER = "P2PHANDSHAKE"
        PROTOCOL = "rdma"
        DEVICE_NAME = self.server_args.mooncake_ib_device
        self.transfer_engine = TransferEngine()
        self.transfer_engine.initialize(
            HOSTNAME,
            METADATA_SERVER,
            PROTOCOL,
            DEVICE_NAME
        )

        self.params_dict = dict(self.model_runner.model.named_parameters())
        for name, param in self.params_dict.items():
            param_data = param.data
            ret_value = self.transfer_engine.register_memory(
                param_data.data_ptr(), param_data.numel() * param_data.element_size()
            )
            if ret_value != 0:
                raise RuntimeError("GPU buffer memory registration failed.")
    
    def update_weights(self):
        global_expert_location_metadata = get_global_expert_location_metadata()
        num_experts = self.model_config.hf_config.n_routed_experts + self.server_args.ep_num_redundant_experts
        num_local_experts = num_experts // self.moe_ep_size
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )
        for i in range(self.engine_num):
            server_ptr_list = []
            local_ptr_list = []
            weight_size_list = []

            for name, weight_info in self.dram_map_list[i].items():
                layer_id, expert_id = extract_layer_and_expert_id(name)
                if layer_id >= self.model_config.hf_config.num_hidden_layers:
                    continue
                if "mlp.experts" in name and "mlp.shared_experts" not in name:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        physical_expert_ids = global_expert_location_metadata.logical_to_all_physical(
                            layer_id, expert_id
                        )
                        for physical_expert_id in physical_expert_ids:
                            if physical_expert_id not in range(
                                num_local_experts * self.moe_ep_rank,
                                num_local_experts * (self.moe_ep_rank + 1)
                            ):
                                continue
                            name = name.replace(weight_name, param_name)
                            param = self.params_dict[name]
                            param = param[physical_expert_id % num_local_experts]
                            if shard_id == "w1":
                                param = param.narrow(0, 0, param.shape[0] // 2)
                            elif shard_id == "w3":
                                param = param.narrow(0, param.shape[0] // 2, param.shape[0] // 2)
                            weight_info['tensor'] = param
                            server_ptr_list.append(weight_info['weight_ptr'])
                            local_ptr_list.append(weight_info['tensor'].data_ptr())
                            assert weight_info['tensor'].numel() * weight_info['tensor'].element_size() == \
                                    weight_info['byte_size']
                            weight_size_list.append(weight_info['byte_size'])
            before_transfer = time.time()
            ret = self.transfer_engine.batch_transfer_sync_read(
                self.session_id_list[i],
                local_ptr_list,
                server_ptr_list,
                weight_size_list
            )
            after_transfer = time.time()
            logger.info(f"transfer time = {after_transfer - before_transfer} s")

            if ret != 0:
                raise RuntimeError(f"Failed to read weights from backup, error code: {ret}")
        return