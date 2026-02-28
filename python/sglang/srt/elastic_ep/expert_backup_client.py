import logging
import re
import threading
import time

import torch
import zmq

from sglang.srt.distributed.parallel_state import (
    get_world_group,
    get_world_size,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.managers.io_struct import UpdateExpertBackupReq
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_auto

PORT_BASE = envs.SGLANG_BACKUP_PORT_BASE.get()
logger = logging.getLogger(__name__)


def extract_layer_and_expert_id(param_name):
    pattern = r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+?)\."
    match = re.search(pattern, param_name)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return -1, -1, ""


class ExpertBackupClient:
    def __init__(self, server_args: ServerArgs, model_runner):
        context = zmq.Context(2)
        self.server_args = server_args
        self.engine_num = server_args.nnodes
        self.engine_rank = server_args.node_rank
        self.recv_list = [None] * self.engine_num
        self.ready_sockets = [None] * self.engine_num
        self.model_runner = model_runner
        self.moe_ep_size = model_runner.moe_ep_size
        self.model_config = model_runner.model_config
        self.moe_ep_rank = model_runner.moe_ep_rank
        self.dram_map_list = [None] * self.engine_num
        self.session_id_list = [None] * self.engine_num
        self.transfer_engine = None
        self.gpu_buffer = None
        self.buffer_size = 0
        self.use_backup = False
        local_ip = get_local_ip_auto()
        all_ips = [None] * get_world_size()
        torch.distributed.all_gather_object(
            all_ips, local_ip, group=get_world_group().cpu_group
        )
        logger.info(f"all_ips: {all_ips}")

        for i in range(self.engine_num):
            self.recv_list[i] = context.socket(zmq.SUB)
            self.recv_list[i].connect(
                f"tcp://{all_ips[i * get_world_size() // server_args.nnodes]}:{PORT_BASE + i * 2 + 1}"
            )
            self.recv_list[i].setsockopt(zmq.SUBSCRIBE, b"")

            # Synchronization channel to notify the manager when this client is ready.
            self.ready_sockets[i] = context.socket(zmq.PUSH)
            self.ready_sockets[i].connect(
                f"tcp://{all_ips[i * get_world_size() // server_args.nnodes]}:{PORT_BASE + i * 2}"
            )
            self.ready_sockets[i].send_pyobj(UpdateExpertBackupReq())

        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

    def _receive_loop(self):
        cnt = 0
        while cnt < self.engine_num:
            response = self.recv_list[cnt].recv_pyobj()
            self.dram_map_list[response.rank] = response.weight_pointer_map
            self.session_id_list[response.rank] = response.session_id
            self.buffer_size = max(self.buffer_size, response.buffer_size)
            cnt += 1

        self.use_backup = True
        self.start_transfer_client()

    def start_transfer_client(self):
        from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

        self.transfer_engine = get_mooncake_transfer_engine()

        self.params_dict = dict(self.model_runner.model.named_parameters())
        for name, param in self.params_dict.items():
            param_data = param.data
            ret_value = self.transfer_engine.engine.register_memory(
                param_data.data_ptr(), param_data.numel() * param_data.element_size()
            )
            if ret_value != 0:
                self.use_backup = False
                logger.warning("Register fails. Stop using expert weight backup!")
                break

    def update_weights(self):
        global_expert_location_metadata = get_global_expert_location_metadata()
        num_experts = (
            self.model_config.hf_config.n_routed_experts
            + self.server_args.ep_num_redundant_experts
        )
        num_local_experts = num_experts // self.moe_ep_size
        for i in range(self.engine_num):
            server_ptr_list = []
            local_ptr_list = []
            weight_size_list = []

            for name, weight_info in self.dram_map_list[i].items():
                layer_id, expert_id, weight_name = extract_layer_and_expert_id(name)
                if layer_id >= self.model_config.hf_config.num_hidden_layers:
                    continue

                if weight_name == "gate_proj":
                    shard_id = "w1"
                    param_name = "experts.w13_"
                elif weight_name == "down_proj":
                    shard_id = "w2"
                    param_name = "experts.w2_"
                elif weight_name == "up_proj":
                    shard_id = "w3"
                    param_name = "experts.w13_"
                else:
                    raise RuntimeError(f"Unknown weight name {weight_name}")

                name = name.replace(f"experts.{expert_id}.{weight_name}.", param_name)
                weight_param = self.params_dict[name]

                physical_expert_ids = (
                    global_expert_location_metadata.logical_to_all_physical(
                        layer_id, expert_id
                    )
                )
                for physical_expert_id in physical_expert_ids:
                    if physical_expert_id not in range(
                        num_local_experts * self.moe_ep_rank,
                        num_local_experts * (self.moe_ep_rank + 1),
                    ):
                        continue
                    param = weight_param[physical_expert_id % num_local_experts]
                    if shard_id == "w1":
                        param = param.narrow(0, 0, param.shape[0] // 2)
                    elif shard_id == "w3":
                        param = param.narrow(
                            0, param.shape[0] // 2, param.shape[0] // 2
                        )
                    server_ptr_list.append(weight_info["weight_ptr"])
                    local_ptr_list.append(param.data_ptr())
                    assert (
                        param.numel() * param.element_size() == weight_info["byte_size"]
                    )
                    weight_size_list.append(weight_info["byte_size"])
            before_transfer = time.time()
            ret = self.transfer_engine.engine.batch_transfer_sync_read(
                self.session_id_list[i],
                local_ptr_list,
                server_ptr_list,
                weight_size_list,
            )
            after_transfer = time.time()
            logger.info(f"transfer time = {after_transfer - before_transfer} s")

            if ret != 0:
                raise RuntimeError(
                    f"Failed to read weights from backup, error code: {ret}"
                )
        return
