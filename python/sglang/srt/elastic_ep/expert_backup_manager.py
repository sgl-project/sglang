import logging
import multiprocessing as mp
import re
import signal

import torch
import zmq

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import BackupDramReq
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_local_ip_auto

PORT_BASE = envs.SGLANG_BACKUP_PORT_BASE.get()
logger = logging.getLogger(__name__)


def extract_expert_id(param_name):
    pattern = r"\.experts\.(\d+)\."
    match = re.search(pattern, param_name)
    if match:
        return int(match.group(1))
    return -1


class ExpertBackupManager:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
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
        # Synchronization socket to avoid PUB/SUB slow joiner issues.
        self.recv_from_expert_backup_client = context.socket(zmq.PULL)
        self.recv_from_expert_backup_client.bind(
            f"tcp://{get_local_ip_auto()}:{PORT_BASE + server_args.node_rank * 2}"
        )
        self.send_to_expert_backup_client = context.socket(zmq.PUB)
        self.send_to_expert_backup_client.bind(
            f"tcp://{get_local_ip_auto()}:{PORT_BASE + server_args.node_rank * 2 + 1}"
        )
        self.backup_weights_from_disk()
        self.start_transfer_server()

        # Block until all expert backup clients have reported readiness, to avoid
        # losing the initial PUB message due to slow joiners.
        num_ready_clients = 0

        while num_ready_clients < server_args.tp_size:
            self.recv_from_expert_backup_client.recv_pyobj()
            num_ready_clients += 1

        back_req = BackupDramReq(
            rank=self.engine_rank,
            weight_pointer_map=self.weight_pointer_map,
            session_id=self.session_id,
            buffer_size=self.continuous_buffer.numel()
            * self.continuous_buffer.element_size(),
        )
        self.send_to_expert_backup_client.send_pyobj(back_req)

        # Keep the manager subprocess alive until signals
        signal.pause()

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
                        "name": name,
                        "weight": weight,
                        "numel": numel,
                        "shape": weight.shape,
                        "dtype": weight.dtype,
                        "element_size": element_size,
                        "byte_size": byte_size,
                    }
                    total_bytes += byte_size

            if total_bytes == 0:
                self.continuous_buffer = None
                self.weight_pointer_map = {}
                return

            self.continuous_buffer = torch.empty(
                total_bytes, dtype=torch.uint8, device="cpu"
            )
            buffer_base_ptr = self.continuous_buffer.data_ptr()
            self.weight_pointer_map = {}
            current_byte_offset = 0

            for name in sorted(weight_info_dict.keys()):
                weight_info = weight_info_dict[name]
                weight = weight_info["weight"]
                byte_size = weight_info["byte_size"]
                weight_flat = weight.flatten().contiguous()
                weight_bytes = weight_flat.view(torch.uint8)
                start_byte = current_byte_offset
                end_byte = current_byte_offset + byte_size
                weight_ptr = buffer_base_ptr + current_byte_offset
                self.continuous_buffer[start_byte:end_byte].copy_(weight_bytes)
                self.weight_pointer_map[name] = {
                    "name": name,
                    "weight_ptr": weight_ptr,
                    "shape": weight_info["shape"],
                    "numel": weight_info["numel"],
                    "dtype": weight_info["dtype"],
                    "element_size": weight_info["element_size"],
                    "byte_size": byte_size,
                }

                current_byte_offset = end_byte

    def start_transfer_server(self):
        from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

        self.transfer_engine = get_mooncake_transfer_engine()
        self.session_id = self.transfer_engine.session_id
        server_ptr = self.continuous_buffer.data_ptr()
        server_len = (
            self.continuous_buffer.numel() * self.continuous_buffer.element_size()
        )

        ret_value = self.transfer_engine.engine.register_memory(server_ptr, server_len)
        if ret_value != 0:
            raise RuntimeError("Mooncake memory registration failed.")


def run_expert_backup_manager_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    set_global_server_args_for_scheduler(server_args)
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        init_mooncake_transfer_engine,
    )

    init_mooncake_transfer_engine(
        hostname=get_local_ip_auto(),
        gpu_id=0,
        ib_device=(
            server_args.disaggregation_ib_device or server_args.mooncake_ib_device
        ),
    )
    manager = ExpertBackupManager(server_args, port_args)


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
