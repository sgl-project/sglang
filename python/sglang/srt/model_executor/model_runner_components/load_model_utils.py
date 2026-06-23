from __future__ import annotations

import logging
import socket
import threading
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def maybe_downgrade_dtype_for_legacy_gpu(
    *, server_args: ServerArgs, model_config: ModelConfig
) -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        logger.info(
            "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
        )
        server_args.dtype = "float16"
        model_config.dtype = torch.float16
        if torch.cuda.get_device_capability()[1] < 5:
            raise RuntimeError("SGLang only supports sm75 and above.")


def maybe_trigger_remote_instance_nccl_send_group(
    *, server_args: ServerArgs, tp_rank: int
) -> None:
    if (
        server_args.load_format == LoadFormat.REMOTE_INSTANCE
        and server_args.remote_instance_weight_loader_backend
        == RemoteInstanceWeightLoaderBackend.NCCL
    ):
        if tp_rank == 0:
            instance_ip = NetworkAddress.resolve_host(socket.gethostname())
            t = threading.Thread(
                target=trigger_init_weights_send_group_for_remote_instance_request,
                args=(
                    server_args.remote_instance_weight_loader_seed_instance_ip,
                    server_args.remote_instance_weight_loader_seed_instance_service_port,
                    server_args.remote_instance_weight_loader_send_weights_group_ports,
                    instance_ip,
                ),
            )
            t.start()
