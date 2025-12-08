# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, Optional, Tuple
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from sglang.srt.connector import BaseConnector
from sglang.srt.utils import init_custom_process_group

logger = logging.getLogger(__name__)


class RemoteInstanceConnector(BaseConnector):

    def __init__(self, url: str, device: torch.device = "cpu"):
        assert (
            device.type == "cuda"
        ), "RemoteInstanceConnector only supports cuda device."
        super().__init__(url)
        self.url = url
        self.device = device

    def build_group(
        self,
        gpu_id: int = -1,
        tp_rank: int = -1,
        instance_ip: str = None,
        group_rank: int = 1,
        world_size: int = 2,
    ):
        assert (
            self.device.type == "cuda"
        ), "RemoteInstanceConnector only supports cuda device."
        assert (
            gpu_id != -1 and tp_rank != -1
        ), "gpu_id and tp_rank must be specified for RemoteInstanceConnector. "

        self.device_id = torch.device(self.device.type, gpu_id)

        parsed_url = urlparse(self.url)
        master_address = parsed_url.hostname
        master_port = parsed_url.port
        group_name = f"send_weights_{instance_ip}_{master_port}_{tp_rank}"
        backend = "nccl"

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=self.device_id,
            )
            dist.barrier(group=self._model_update_group)
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    # Implemented as a no-op to make BaseConnector interface consistent.
    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        return

    # Implemented as a no-op to make BaseConnector interface consistent.
    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        return
