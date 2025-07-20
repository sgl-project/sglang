# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from sglang.srt.connector import BaseConnector
from sglang.srt.utils import init_custom_process_group

logger = logging.getLogger(__name__)

class RemoteInstanceConnector(BaseConnector):

    def __init__(self, url: str, device: torch.device = "cpu"):
        assert(
            device.type == "cuda"
        ), "RemoteInstanceConnector only supports cuda device."
        super().__init__(url, device)
        self.url = url

    def build_group(self, gpu_id: int = -1, tp_rank: int = -1, client_instance_id: str = None, group_rank: int = 1, world_size: int = 2):
        assert(
            self.device.type == "cuda"
        ), "RemoteInstanceConnector only supports cuda device."
        assert(
            gpu_id != -1 and tp_rank != -1
        ), "gpu_id and tp_rank must be specified for RemoteInstanceConnector. "

        self.device_id = torch.device(self.device.type, gpu_id)

        parsed_url = urlparse(self.url)

        self.init_load_weights_group(
            master_address=parsed_url.hostname,
            master_port=parsed_url.port,
            rank_offset=group_rank,
            world_size=world_size,
            group_name=f"send_weights_{client_instance_id}_{tp_rank}",
            backend="nccl",
            device_id=self.device_id,
        )

    def init_load_weights_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        device_id=None,
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in loading weights from remote instance,
        where rank 0 is the seed instance, and the other rank is the client instance,
        which needs to load weights for initialization.
        """
        assert group_name != "", "Group name cannot be empty"

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank_offset,
                group_name=group_name,
                device_id=device_id,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def load_weights_from_remote_instance(self, name, dtype, shape):
        target_dtype = (
            dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        )

        assert (
            self._model_update_group is not None
        ), "model update group must be initialized"

        try:
            weights = torch.empty(shape, dtype=target_dtype, device=self.device)
            dist.broadcast(weights, src=0, group=self._model_update_group)
            return weights

        except Exception as e:
            error_msg = (
                f"Failed to load weights from remote instance: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        return

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        return
