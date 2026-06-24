from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist

from sglang.srt.platforms import current_platform
from sglang.srt.utils import init_custom_process_group
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


# Mutable ``_weights_send_group`` dict prevents ``frozen=True``; explicit
# Rule-5 exception per the dataclass-defaults sprint-wide rule.
# tp_rank / tp_size / gpu_id read via ``self._model_runner`` (consistent with
# WeightUpdater) — no redundant storage.
@dataclass(slots=True, kw_only=True)
class WeightExporter:
    _model_runner: Any  # ModelRunner — kept untyped to avoid TYPE_CHECKING import here
    _weights_send_group: dict = field(default_factory=dict)

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self._model_runner.tp_size
        ), f"Expected {self._model_runner.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self._model_runner.tp_rank]
        group_name = f"{group_name}_{group_port}_{self._model_runner.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self._model_runner.tp_rank}, gpu_id={self._model_runner.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        current_platform.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self._model_runner.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        current_platform.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self._model_runner.tp_size
        ), f"Expected {self._model_runner.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self._model_runner.tp_rank]
        group_name = f"{group_name}_{group_port}_{self._model_runner.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        current_platform.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            for _, weights in self._model_runner.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        current_platform.empty_cache()
        return success, message

    def save_remote_model(self: WeightExporter, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(
            self._model_runner.model, self._model_runner.model_config.model_path, url
        )

    def save_sharded_model(
        self: WeightExporter,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self._model_runner.model, path, pattern, max_size)

    def get_weights_by_name(
        self: WeightExporter, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self._model_runner.model.get_weights_by_name(
                name, truncate_size, tp_size=self._model_runner.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None
