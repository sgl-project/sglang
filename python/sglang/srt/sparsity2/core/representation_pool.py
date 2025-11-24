import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class RepresentationPool:
    """Storage pool for sparse attention representations."""

    def __init__(
        self,
        total_num_pages: int,
        start_layer: int,
        end_layer: int,
        device: torch.device,
    ):
        self.total_num_pages = total_num_pages
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.storages: Dict[str, Dict[int, torch.Tensor]] = {}

    def register_storage(self, name: str, shape: tuple, dtype: torch.dtype) -> None:
        if name in self.storages:
            logger.warning(f"Storage '{name}' already registered")
            return

        layer_pools = {
            layer_id: torch.zeros(
                (self.total_num_pages, *shape), dtype=dtype, device=self.device
            )
            for layer_id in range(self.start_layer, self.end_layer)
        }
        self.storages[name] = layer_pools
        logger.info(
            f"Registered storage '{name}': shape={shape}, dtype={dtype}, layers={self.end_layer - self.start_layer}"
        )

    def store(
        self,
        layer_id: int,
        storage_name: str,
        physical_page_ids: torch.Tensor,
        data: torch.Tensor,
    ) -> None:
        if storage_name not in self.storages:
            raise ValueError(f"Storage '{storage_name}' not registered")

        target_dtype = self.storages[storage_name][layer_id].dtype
        if data.dtype != target_dtype:
            data = data.to(dtype=target_dtype)

        self.storages[storage_name][layer_id][physical_page_ids] = data.to(
            device=self.device
        )

    def retrieve(
        self, layer_id: int, storage_name: str, physical_page_ids: torch.Tensor
    ) -> torch.Tensor:
        if storage_name not in self.storages:
            raise ValueError(f"Storage '{storage_name}' not registered")
        return self.storages[storage_name][layer_id][physical_page_ids]

    def retrieve_all(
        self, layer_id: int, physical_page_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            storage_name: self.retrieve(layer_id, storage_name, physical_page_ids)
            for storage_name in self.storages.keys()
        }

    def get_writable_slots(
        self, layer_id: int, physical_page_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            storage_name: self.storages[storage_name][layer_id][physical_page_ids]
            for storage_name in self.storages.keys()
        }

    def get_layer_storage(self, layer_id: int, storage_name: str) -> torch.Tensor:
        if storage_name not in self.storages:
            raise ValueError(f"Storage '{storage_name}' not registered")
        return self.storages[storage_name][layer_id]

    def get_storage_names(self) -> list:
        return list(self.storages.keys())
