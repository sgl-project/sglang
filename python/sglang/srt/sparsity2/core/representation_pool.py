import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class RepresentationPool:
    """
    Multi-slot representation storage pool.
    Supports algorithms registering multiple named storage slots with different shapes/dtypes.
    """

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
        self.storages: Dict[str, list] = {}

    def register_storage(self, name: str, shape: tuple, dtype: torch.dtype):
        """Register a storage slot for algorithm."""
        if name in self.storages:
            logger.warning(f"Storage '{name}' already registered, skipping")
            return

        layer_pools = {
            layer_id: torch.zeros(
                (self.total_num_pages, *shape), dtype=dtype, device=self.device
            )
            for layer_id in range(self.start_layer, self.end_layer)
        }
        self.storages[name] = layer_pools
        logger.info(
            f"Registered storage '{name}' with shape {shape}, dtype {dtype}, "
            f"{self.end_layer - self.start_layer} layers"
        )

    def store(
        self,
        layer_id: int,
        storage_name: str,
        physical_page_ids: torch.Tensor,
        data: torch.Tensor,
    ):
        """Store data to specific storage slot."""
        if storage_name not in self.storages:
            raise ValueError(f"Storage '{storage_name}' not registered")

        if data.dtype != self.storages[storage_name][layer_id].dtype:
            data = data.to(dtype=self.storages[storage_name][layer_id].dtype)

        data = data.to(device=self.device)
        self.storages[storage_name][layer_id][physical_page_ids] = data

    def retrieve(
        self, layer_id: int, storage_name: str, physical_page_ids: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve data from specific storage slot."""
        if storage_name not in self.storages:
            raise ValueError(f"Storage '{storage_name}' not registered")

        return self.storages[storage_name][layer_id][physical_page_ids]

    def retrieve_all(
        self, layer_id: int, physical_page_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Retrieve all storage slots for given pages."""
        result = {}
        for storage_name in self.storages.keys():
            result[storage_name] = self.retrieve(
                layer_id, storage_name, physical_page_ids
            )
        return result

    def get_writable_slots(
        self, layer_id: int, physical_page_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get writable tensor slots for zero-copy writing.
        Note: When physical_page_ids is a tensor with multiple indices, PyTorch's
        fancy indexing returns a COPY, not a view. For true zero-copy, use scalar indexing.
        """
        result = {}
        for storage_name in self.storages.keys():
            result[storage_name] = self.storages[storage_name][layer_id][
                physical_page_ids
            ]
        return result

    def get_layer_storage(self, layer_id: int, storage_name: str) -> torch.Tensor:
        """
        Get the full storage tensor for a layer. Algorithm can directly index into it.
        Single-element indexing (e.g., storage[idx]) returns a view for zero-copy writes.
        """
        if storage_name not in self.storages:
            raise ValueError(f"Storage '{storage_name}' not registered")
        return self.storages[storage_name][layer_id]

    def get_storage_names(self) -> list:
        """Get all registered storage names."""
        return list(self.storages.keys())
