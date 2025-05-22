# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import List, Optional
from urllib.parse import urlparse

import torch

from sglang.srt.connector import BaseKVConnector

logger = logging.getLogger(__name__)


class FileConnector(BaseKVConnector):

    def __init__(self, url: str):
        self.url = url
        parsed_url = urlparse(url)
        self.file_path = parsed_url.path
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        tensor_path = f"{self.file_path}/{key}.bin"
        try:
            # todo: fixing the target_location logic to enable in-place loading
            loaded_tensor = torch.load(tensor_path)
            if isinstance(loaded_tensor, torch.Tensor):
                return loaded_tensor
            else:
                logger.error(f"Loaded data for key {key} is not a tensor.")
                return None
        except FileNotFoundError:
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(self, key: str, value: torch.Tensor) -> bool:
        tensor_path = f"{self.file_path}/{key}.bin"
        if self.exists(key):
            logger.warning(f"Key {key} already exists. Skipped.")
            return True
        try:
            torch.save(value, tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False

    def exists(self, key: str) -> bool:
        tensor_path = f"{self.file_path}/{key}.bin"
        return os.path.exists(tensor_path)

    def list(self, prefix: str) -> List[str]:
        pass
