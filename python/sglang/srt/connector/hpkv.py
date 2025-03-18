# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Optional
from urllib.parse import urlparse

import hpkv
import torch

from sglang.srt.connector import BaseKVConnector
from sglang.srt.connector.utils import pull_files_from_db

logger = logging.getLogger(__name__)

METADATA_LENGTH = 16
MAX_TENSOR_DIMENSIONS = 14
METADATA_DTYPE = torch.int64
FLOAT16_INT = -543205003776624
INT64_INT = -375623078607432
BOOL_INT = -28035262008646
BFLOAT16_INT = -452084912267662
FLOAT32_INT = -1049557997456592
FLOAT64_INT = -452201007054137
FLOAT8_E4M3FN_INT = -1066697177659525
FLOAT8_E5M2_INT = -618182574682355
INT8_INT = -4012536253571955326
DTYPE2INT = {
    torch.float16: FLOAT16_INT,
    torch.int64: INT64_INT,
    torch.bool: BOOL_INT,
    torch.bfloat16: BFLOAT16_INT,
    torch.float32: FLOAT32_INT,
    torch.float64: FLOAT64_INT,
    torch.float8_e4m3fn: FLOAT8_E4M3FN_INT,
    torch.float8_e5m2: FLOAT8_E5M2_INT,
    torch.int8: INT8_INT,
}
INT2DTYPE = {
    FLOAT16_INT: torch.float16,
    INT64_INT: torch.int64,
    BOOL_INT: torch.bool,
    BFLOAT16_INT: torch.bfloat16,
    FLOAT32_INT: torch.float32,
    FLOAT64_INT: torch.float64,
    FLOAT8_E4M3FN_INT: torch.float8_e4m3fn,
    FLOAT8_E5M2_INT: torch.float8_e5m2,
    INT8_INT: torch.int8,
}


def _parse_metadata(d_metadata_buffer: torch.Tensor, device="cpu") -> torch.Tensor:
    h_buffer = d_metadata_buffer.cpu().numpy()
    dtype = INT2DTYPE[h_buffer[0]]
    ndims = h_buffer[1]
    shape = tuple(h_buffer[2 : 2 + ndims])
    return torch.empty(shape, dtype=dtype, device=device)


def _make_metadata(tensor: torch.Tensor) -> torch.Tensor:
    """Create the metadata on based on the input tensor, and move it to GPU.
    The metadata's length is `TorchDistributedPipe.METADATA_LENGTH`.

    Currently, the metadata is a int64 tensor and it includes dtype, number
    of dimensions, and the shape information of the input tensor.


    The information follows the layout below:
    - metadata[0] -- dtype
    - metadata[1] -- number of dimensions
    - metadata[2 : 2+ndims] -- the shape of the input tensor

    Parameters:
        - tensor: the input tensor

    Returns:
        - metadata: the metadata tensor, on self.device
    """

    buffer = torch.empty(METADATA_LENGTH, dtype=METADATA_DTYPE, device="cpu")
    buffer[0] = DTYPE2INT[tensor.dtype]
    ndims = len(tensor.shape)
    buffer[1] = len(tensor.shape)
    buffer[2 : 2 + ndims] = torch.tensor(tensor.shape, dtype=METADATA_DTYPE)
    return buffer


class HPKVConnector(BaseKVConnector):

    def __init__(self, url: str, device: torch.device = "cpu"):
        super().__init__(url)
        parsed_url = urlparse(url)
        self.device = device
        self.connection = hpkv.HPKVTensorClient(
            parsed_url.hostname, parsed_url.port, None, 0
        )
        self.model_name = parsed_url.path.lstrip("/")

    def get(self, key: str) -> Optional[torch.Tensor]:
        meta_key = key + "/metadata"
        meta = torch.empty(METADATA_LENGTH, dtype=METADATA_DTYPE, device="cpu")
        ret = self.connection.get(meta_key, meta)
        if ret != 0:
            logger.error("Failed to get metadata for key %s, ret = %d", meta_key, ret)
            return None

        obj = _parse_metadata(meta, device=self.device)
        ret = self.connection.get(key, obj)
        if ret != 0:
            logger.error("Failed to get tensor for key %s, ret = %d", key, ret)

        return obj

    def getstr(self, key):
        ret = self.connection.getstr(key)
        if ret is None:
            logger.error("Key %s not found", key)
            return None

        return ret

    def set(self, key: str, tensor: torch.Tensor) -> None:
        meta_key = key + "/metadata"
        meta = _make_metadata(tensor)
        ret = self.connection.set(meta_key, meta)
        if ret != 0:
            raise ValueError(f"Failed to set metadata for key {meta_key}, ret = {ret}")

        ret = self.connection.set(key, tensor)
        if ret != 0:
            raise ValueError(f"Failed to set tensor for key {key}, ret = {ret}")

    def setstr(self, key, obj):
        ret = self.connection.setstr(key, obj)
        if ret != 0:
            raise ValueError(f"Failed to set str for key {key}, ret = {ret}")

    def list(self, prefix: str) -> list[str]:
        return self.connection.keys(f"{prefix}.*")

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        keys = self.list(f"{self.model_name}/keys/rank_{rank}/")
        for key in keys:
            if not key.endswith("/metadata"):
                tensor = self.get(key)
                key = key.removeprefix(f"{self.model_name}/keys/rank_{rank}/")
                yield key, tensor

    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        pull_files_from_db(self, self.model_name, allow_pattern, ignore_pattern)

    def close(self):
        self.connection.close()
        super().close()
