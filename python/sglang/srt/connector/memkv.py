# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import torch

from sglang.srt.connector import BaseKVConnector
from sglang.srt.utils import convert_to_bytes

logger = logging.getLogger(__name__)


class MemKVConnector(BaseKVConnector):
    def __init__(self, url: str, **kargs):
        super().__init__(url)

        # url: memkv://0?size=20GB
        # 0: the index of the memkv connector. It's possible to add another
        #    memkv for other purpose, or extend the total size of memkv in the future.
        # size: the size of memory used, 20KB, 20MB, 20GB or 20 TB.
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        size_str = query_params.get("size", [None])[0]

        if size_str is None:
            raise RuntimeError("failed to parse memory size")

        self.size = convert_to_bytes(size_str)
        self.total_size = 0
        self.tensor_cache = {}
        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    def get(self, key: str, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if key not in self.tensor_cache:
            return

        with torch.cuda.stream(self.write_stream):
            cpu_val = self.tensor_cache[key]
            tensor.copy_(cpu_val, non_blocking=False)

    def set(self, key: str, tensor: torch.Tensor) -> None:
        if self.total_size >= self.size:
            logger.error("mem_kv space is full.")
            return -1

        with torch.cuda.stream(self.write_stream):
            cpu_val = tensor.to(device="cpu", non_blocking=False)
            self.tensor_cache[key] = cpu_val
            self.total_size += cpu_val.element_size() * cpu_val.numel()
            return 0

    def close(self):
        del self.tensor_cache

    def getstr(self, key: str) -> Optional[str]:
        raise NotImplementedError()

    def setstr(self, key: str, obj: str) -> None:
        raise NotImplementedError()

    def list(self, prefix: str) -> List[str]:
        raise NotImplementedError()

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, bytes], None, None]:
        raise NotImplementedError()

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError()
