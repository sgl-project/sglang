# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlparse

import torch

from sglang.srt.connector import BaseKVConnector
from sglang.srt.connector.serde import create_serde
from sglang.srt.connector.utils import pull_files_from_db

logger = logging.getLogger(__name__)


class RedisConnector(BaseKVConnector):

    def __init__(self, url: str):
        import redis

        super().__init__(url)
        parsed_url = urlparse(url)
        self.connection = redis.Redis(host=parsed_url.hostname, port=parsed_url.port)
        self.model_name = parsed_url.path.lstrip("/")
        # TODO: more serde options
        self.s, self.d = create_serde("safe")

    def get(self, key: str) -> Optional[torch.Tensor]:
        val = self.connection.get(key)

        if val is None:
            logger.error("Key %s not found", key)
            return None

        return self.d.from_bytes(val)

    def getstr(self, key: str) -> Optional[str]:
        val = self.connection.get(key)
        if val is None:
            logger.error("Key %s not found", key)
            return None

        return val.decode("utf-8")

    def set(self, key: str, tensor: torch.Tensor) -> None:
        assert tensor is not None
        self.connection.set(key, self.s.to_bytes(tensor))

    def setstr(self, key: str, obj: str) -> None:
        self.connection.set(key, obj)

    def list(self, prefix: str) -> List[str]:
        cursor = 0
        all_keys: List[bytes] = []

        while True:
            ret: Tuple[int, List[bytes]] = self.connection.scan(
                cursor=cursor, match=f"{prefix}*"
            )  # type: ignore
            cursor, keys = ret
            all_keys.extend(keys)
            if cursor == 0:
                break

        return [key.decode("utf-8") for key in all_keys]

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, bytes], None, None]:
        keys = self.list(f"{self.model_name}/keys/rank_{rank}/")
        for key in keys:
            val = self.get(key)
            key = key.removeprefix(f"{self.model_name}/keys/rank_{rank}/")
            yield key, val

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        pull_files_from_db(self, self.model_name, allow_pattern, ignore_pattern)

    def close(self):
        self.connection.close()
        super().close()
