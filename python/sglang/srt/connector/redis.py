# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlparse

import torch

from sglang.srt.connector import BaseKVConnector, BaseWeightConnector
from sglang.srt.connector.serde import create_serde
from sglang.srt.connector.utils import pull_files_from_db

logger = logging.getLogger(__name__)


class RedisConnector(BaseWeightConnector, BaseKVConnector):

    def __init__(self, url: str):
        import redis

        super().__init__(url)
        parsed_url = urlparse(url)
        self.connection = redis.Redis(host=parsed_url.hostname, port=parsed_url.port)
        self.model_name = parsed_url.path.lstrip("/")
        # TODO: more serde options
        self.s, self.d = create_serde("safe")

    def get(
        self, key: str, target: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        val = self.connection.get(key)

        if val is None:
            logger.error("Key %s not found", key)
            return None

        return self.d.from_bytes(val)

    def set(self, key: str, val) -> bool:
        assert val is not None
        if isinstance(val, str):
            self.connection.set(key, val)
            return True
        elif isinstance(val, torch.Tensor):
            self.connection.set(key, self.s.to_bytes(val))
            return True
        else:
            logger.error(f"unsupported Redis set type {type(val)}")
            return False

    def exists(self, key: str) -> bool:
        return self.connection.exists(key)

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
        def getstr(self, file):
            return self.connection.get(file)

        pull_files_from_db(self, self.model_name, getstr, allow_pattern, ignore_pattern)

    def close(self):
        self.connection.close()
        super().close()
