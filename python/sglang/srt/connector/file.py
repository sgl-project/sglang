# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Generator, List, Optional, Tuple

import torch

from sglang.srt.connector import BaseKVConnector
from sglang.srt.connector.serde import create_serde

logger = logging.getLogger(__name__)


class FileConnector(BaseKVConnector):

    def __init__(self, url: str):

        super().__init__(url)
        self.path = url.split("://")[1]
        logger.info(f"saving file to {self.path}")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # TODO: more serde options
        self.s, self.d = create_serde("safe")

    def get(self, key: str) -> Optional[torch.Tensor]:
        if not os.path.exists(os.path.join(self.path, key)):
            return None

        with open(os.path.join(self.path, key), "rb") as f:
            return self.d.from_bytes(f.read())

    def getstr(self, key: str) -> Optional[str]:
        if not os.path.exists(os.path.join(self.path, key)):
            return None

        with open(os.path.join(self.path, key), "rb") as f:
            return f.read().decode("utf-8")

    def set(self, key: str, tensor: torch.Tensor) -> None:
        assert tensor is not None
        logger.info("set key %s to %s", key, os.path.join(self.path, key))
        with open(os.path.join(self.path, key), "wb") as f:
            f.write(self.s.to_bytes(tensor))

    def setstr(self, key: str, obj: str) -> None:
        with open(os.path.join(self.path, key), "wb") as f:
            f.write(obj.encode("utf-8"))

    def list(self, prefix: str) -> List[str]:
        return os.listdir(self.path)

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
