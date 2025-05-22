# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import signal
import tempfile
from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Tuple

import torch


class BaseConnector(ABC):
    pass


class BaseKVConnector(BaseConnector):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    @abstractmethod
    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_get(
        self, keys: List[str], target_locations: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(self, key, value) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        raise NotImplementedError()


class BaseWeightConnector(BaseConnector):
    """
    For fs connector such as s3, url looks like:
    <connector_type>://<path>/<filename>

    For kv connector such as redis, url looks like:
    <connector_type>://<host>:<port>/<model_name>/keys/<key>
    <connector_type://<host>:<port>/<model_name>/files/<filename>
    """

    def __init__(self, url: str):
        self.url = url
        self.closed = False
        self.local_dir = tempfile.mkdtemp()
        for sig in (signal.SIGINT, signal.SIGTERM):
            existing_handler = signal.getsignal(sig)
            signal.signal(sig, self._close_by_signal(existing_handler))

    def get_local_dir(self):
        return self.local_dir

    @abstractmethod
    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError()

    @abstractmethod
    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError()

    def close(self):
        if self.closed:
            return

        self.closed = True
        if os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def _close_by_signal(self, existing_handler=None):

        def new_handler(signum, frame):
            self.close()
            if existing_handler:
                existing_handler(signum, frame)

        return new_handler


class BaseFileSystemConnector(BaseConnector):
    """
    List full file names from remote fs path and filter by allow pattern.

    Args:
        allow_pattern: A list of patterns of which files to pull.

    Returns:
        list[str]: List of full paths allowed by the pattern
    """

    @abstractmethod
    def glob(self, allow_pattern: str) -> List[str]:
        raise NotImplementedError()
