import logging
import os
import urllib.parse
from typing import Any, List, Optional

import pygd2fs
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


class GD2FSStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig):
        if storage_config is None:
            raise ValueError("not found GD2FS storage configuration")

        extra_config = getattr(storage_config, "extra_config", None)
        if extra_config is None:
            raise ValueError("not found GD2FS configuration")

        self.cpaddr = extra_config.get("cpaddr", None)
        if self.cpaddr is None:
            raise ValueError("not found GD2FS cpaddr")

        self.dpaddr = extra_config.get("dpaddr", None)
        if self.dpaddr is None:
            raise ValueError("not found GD2FS dpaddr")

        self.cluster = extra_config.get("cluster", None)
        if self.cluster is None:
            raise ValueError("not found GD2FS cluster")

        self.iothreads = extra_config.get("iothreads", 1)
        self.memthreads = extra_config.get("memthreads", 1)
        self.streams = extra_config.get("streams", 1)

        self.client = pygd2fs.Client(
            self.cpaddr,
            self.dpaddr,
            self.cluster,
            iothreads=self.iothreads,
            memthreads=self.memthreads,
            streams=self.streams,
        )
        if self.client is None:
            raise RuntimeError("cannot connect to GD2FS cluster")

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[Optional[dict], str]:
        if not uri.startswith("gd2fs://"):
            return (None, uri)

        parsed = urllib.parse.urlparse(uri)

        addr_list = [a.strip() for a in parsed.netloc.split(",") if a.strip()]
        cpaddr = ",".join(f"gd2fs://{a}" for a in addr_list)

        dpaddr = os.getenv("GD2FS_DPADDR", "tcp://127.0.0.1")
        iothreads = int(os.getenv("GD2FS_IOTHREADS", 1))
        memthreads = int(os.getenv("GD2FS_MEMTHREADS", 1))
        streams = int(os.getenv("GD2FS_STREAMS", 1))

        query = {
            k: v[0] if v else "" for k, v in urllib.parse.parse_qs(parsed.query).items()
        }

        return {
            "cpaddr": cpaddr,
            "dpaddr": dpaddr,
            "iothreads": iothreads,
            "memthreads": memthreads,
            "streams": streams,
            **query,
        }, parsed.path

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        self.mem_pool_host = mem_pool_host
        # kv_buffer = self.mem_pool_host.kv_buffer
        # self.gd2fs_iomem = self.client.RegIOMEM(
        #     kv_buffer.data_ptr(), kv_buffer.numel() * kv_buffer.element_size()
        # )
        # if self.gd2fs_iomem is None:
        #     raise RuntimeError("cannot register memory to GD2FS client")

    def _wait_requests(self, reqs: list[pygd2fs.Request]):
        for req in reqs:
            if req is None:
                raise RuntimeError("GD2FS request is None")

        while len(reqs) > 0:
            rs = self.client.Wait(-1)
            for r in rs:
                if r.Status() != 0:
                    raise RuntimeError(f"GD2FS request failed, {r}")
                reqs.remove(r)

    def _wait_request(self, req: pygd2fs.Request) -> tuple[int, int]:
        if req is None:
            raise RuntimeError("GD2FS request is None")

        reqs = self.client.Wait(-1)
        if len(reqs) > 1:
            raise RuntimeError(f"wait {len(reqs)} requests, expect 1")

        if reqs[0] != req:
            raise RuntimeError(f"wait wrong request {reqs[0]}, expect {req}")

        return req.Status(), req.Value()

    def load(self, filepath: str, offset: int, data: torch.Tensor) -> None:
        address = data.data_ptr()
        length = data.element_size() * data.numel()

        iomem = self.client.RegIOMEM(address, length)
        if iomem is None:
            raise RuntimeError("register memory error")

        try:
            sge = pygd2fs.SGE(address, length, iomem)
            req = self.client.Read(filepath, offset, [sge], 0, "")
            if req is None:
                raise RuntimeError("GD2FS read error")

            status, value = self._wait_request(req)
            if status != 0 or value != length:
                raise RuntimeError(f"GD2FS request failed, {req}")

        finally:
            self.client.DeregIOMEM(iomem)

    def save(self, filepath: str, offset: int, data: torch.Tensor) -> None:
        if not data.is_contiguous():
            data = data.clone()

        address = data.data_ptr()
        length = data.element_size() * data.numel()
        iomem = self.client.RegIOMEM(address, length)
        if iomem is None:
            raise RuntimeError("register memory error")

        try:
            sge = pygd2fs.SGE(address, length, iomem)
            req = self.client.Write(filepath, offset, [sge], 0, "")
            if req is None:
                raise RuntimeError("GD2FS write error")

            status, value = self._wait_request(req)
            if status != 0 or value != length:
                raise RuntimeError(f"GD2FS request failed, {req}")

        finally:
            self.client.DeregIOMEM(iomem)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError
