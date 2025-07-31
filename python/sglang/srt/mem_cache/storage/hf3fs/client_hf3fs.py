import logging
import multiprocessing
import os
import threading
from functools import wraps
from pathlib import Path
from typing import List

import torch
from torch.utils.cpp_extension import load

root = Path(__file__).parent.resolve()
hf3fs_utils = load(name="hf3fs_utils", sources=[f"{root}/hf3fs_utils.cpp"])

logger = logging.getLogger(__name__)

try:
    from hf3fs_fuse.io import (
        deregister_fd,
        extract_mount_point,
        make_ioring,
        make_iovec,
        register_fd,
    )
except ImportError as e:
    logger.warning(f"hf3fs_fuse.io is not available: {e}")


def rsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.rlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


def wsynchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.wlock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


class Hf3fsClient:
    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):
        self.path = path
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.entries = entries

        self.file = os.open(self.path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.file, size)
        register_fd(self.file)

        self.hf3fs_mount_point = extract_mount_point(path)
        self.bs = self.bytes_per_page
        self.shm_r = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )
        self.shm_w = multiprocessing.shared_memory.SharedMemory(
            size=self.bs * self.entries, create=True
        )

        self.shm_r_tensor = torch.frombuffer(self.shm_r.buf, dtype=torch.uint8)
        self.shm_w_tensor = torch.frombuffer(self.shm_w.buf, dtype=torch.uint8)

        self.numa = -1
        self.ior_r = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=True,
            timeout=1,
            numa=self.numa,
        )
        self.ior_w = make_ioring(
            self.hf3fs_mount_point,
            self.entries,
            for_read=False,
            timeout=1,
            numa=self.numa,
        )
        self.iov_r = make_iovec(self.shm_r, self.hf3fs_mount_point)
        self.iov_w = make_iovec(self.shm_w, self.hf3fs_mount_point)

        self.rlock = threading.RLock()
        self.wlock = threading.RLock()

    @rsynchronized()
    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)

        # prepare
        current = 0
        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize
            self.ior_r.prepare(
                self.iov_r[current : current + size], True, self.file, offset
            )
            current += size

        # submit
        ionum = len(offsets)
        resv = self.ior_r.submit().wait(min_results=ionum)

        # results
        hf3fs_utils.read_shm(self.shm_r_tensor, tensors)
        results = [res.result for res in resv]

        return results

    @wsynchronized()
    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)

        # prepare
        hf3fs_utils.write_shm(tensors, self.shm_w_tensor)
        current = 0
        for offset, tensor in zip(offsets, tensors):
            size = tensor.numel() * tensor.itemsize
            self.ior_w.prepare(
                self.iov_w[current : current + size], False, self.file, offset
            )
            current += size

        # submit
        ionum = len(offsets)
        resv = self.ior_w.submit().wait(min_results=ionum)

        # results
        results = [res.result for res in resv]

        return results

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:
        sizes = [t.numel() * t.itemsize for t in tensors]
        if any(
            [
                len(offsets) > self.entries,
                len(offsets) != len(sizes),
                all(
                    [
                        offset < 0 or offset + size > self.size
                        for offset, size in zip(offsets, sizes)
                    ]
                ),
                all([size > self.bytes_per_page for size in sizes]),
            ]
        ):
            self.close()
            raise ValueError(f"Hf3fsClient.check: {offsets=}, {sizes=}")

    def get_size(self) -> int:
        return self.size

    def close(self) -> None:
        deregister_fd(self.file)
        os.close(self.file)
        del self.ior_r
        del self.ior_w
        del self.iov_r
        del self.iov_w
        self.shm_r.close()
        self.shm_w.close()
        self.shm_r.unlink()
        self.shm_w.unlink()

    def flush(self) -> None:
        os.fsync(self.file)
