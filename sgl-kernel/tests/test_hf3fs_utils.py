import multiprocessing.shared_memory

import pytest
import torch
from sgl_kernel.hf3fs_utils import read_shm, write_shm


def test_rw_shm():
    numel = 8 << 20
    dtype = torch.bfloat16
    page_num = 128
    page_bytes = numel * dtype.itemsize
    shm = multiprocessing.shared_memory.SharedMemory(
        size=page_num * page_bytes, create=True
    )
    tshm = torch.frombuffer(shm.buf, dtype=torch.uint8)
    a = [torch.randn(numel, dtype=dtype) for _ in range(page_num)]
    b = [torch.empty(numel, dtype=dtype) for _ in range(page_num)]
    write_shm(a, tshm, page_bytes)
    read_shm(tshm, b, page_bytes)
    for _a, _b in zip(a, b):
        torch.testing.assert_close(_a, _b)

    del tshm
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
