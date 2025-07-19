import multiprocessing.shared_memory

import pytest
import torch
from sgl_kernel.hf3fs_utils import read_shm, write_shm
from tqdm import tqdm


def test_rw_shm():
    numel = 8 << 20
    dtype = torch.bfloat16
    page_num = 128
    page_bytes = numel * dtype.itemsize
    shm = multiprocessing.shared_memory.SharedMemory(
        size=page_num * page_bytes, create=True
    )
    tshm = torch.frombuffer(shm.buf, dtype=torch.uint8)
    a = [
        torch.randn(numel, dtype=dtype)
        for _ in tqdm(range(page_num), desc="prepare input")
    ]
    b = [
        torch.empty(numel, dtype=dtype)
        for _ in tqdm(range(page_num), desc="prepare output")
    ]
    write_shm(a, tshm)
    read_shm(tshm, b)
    for _a, _b in tqdm(zip(a, b), desc="assert_close"):
        torch.testing.assert_close(_a, _b)

    del tshm
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
