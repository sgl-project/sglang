import logging
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.distributed.naive_distributed import get_naive_distributed
from sglang.srt.utils import check_cuda_result

logger = logging.getLogger(__name__)


class HostSharedMemoryManager:
    def __init__(self, base_name: str):
        self._base_name = Path(base_name)
        self._operation_index = 0
        self._records: List[_Record] = []

    def malloc(self, *, shape, dtype):
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self._malloc_raw(num_bytes=meta_tensor.nbytes)
        return raw.view(dtype).view(*shape)

    def _malloc_raw(self, *, num_bytes: int) -> torch.Tensor:
        import cuda.bindings.runtime as cuda_rt

        self._operation_index += 1
        shm_name = f"{self._base_name}_op{self._operation_index}"

        # TODO handle dispose
        if get_naive_distributed().get_rank() == 0:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)

        get_naive_distributed().barrier()

        if get_naive_distributed().get_rank() != 0:
            shm = shared_memory.SharedMemory(name=shm_name)

        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)
        tensor = torch.from_numpy(np_array)

        check_cuda_result(
            cuda_rt.cudaHostRegister(
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )

        get_naive_distributed().barrier()

        self._records.append(
            _Record(
                shm=shm,
                np_array=np_array,
                tensor=tensor,
            )
        )
        return tensor


@dataclass
class _Record:
    shm: shared_memory.SharedMemory
    np_array: np.ndarray
    tensor: torch.Tensor


# Can have multi instances if needed
_instance: Optional[HostSharedMemoryManager] = None


def get_host_shared_memory_manager():
    assert _instance is not None
    return _instance


def set_host_shared_memory_manager(instance: HostSharedMemoryManager):
    global _instance
    assert _instance is None
    _instance = instance
