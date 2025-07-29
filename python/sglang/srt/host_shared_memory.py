import logging
import os
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import List

import cuda.bindings.runtime as cuda_rt
import numpy as np
import torch

from sglang.srt.naive_distributed import NaiveDistributed
from sglang.srt.utils import (
    check_cuda_result,
)

logger = logging.getLogger(__name__)

class _HostSharedMemoryManager:
    def __init__(self):
        self._base_name = Path(os.environ["SGLANG_SHARED_MEMORY_MANAGER_BASE_NAME"])
        self._operation_index = 0
        self._records: List[_Record] = []

    def malloc(self, *, shape, dtype):
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self._malloc_raw(num_bytes=meta_tensor.nbytes)
        return raw.view(dtype).view(*shape)

    def _malloc_raw(self, *, num_bytes: int) -> torch.Tensor:
        self._operation_index += 1
        shm_name = f"{self._base_name}_op{self._operation_index}"

        # TODO handle dispose
        if NaiveDistributed.instance.get_rank() == 0:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)

        NaiveDistributed.instance.barrier()

        if NaiveDistributed.instance.get_rank() != 0:
            shm = shared_memory.SharedMemory(name=shm_name)

        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)
        tensor = torch.from_numpy(np_array)

        logger.info(f"cudaHostRegister({tensor.data_ptr()=})")
        check_cuda_result(
            cuda_rt.cudaHostRegister(
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )

        NaiveDistributed.instance.barrier()

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


_shared_memory_manager = _HostSharedMemoryManager()
