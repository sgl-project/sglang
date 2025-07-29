import base64
import ctypes
import gc
import logging
import os
import pickle
import sys
import time
from abc import ABC
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional

import cuda.bindings.runtime as cuda_rt
import numpy as np
import psutil
import torch
from torch.func import functional_call

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.naive_distributed import NaiveDistributed
from sglang.srt.utils import (
    MultiprocessingSerializer,
    check_cuda_result,
    dispose_tensor,
    get_bool_env_var,
    get_int_env_var,
    is_pin_memory_available,
)


class _SharedMemoryManager:
    def __init__(self):
        self._base_name = Path(os.environ["SGLANG_SHARED_MEMORY_MANAGER_BASE_NAME"])
        self._operation_index = 0
        self._records: List[_SharedMemoryRecord] = []

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
            _SharedMemoryRecord(
                shm=shm,
                np_array=np_array,
                tensor=tensor,
            )
        )
        return tensor


@dataclass
class _SharedMemoryRecord:
    shm: shared_memory.SharedMemory
    np_array: np.ndarray
    tensor: torch.Tensor


_shared_memory_manager = _SharedMemoryManager()
