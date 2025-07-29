import psutil
import cuda.bindings.runtime as cuda_rt
import gc
from dataclasses import dataclass
import ctypes

import numpy as np
import base64
import logging
import os
import pickle
import time
from abc import ABC
from pathlib import Path
from typing import Callable, Generator, List, Any, Optional
from multiprocessing import shared_memory

import torch
from torch.func import functional_call

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.naive_distributed import NaiveDistributed
from sglang.srt.utils import get_int_env_var, is_pin_memory_available, MultiprocessingSerializer, get_bool_env_var, \
    dispose_tensor, check_cuda_result
import sys


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
        check_cuda_result(cuda_rt.cudaHostRegister(tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable))

        NaiveDistributed.instance.barrier()

        self._records.append(_SharedMemoryRecord(
            shm=shm,
            np_array=np_array,
            tensor=tensor,
        ))
        return tensor

@dataclass
class _SharedMemoryRecord:
    shm: shared_memory.SharedMemory
    np_array: np.ndarray
    tensor: torch.Tensor

_shared_memory_manager = _SharedMemoryManager()

