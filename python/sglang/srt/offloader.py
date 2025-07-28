import psutil
import gc
from dataclasses import dataclass
import ctypes


import cuda.bindings.runtime as cuda_rt
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
from sglang.srt.utils import get_int_env_var, is_pin_memory_available, MultiprocessingSerializer, get_bool_env_var, \
    dispose_tensor

logger = logging.getLogger(__name__)


# TODO improve
class ModuleOffloader:
    def __init__(self):
        self.group_size = get_int_env_var("SGLANG_OFFLOAD_GROUP_SIZE", -1)
        self.num_offload_in_group = get_int_env_var("SGLANG_OFFLOAD_NUM_OFFLOAD_IN_GROUP", 1)
        self.prefetch_step = get_int_env_var("SGLANG_OFFLOAD_PREFETCH_STEP", 1)
        self.mode = os.environ.get("SGLANG_OFFLOAD_MODE", "cpu")
        self.enabled = self.group_size > 0

        if self.mode in {"sharded_gpu", "shm_cpu"}:
            NaiveDistributed.initialize(
                rank=global_server_args_dict["dp_rank"],
                world_size=global_server_args_dict["dp_size"],
            )

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Callable[[torch.nn.Module], torch.nn.Module],
        whitelist_param_names_creator: Callable[[torch.nn.Module], List[str]],
    ):
        if not self.enabled:
            return list(all_modules_generator)

        logger.info(f"[offloader] {self.group_size=} {self.num_offload_in_group=} {self.prefetch_step=}")

        alt_stream = torch.cuda.Stream()

        # TODO maybe improve
        all_modules = []
        offload_submodules = []
        self.offloaders = []
        for module_index, module in enumerate(all_modules_generator):
            logger.info(
                f"[offloader] {module_index=} {torch.cuda.memory_allocated()=}"
            )
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_offload_in_group:
                submodule = submodule_accessor(module)
                whitelist_param_names = whitelist_param_names_creator(submodule)
                logger.info(
                    f"[offloader] offload {module_index=} submodule={type(submodule)} params={whitelist_param_names}"
                )
                offload_submodules.append(submodule)
                self.offloaders.append(_ModuleOffloader(
                    mode=self.mode, module=submodule, alt_stream=alt_stream,
                    whitelist_param_names=whitelist_param_names,
                ))

        for index, module in enumerate(offload_submodules):
            _hook_module_forward_for_offloader(
                index=index, module=module, offloaders=self.offloaders,
                prefetch_step=self.prefetch_step,
            )

        return all_modules

    def post_init(self):
        if not self.enabled:
            return

        for offloader in self.offloaders:
            offloader.post_init()

        for i in range(self.prefetch_step):
            self.offloaders[i].start_onload()

        if get_bool_env_var("SGLANG_HACK_MEM_PROFILE_STARTUP"):
            print("start save mem profile")
            memory_profile_path = os.path.join(
                "/data/numa0/tom/temp_sglang_server2local",
                str(time.time())
                + f"-DP-{NaiveDistributed.instance.get_rank()}-memory"
                + ".pickle",
                )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print("end save mem profile")


def _hook_module_forward_for_offloader(index, module, offloaders, prefetch_step):
    def _on_forward_end():
        offloaders[(index + prefetch_step) % len(offloaders)].start_onload()
        offloaders[index].offload()

    _hook_module_forward_raw(
        module,
        on_forward_end=_on_forward_end,
        get_parameter_and_buffer_dicts=lambda: offloaders[
            index
        ].wait_and_get_device_tensors(),
    )


def _hook_module_forward_raw(
    module, on_forward_end, get_parameter_and_buffer_dicts
):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        output = functional_call(
            module, get_parameter_and_buffer_dicts(), args=args, kwargs=kwargs
        )
        on_forward_end()
        module.forward = forward
        return output

    module.forward = forward


class _ModuleOffloader(ABC):
    def __init__(self, mode: str, module: torch.nn.Module, alt_stream: torch.cuda.Stream, whitelist_param_names: List[str]):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream

        assert self.device != torch.device(
            "cpu"
        ), "not handled device=cpu case yet (should skip this tensor)"

        self._device_tensors = None
        self._load_event = None

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), f"{whitelist_param_names=} {list(param_dict.keys())=}"

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        for name, param_offloader in self._param_offloaders.items():
            param_offloader.post_init()

    def start_onload(self):
        self.alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.alt_stream):
            self._device_tensors = self._create_device_tensors()
            self._load_event = torch.cuda.Event()
            self._load_event.record()

    def offload(self):
        self._device_tensors = None
        self._load_event = None

    def wait_and_get_device_tensors(self):
        assert self._device_tensors is not None
        self._load_event.wait()
        return self._device_tensors

    def _create_device_tensors(self):
        return {
            k: v.create_device_tensor()
            for k, v in self._param_offloaders.items()
        }


class _BaseParamOffloader(ABC):
    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        return {
            "meta": _MetaParamOffloader,
            "cpu": _CpuParamOffloader,
            "shm_cpu": _ShmCpuParamOffloader,
            "sharded_gpu": _ShardedGpuParamOffloader,
        }[mode](**kwargs)

    def __init__(self, module, param_name):
        self._module = module
        self._param_name = param_name

    @property
    def _param(self):
        return getattr(self._module, self._param_name)

    def post_init(self):
        pass

    def create_device_tensor(self):
        raise NotImplementedError

class _MetaParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        _move_param_to_meta(module, param_name)

    def create_device_tensor(self):
        return torch.empty_like(self._param.data, device="cuda")


class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        _move_param_to_cpu(self._param, pin_memory=True)

    def create_device_tensor(self):
        return self._param.to("cuda", non_blocking=True)

class _ShmCpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._rank = NaiveDistributed.instance.get_rank()
        self._world_size = NaiveDistributed.instance.get_world_size()

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"
        assert self._param.data.is_contiguous(), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        if self._rank == 0:
            _move_param_to_cpu(self._param, pin_memory=False)
        else:
            _move_param_to_meta(self._module, self._param_name)

        self.shm_cpu_data: Optional[torch.Tensor] = None

    def post_init(self):
        # logger.info(f"hack post_init: only do move to meta {self._param_name} {psutil.Process().memory_info().rss=}")
        # _move_param_to_meta(self._module, self._param_name)

        # check again since it may be changed
        assert self._param.data.is_contiguous(), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        assert self._param.is_contiguous()
        self.shm_cpu_data = _shared_memory_manager.malloc(shape=self._param.shape, dtype=self._param.dtype)

        if self._rank == 0:
            self.shm_cpu_data.copy_(self._param.data.to("cpu"))
        NaiveDistributed.instance.barrier()

        _move_param_to_meta(self._module, self._param_name)


    def create_device_tensor(self):
        return self.shm_cpu_data.to("cuda", non_blocking=True)

# TODO unify with ShmCpu mode
class _ShardedGpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module, param_name):
        super().__init__(module, param_name)
        self._rank = NaiveDistributed.instance.get_rank()
        self._world_size = NaiveDistributed.instance.get_world_size()

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"
        assert self._param.data.is_contiguous(), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        if self._rank == 0:
            _move_param_to_cpu(self._param, pin_memory=True)
        else:
            _move_param_to_meta(self._module, self._param_name)

        self.sharded_param_handles = None

    def post_init(self):
        # check again since it may be changed
        assert self._param.data.is_contiguous(), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        scatter_src = self._param.data

        logger.info(f"[offloader] post_init {scatter_src.nbytes=} {scatter_src.dtype=} {scatter_src.shape=} {torch.cuda.memory_allocated()=}")

        if self._rank == 0:
            scatter_src = scatter_src.to("cuda")
        scatter_list = _even_chunk(scatter_src, self._world_size)

        sharded_param = torch.empty(scatter_list[0].shape, dtype=scatter_list[0].dtype, device="cuda")
        self.sharded_param_handles = _create_shared_buffer_tensors(local_tensor=sharded_param)

        NaiveDistributed.instance.scatter(sharded_param, scatter_list if self._rank == 0 else None)

        _move_param_to_meta(self._module, self._param_name)

    def create_device_tensor(self):
        output = _empty_strided_like(self._param, device="cuda")
        output_chunks = output.chunk(self._world_size)

        for index in range(self._world_size):
            src_rank = (self._rank + index) % self._world_size
            src_buf = self.sharded_param_handles[src_rank]
            output_chunks[src_rank].copy_(src_buf)

        return output

def _even_chunk(x: torch.Tensor, chunks: int):
    assert x.shape[0] % chunks == 0, f"{x.shape=} {chunks=}"
    return list(x.chunk(chunks))

def _move_param_to_cpu(param, pin_memory: bool):
    cpu_data = _empty_strided_like(
        param.data,
        device="cpu",
        pin_memory=pin_memory,
    )
    cpu_data.copy_(param.data)
    param.data = cpu_data

def _move_param_to_meta(module, param_name):
    old_param = getattr(module, param_name)
    old_param_type = type(old_param)

    new_data = old_param.data.to("meta")

    # TODO improve
    if old_param_type == ModelWeightParameter:
        # manually checked how `w13_weight` and `w2_weight` are constructed
        new_param = ModelWeightParameter(
            data=new_data,
            **{k: getattr(old_param, k) for k in ["input_dim", "output_dim", "weight_loader"]}
        )
    elif old_param_type == torch.nn.Parameter:
        new_param = torch.nn.Parameter(
            data=new_data,
            requires_grad=False,
        )
    else:
        raise ValueError(f"Unknown {old_param_type=} {old_param=}")

    setattr(module, param_name, new_param)

    logger.info(f"hi move_param_to_meta {old_param.device=} {old_param.dtype=} {old_param.shape=}")
    dispose_tensor(old_param)

    # TODO do not call it *per* param
    gc.collect()
    trim_memory()

def _empty_strided_like(x: torch.Tensor, device, pin_memory=False):
    return torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        layout=x.layout,
        device=device,
        pin_memory=pin_memory,
    )

def _create_shared_buffer_tensors(local_tensor: torch.Tensor) -> List[torch.Tensor]:
    self_rank = NaiveDistributed.instance.get_rank()
    world_size = NaiveDistributed.instance.get_world_size()

    object_list = NaiveDistributed.instance.all_gather_object(
        dict(
            dup_serialized_local_tensor=[
                None
                if interesting_rank == self_rank
                else MultiprocessingSerializer.serialize(local_tensor)
                for interesting_rank in range(world_size)
            ]
        )
    )

    output_tensors = []
    for output_rank in range(world_size):
        remote_serialized_tensor = object_list[output_rank]["dup_serialized_local_tensor"][self_rank]
        if output_rank == self_rank:
            assert remote_serialized_tensor is None
            output_tensors.append(local_tensor)
        else:
            output_tensors.append(MultiprocessingSerializer.deserialize(remote_serialized_tensor))

    return output_tensors


class NaiveDistributed:
    instance: Optional["NaiveDistributed"] = None

    @staticmethod
    def initialize(**kwargs):
        assert NaiveDistributed.instance is None
        NaiveDistributed.instance = NaiveDistributed(**kwargs)

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size
        self._operation_index = 0
        self._directory = Path(os.environ["SGLANG_NAIVE_DISTRIBUTED_DIRECTORY"])
        self._directory.mkdir(parents=True, exist_ok=True)
        assert 0 <= rank < world_size

        # both barrier to be safe, and as a sanity check
        self.barrier()

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def scatter(self, tensor: torch.Tensor, scatter_list: List[torch.Tensor], src: int = 0):
        if self._rank == src:
            assert len(scatter_list) == self._world_size
        else:
            assert scatter_list is None

        gathered_objects = self.all_gather_object(
            dict(serialized_scatter_list=[
                None
                if item_rank == src
                else MultiprocessingSerializer.serialize(item)
                for item_rank, item in enumerate(scatter_list)
            ])
            if self._rank == src
            else dict()
        )

        remote_serialized_tensor = gathered_objects[src]["serialized_scatter_list"][self._rank]
        if self._rank == src:
            assert remote_serialized_tensor is None
            remote_tensor = scatter_list[self._rank]
        else:
            remote_tensor = MultiprocessingSerializer.deserialize(remote_serialized_tensor)
        tensor.copy_(remote_tensor)

        # avoid src tensor be deleted too early
        self.barrier()

    def all_gather_object(self, obj: Any) -> List[Any]:
        self._operation_index += 1

        text_postfix = "\n"

        def _get_path(interesting_rank: int):
            return self._directory / f"rank{interesting_rank}_op{self._operation_index}.txt"

        _get_path(self._rank).write_text(base64.b64encode(pickle.dumps(obj)).decode("utf-8") + text_postfix)

        def _read_one(interesting_rank: int):
            p = _get_path(interesting_rank)
            while True:
                if p.exists() and (text := p.read_text()).endswith(text_postfix):
                    return pickle.loads(base64.b64decode(text[:-len(text_postfix)]))
                time.sleep(0.001)

        return [_read_one(interesting_rank) for interesting_rank in range(self._world_size)]

    def barrier(self):
        actual_objs = self.all_gather_object(self._rank)
        assert actual_objs == list(range(self._world_size)), f"{actual_objs=}"

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

        # TODO
        # TODO
        # TODO temp
        # TODO
        # TODO
        # logger.info(f"cudaHostRegister({tensor.data_ptr()=})")
        # check_cuda_result(cuda_rt.cudaHostRegister(tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable))

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

def check_cuda_result(raw_output):
    err, *results = raw_output
    if err != cuda_rt.cudaError_t.cudaSuccess:
        raise Exception(f"CUDA error: {err}")
    return results

def trim_memory():
    rss_before = psutil.Process().memory_info().rss

    libc = ctypes.CDLL("libc.so.6")
    ret = libc.malloc_trim(0)

    rss_after = psutil.Process().memory_info().rss

    logger.info(f"trim_memory {ret=} {rss_after=} {rss_before=} reduced={rss_before - rss_after}")
