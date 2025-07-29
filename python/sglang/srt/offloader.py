import base64
import ctypes
import gc
import logging
import os
import pickle
import time
from abc import ABC
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional

import numpy as np
import psutil
import torch
from torch.func import functional_call

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import (
    MultiprocessingSerializer,
    dispose_tensor,
    get_bool_env_var,
    get_int_env_var,
    is_pin_memory_available,
)

logger = logging.getLogger(__name__)


# TODO improve
class ModuleOffloader:
    def __init__(self):
        self.group_size = get_int_env_var("SGLANG_OFFLOAD_GROUP_SIZE", -1)
        self.num_offload_in_group = get_int_env_var(
            "SGLANG_OFFLOAD_NUM_OFFLOAD_IN_GROUP", 1
        )
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

        logger.info(
            f"[offloader] {self.group_size=} {self.num_offload_in_group=} {self.prefetch_step=}"
        )

        alt_stream = torch.cuda.Stream()

        # TODO maybe improve
        all_modules = []
        offload_submodules = []
        self.offloaders = []
        for module_index, module in enumerate(all_modules_generator):
            logger.info(f"[offloader] {module_index=} {torch.cuda.memory_allocated()=}")
            all_modules.append(module)
            if (
                module_index % self.group_size
                >= self.group_size - self.num_offload_in_group
            ):
                submodule = submodule_accessor(module)
                whitelist_param_names = whitelist_param_names_creator(submodule)
                logger.info(
                    f"[offloader] offload {module_index=} submodule={type(submodule)} params={whitelist_param_names}"
                )
                offload_submodules.append(submodule)
                self.offloaders.append(
                    _ModuleOffloader(
                        mode=self.mode,
                        module=submodule,
                        alt_stream=alt_stream,
                        whitelist_param_names=whitelist_param_names,
                    )
                )

        for index, module in enumerate(offload_submodules):
            _hook_module_forward_for_offloader(
                index=index,
                module=module,
                offloaders=self.offloaders,
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


def _hook_module_forward_raw(module, on_forward_end, get_parameter_and_buffer_dicts):
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
    def __init__(
        self,
        mode: str,
        module: torch.nn.Module,
        alt_stream: torch.cuda.Stream,
        whitelist_param_names: List[str],
    ):
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
        assert all(
            name in param_dict for name in whitelist_param_names
        ), f"{whitelist_param_names=} {list(param_dict.keys())=}"

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
        return {k: v.create_device_tensor() for k, v in self._param_offloaders.items()}


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
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        self.shm_cpu_data = _shared_memory_manager.malloc(
            shape=self._param.shape, dtype=self._param.dtype
        )

        if self._rank == 0:
            self.shm_cpu_data.copy_(self._param.data.to("cpu"))
            self._param.data = self.shm_cpu_data
        else:
            _move_param_to_meta(self._module, self._param_name)
        NaiveDistributed.instance.barrier()

    def post_init(self):
        if self._rank == 0:
            assert (
                self.shm_cpu_data.data_ptr() == self._param.data.data_ptr()
            ), f"{self.shm_cpu_data.data_ptr()=} {self._param.data.data_ptr()=} {self.shm_cpu_data=} {self._param.data=}"

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
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        if self._rank == 0:
            _move_param_to_cpu(self._param, pin_memory=True)
        else:
            _move_param_to_meta(self._module, self._param_name)

        self.sharded_param_handles = None

    def post_init(self):
        # check again since it may be changed
        assert (
            self._param.data.is_contiguous()
        ), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        scatter_src = self._param.data

        logger.info(
            f"[offloader] post_init {scatter_src.nbytes=} {scatter_src.dtype=} {scatter_src.shape=} {torch.cuda.memory_allocated()=}"
        )

        if self._rank == 0:
            scatter_src = scatter_src.to("cuda")
        scatter_list = _even_chunk(scatter_src, self._world_size)

        sharded_param = torch.empty(
            scatter_list[0].shape, dtype=scatter_list[0].dtype, device="cuda"
        )
        self.sharded_param_handles = _create_shared_buffer_tensors(
            local_tensor=sharded_param
        )

        NaiveDistributed.instance.scatter(
            sharded_param, scatter_list if self._rank == 0 else None
        )

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
            **{
                k: getattr(old_param, k)
                for k in ["input_dim", "output_dim", "weight_loader"]
            },
        )
    elif old_param_type == torch.nn.Parameter:
        new_param = torch.nn.Parameter(
            data=new_data,
            requires_grad=False,
        )
    else:
        raise ValueError(f"Unknown {old_param_type=} {old_param=}")

    setattr(module, param_name, new_param)

    # logger.info(f"hi move_param_to_meta {old_param.device=} {old_param.dtype=} {old_param.shape=}")
    # dispose_tensor(old_param)
    #
    # # TODO do not call it *per* param
    # gc.collect()
    # trim_memory()


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
                (
                    None
                    if interesting_rank == self_rank
                    else MultiprocessingSerializer.serialize(local_tensor)
                )
                for interesting_rank in range(world_size)
            ]
        )
    )

    output_tensors = []
    for output_rank in range(world_size):
        remote_serialized_tensor = object_list[output_rank][
            "dup_serialized_local_tensor"
        ][self_rank]
        if output_rank == self_rank:
            assert remote_serialized_tensor is None
            output_tensors.append(local_tensor)
        else:
            output_tensors.append(
                MultiprocessingSerializer.deserialize(remote_serialized_tensor)
            )

    return output_tensors


def trim_memory():
    rss_before = psutil.Process().memory_info().rss

    libc = ctypes.CDLL("libc.so.6")
    ret = libc.malloc_trim(0)

    rss_after = psutil.Process().memory_info().rss

    logger.info(
        f"trim_memory {ret=} {rss_after=} {rss_before=} reduced={rss_before - rss_after}"
    )
