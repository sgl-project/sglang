import torch.distributed._symmetric_memory as symm_mem
import torch.distributed as dist
import logging
import os
from abc import ABC
from typing import Callable, Generator, List

import torch
from torch.func import functional_call

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.utils import get_int_env_var, is_pin_memory_available

logger = logging.getLogger(__name__)


# TODO improve
class ModuleOffloader:
    def __init__(self):
        self.group_size = get_int_env_var("SGLANG_OFFLOAD_GROUP_SIZE", -1)
        self.num_offload_in_group = get_int_env_var("SGLANG_OFFLOAD_NUM_OFFLOAD_IN_GROUP", 1)
        self.prefetch_step = get_int_env_var("SGLANG_OFFLOAD_PREFETCH_STEP", 1)
        self.mode = os.environ.get("SGLANG_OFFLOAD_MODE", "cpu")
        self.enabled = self.group_size > 0
        assert self.mode in ["cpu", "sharded_gpu"]

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Callable[[torch.nn.Module], torch.nn.Module],
        whitelist_param_names_creator: Callable[[torch.nn.Module], List[str]],
    ):
        if not self.enabled:
            return list(all_modules_generator)

        logger.info(f"offload_module {self.group_size=} {self.num_offload_in_group=} {self.prefetch_step=}")

        alt_stream = torch.cuda.Stream()

        # TODO maybe improve
        all_modules = []
        offload_submodules = []
        self.offloaders = []
        for module_index, module in enumerate(all_modules_generator):
            logger.info(
                f"[offload_modules] {module_index=} {torch.cuda.memory_allocated()=}"
            )
            all_modules.append(module)
            if module_index % self.group_size >= self.group_size - self.num_offload_in_group:
                submodule = submodule_accessor(module)
                whitelist_param_names = whitelist_param_names_creator(submodule)
                logger.info(
                    f"[offload_modules] offload {module_index=} submodule={type(submodule)} params={whitelist_param_names}"
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
            name: _BaseParamOffloader.create(mode, param=param_dict[name])
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
            "cpu": _CpuParamOffloader,
            "sharded_gpu": _ShardedGpuParamOffloader,
        }[mode](**kwargs)

    def __init__(self, param):
        self._param = param

    def post_init(self):
        pass

    def create_device_tensor(self):
        raise NotImplementedError

class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, param):
        super().__init__(param)
        _move_param_to_cpu(param)

    def create_device_tensor(self):
        return self._param.to("cuda", non_blocking=True)

class _ShardedGpuParamOffloader(_BaseParamOffloader):
    def __init__(self, param):
        super().__init__(param)
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"
        assert param.data.is_contiguous(), f"not yet support non-contiguous tensor {param.shape=} {param.stride()=}"

        if self._rank == 0:
            _move_param_to_cpu(param)
        else:
            _move_param_to_meta(param)

        self.sharded_param_handle = None

    def post_init(self):
        # check again since it may be changed
        assert self._param.data.is_contiguous(), f"not yet support non-contiguous tensor {self._param.shape=} {self._param.stride()=}"

        scatter_src = self._param.data
        if self._rank == 0:
            scatter_src = scatter_src.to("cuda")
        else:
            assert scatter_src.device.type == "meta", f"{scatter_src.device.type=}"
        scatter_list = _even_chunk(scatter_src, self._world_size)

        sharded_param = symm_mem.empty(scatter_list[0].shape, dtype=scatter_list[0].dtype, device="cuda")
        handle = symm_mem.rendezvous(sharded_param, dist.group.WORLD)

        dist.scatter(sharded_param, scatter_list if self._rank == 0 else None, src=0)

        self.sharded_param_handle = handle

        _move_param_to_meta(self._param)

    def create_device_tensor(self):
        output = _empty_strided_like(self._param, device="cuda")
        output_chunks = output.chunk(self._world_size)

        for index in range(self._world_size):
            src_rank = (self._rank + index) % self._world_size
            src_buf = symm_mem.get_buffer(src_rank, output.shape, output.dtype)
            output_chunks[src_rank].copy_(src_buf)

        return output

def _even_chunk(x: torch.Tensor, chunks: int):
    assert x.shape[0] % chunks == 0, f"{x.shape=} {chunks=}"
    return list(x.chunk(chunks))

def _move_param_to_cpu(param):
    cpu_data = _empty_strided_like(
        param.data,
        device="cpu",
        pin_memory=is_pin_memory_available(),
    )
    cpu_data.copy_(param.data)
    param.data = cpu_data

def _move_param_to_meta(param):
    def _print(name, x):
        print(f"{name=} {x.device=} {x.shape=} {x.stride()=} {x.dtype=}")
    _print("move_param_to_meta-src", param.data)
    _print("move_param_to_meta-dst", param.data.to("meta"))

    param.data = param.data.to("meta")

def _empty_strided_like(x: torch.Tensor, device, pin_memory=False):
    return torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        layout=x.layout,
        device=device,
        pin_memory=pin_memory,
    )

