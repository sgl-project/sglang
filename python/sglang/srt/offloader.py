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

    def offload_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Callable[[torch.nn.Module], torch.nn.Module],
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
                logger.info(
                    f"[offload_modules] move {module_index=} submodule={type(submodule)} to cpu"
                )
                offload_submodules.append(submodule)
                self.offloaders.append(_ModuleOffloader(mode=self.mode, module=submodule, alt_stream=alt_stream))

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


def _parse_config():
    # TODO rename env var
    raw = os.environ.get("SGLANG_OFFLOAD_MODULE_INTERVAL")
    if raw is None:
        return None, None

    if "/" not in raw:
        raw = f"{raw}/1"

    group_size, num_offload_in_group = raw.split("/")
    return int(group_size), int(num_offload_in_group)


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
    def __init__(self, mode: str, module: torch.nn.Module, alt_stream: torch.cuda.Stream):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream

        assert self.device != torch.device(
            "cpu"
        ), "not handled device=cpu case yet (should skip this tensor)"

        self._device_tensors = None
        self._load_event = None

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, param=param)
            for name, param in self.module.named_parameters()
        }

    def post_init(self):
        pass

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
        raise NotImplementedError


class _BaseParamOffloader(ABC):
    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        return {
            "cpu": _CpuParamOffloader,
            "sharded_gpu": _ShardedGpuParamOffloader,
        }[mode](**kwargs)

    def __init__(self, param):
        self._param = param


class _CpuParamOffloader(_BaseParamOffloader):
    def __init__(self, param):
        super().__init__(param)
        _StatelessOffloaderUtil.move_param_to_cpu(param)

    def _create_device_tensors(self):
        return _StatelessOffloaderUtil.create_device_tensors(self.module, self.device)

class _ShardedGpuParamOffloader(_BaseParamOffloader):
    def __init__(self, module: torch.nn.Module, alt_stream: torch.cuda.Stream):
        super().__init__(module, alt_stream)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"

        for name, param in self.module.named_parameters():
            assert param.data.contiguous(), f"not yet support non-contiguous tensor {name=} {param.shape=} {param.stride()=}"

        if self.rank == 0:
            _StatelessOffloaderUtil.move_params_to_cpu(module)
        else:
            _StatelessOffloaderUtil.move_params_to_meta(module)

        self.sharded_named_param_handles = {}

    def post_init(self):
        for name, param in self.module.named_parameters():
            scatter_list = param.data.chunk(self.world_size)

            sharded_param = symm_mem.empty(size=scatter_list[0].shape, dtype=scatter_list[0].dtype, device="cuda")
            handle = symm_mem.rendezvous(sharded_param, dist.group.WORLD)

            dist.scatter(sharded_param, scatter_list if self.rank == 0 else None, src=0)

            self.sharded_named_param_handles[name] = handle

        _StatelessOffloaderUtil.move_params_to_meta(self.module)

    def _create_device_tensors(self):
        output_params = {}
        for name, meta_param in self.module.named_parameters():
            output_param = _empty_strided_like(meta_param, device="cuda")
            output_param_chunks = output_param.chunk(self.world_size)

            for index in range(self.world_size):
                src_rank = (self.rank + index) % self.world_size
                src_buf = symm_mem.get_buffer(src_rank, output_param.shape, output_param.dtype)
                output_param_chunks[src_rank].copy_(src_buf)

            output_params[name] = output_param
        return output_params



class _StatelessOffloaderUtil:
    @staticmethod
    def move_param_to_cpu(param):
        cpu_data = _empty_strided_like(
            param.data,
            device="cpu",
            pin_memory=is_pin_memory_available(),
        )
        cpu_data.copy_(param.data)
        param.data = cpu_data

    @staticmethod
    def move_params_to_meta(module):
        for name, param in module.named_parameters():
            param.data = param.data.to("meta")

    @staticmethod
    def create_device_tensors(module, device):
        return {
            k: v.to(device, non_blocking=True) for k, v in module.state_dict().items()
        }


def _empty_strided_like(x: torch.Tensor, device, pin_memory=False):
    return torch.empty_strided(
        size=x.size(),
        stride=x.stride(),
        dtype=x.dtype,
        layout=x.layout,
        device=device,
        pin_memory=pin_memory,
    )

