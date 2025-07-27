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
                self.offloaders.append(_BaseModuleOffloader.create(mode=self.mode, module=submodule, alt_stream=alt_stream))

        for index, module in enumerate(offload_submodules):
            _hook_module_forward_for_offloader(
                index=index, module=module, offloaders=self.offloaders,
                prefetch_step=self.prefetch_step,
            )

        return all_modules

    def on_post_load(self):
        if not self.enabled:
            return

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


class _BaseModuleOffloader(ABC):
    @staticmethod
    def create(mode, **kwargs):
        return {
            "cpu": _CpuModuleOffloader,
            "sharded_gpu": _ShardedGpuModuleOffloader,
        }[mode](**kwargs)

    def __init__(self, module: torch.nn.Module, alt_stream: torch.cuda.Stream):
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream
        assert self.device != torch.device(
            "cpu"
        ), "not handled device=cpu case yet (should skip this tensor)"


class _CpuModuleOffloader(_BaseModuleOffloader):
    def __init__(self, module: torch.nn.Module, alt_stream: torch.cuda.Stream):
        super().__init__(module, alt_stream)
        self._device_tensors = None
        self._load_event = None
        _StatelessOffloaderUtil.move_params_to_cpu(module)

    def start_onload(self):
        self.alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.alt_stream):
            self._device_tensors = _StatelessOffloaderUtil.create_device_tensors(
                self.module, self.device
            )
            self._load_event = torch.cuda.Event()
            self._load_event.record()

    def offload(self):
        self._device_tensors = None
        self._load_event = None

    def wait_and_get_device_tensors(self):
        assert self._device_tensors is not None
        self._load_event.wait()
        return self._device_tensors

class _ShardedGpuModuleOffloader(_BaseModuleOffloader):
    def __init__(self, module: torch.nn.Module, alt_stream: torch.cuda.Stream):
        super().__init__(module, alt_stream)
        assert get_tensor_model_parallel_world_size() == 1, "not yet support tp_size!=1"

class _StatelessOffloaderUtil:
    @staticmethod
    def move_params_to_cpu(module):
        pin_memory = is_pin_memory_available()
        for name, param in module.named_parameters():
            # print(f"move_params_to_cpu {name=} {param.nbytes=}")
            cpu_data = torch.empty_strided(
                size=param.data.size(),
                stride=param.data.stride(),
                dtype=param.data.dtype,
                layout=param.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(param.data)
            param.data = cpu_data

    @staticmethod
    def create_device_tensors(module, device):
        return {
            k: v.to(device, non_blocking=True) for k, v in module.state_dict().items()
        }
