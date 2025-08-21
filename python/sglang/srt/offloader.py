import logging
from abc import ABC
from typing import Callable, Generator, List, Optional

import torch
from torch.func import functional_call

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_pin_memory_available

logger = logging.getLogger(__name__)

_SubmoduleAccessor = Callable[[torch.nn.Module], torch.nn.Module]
_WhitelistParamNamesCreator = Callable[[torch.nn.Module], List[str]]


class BaseOffloader(ABC):
    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return list(all_modules_generator)

    def post_init(self):
        pass


class NoopOffloader(BaseOffloader):
    pass


# For simplicity use singleton, but can surely support multi instance
_instance: Optional[BaseOffloader] = NoopOffloader()


def get_offloader():
    assert _instance is not None
    return _instance


def set_offloader(instance: BaseOffloader):
    global _instance
    _instance = instance


def create_offloader_from_server_args(server_args: ServerArgs):
    if server_args.cpu_offload_gb > 0:
        return OffloaderV1(
            cpu_offload_max_bytes=int(server_args.cpu_offload_gb * 1024**3)
        )
    return NoopOffloader()


class OffloaderV1(BaseOffloader):
    def __init__(self, cpu_offload_max_bytes: int):
        self._cpu_offload_bytes = 0
        self._cpu_offload_max_bytes = cpu_offload_max_bytes

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        return [self.maybe_offload_to_cpu(module) for module in all_modules_generator]

    def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
            return module

        pin_memory = is_pin_memory_available()
        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for p in module.parameters():
            if self._cpu_offload_bytes >= self._cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            # `torch.empty_like` does not support `pin_memory` argument
            cpu_data = torch.empty_strided(
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)
            p.data = cpu_data
            self._cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device, it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }
                output = functional_call(module, device_state, args=args, kwargs=kwargs)
                module.forward = forward
                return output

            module.forward = forward

        return module
