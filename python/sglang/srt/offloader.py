from typing import List

import torch
from torch.func import functional_call

from sglang.srt.utils import get_int_env_var, is_pin_memory_available


def wrap_modules_for_offload(all_modules: List[torch.nn.Module]):
    module_interval = get_int_env_var("SGLANG_OFFLOAD_MODULE_INTERVAL", 5)

    alt_stream = torch.cuda.Stream()
    offload_modules = all_modules[module_interval - 1:: module_interval]
    offloaders = [_ModuleOffloader(layer, alt_stream) for layer in offload_modules]

    offloaders[0].start_onload()

    for index, module in enumerate(offload_modules):
        _hook_module_forward_for_offloader(
            index=index, module=module, offloaders=offloaders
        )

    return all_modules


def _hook_module_forward_for_offloader(index, module, offloaders):
    _hook_module_forward_raw(
        module,
        on_forward_start=lambda: offloaders[
            (index + 1) % len(offloaders)
            ].start_onload(),
        on_forward_end=lambda: offloaders[index].offload(),
        get_parameter_and_buffer_dicts=lambda: offloaders[index].wait_and_get_device_tensors(),
    )


def _hook_module_forward_raw(
    module, on_forward_start, on_forward_end, get_parameter_and_buffer_dicts
):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        on_forward_start()
        output = functional_call(
            module, get_parameter_and_buffer_dicts(), args=args, kwargs=kwargs
        )
        on_forward_end()
        module.forward = forward
        return output

    module.forward = forward


class _ModuleOffloader:
    def __init__(self, module: torch.nn.Module, alt_stream: torch.cuda.Stream):
        self.module = module
        self.device = next(module.parameters()).device
        self.alt_stream = alt_stream
        assert self.device != torch.device(
            "cpu"
        ), "not handled device=cpu case yet (should skip this tensor)"

        self._device_tensors = None
        _StatelessOffloaderUtil.offload(module)

    def start_onload(self):
        self.alt_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.alt_stream):
            self._device_tensors = _StatelessOffloaderUtil.create_onload_tensors(
                self.module, self.device
            )

    def offload(self):
        self._device_tensors = None

    def wait_and_get_device_tensors(self):
        assert self._device_tensors is not None
        torch.cuda.current_stream().wait_stream(self.alt_stream)
        return self._device_tensors


class _StatelessOffloaderUtil:
    @staticmethod
    def offload(module):
        pin_memory = is_pin_memory_available()
        for p in module.parameters():
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

    @staticmethod
    def create_onload_tensors(module, device):
        TODO_maybe_stream
        return {
            k: v.to(device, non_blocking=True) for k, v in module.state_dict().items()
        }
