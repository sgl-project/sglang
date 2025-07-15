from typing import List

import torch
from torch.func import functional_call

from sglang.srt.utils import is_pin_memory_available, get_int_env_var


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
            k: v.to(device, non_blocking=True)
            for k, v in module.state_dict().items()
        }


class _ModuleOffloader:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.device = next(module.parameters()).device
        assert self.device != torch.device("cpu"), "not handled device=cpu case yet (should skip this tensor)"

        self.device_tensors = None
        _StatelessOffloaderUtil.offload(module)

    def start_onload(self):
        self.device_tensors = _StatelessOffloaderUtil.create_onload_tensors(self.module, self.device)

    def offload(self):
        self.device_tensors = None


def _hook_module_forward(module, on_forward_start, on_forward_end):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        parameter_and_buffer_dicts = on_forward_start()
        output = functional_call(module, parameter_and_buffer_dicts, args=args, kwargs=kwargs)
        on_forward_end()
        module.forward = forward
        return output

    module.forward = forward


def wrap_layers_for_offload(layers: List[torch.nn.Module]):
    layer_interval = get_int_env_var("SGLANG_OFFLOAD_LAYER_INTERVAL", 5)
    offloading_layers = layers[layer_interval - 1::layer_interval]
    offloaders = [_ModuleOffloader(layer) for layer in offloading_layers]
    TODO
    return layers
