from typing import List

import torch
from torch.func import functional_call

from sglang.srt.utils import is_pin_memory_available


class _ModuleOffloader:
    def __init__(self, module: torch.nn.Module):
        device = next(module.parameters()).device
        TODO_if_cpu_then_do_not_offload
        assert device != torch.device("cpu")

        _offload_module_to_cpu(module)

        def _create_parameter_and_buffer_dicts():
            return {
                # here we blindly call `to(device)`
                # if the parameter is already on the device, it will be a no-op
                k: v.to(device, non_blocking=True)
                for k, v in module.state_dict().items()
            }

        _hook_module_forward(module, _create_parameter_and_buffer_dicts)


def _offload_module_to_cpu(module):
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


def _hook_module_forward(module, create_parameter_and_buffer_dicts):
    original_forward = module.forward

    def forward(*args, **kwargs):
        module.forward = original_forward
        output = functional_call(module, create_parameter_and_buffer_dicts(), args=args, kwargs=kwargs)
        module.forward = forward
        return output

    module.forward = forward


def wrap_layers_for_offload(layers: List[torch.nn.Module]):
    TODO_is_the_offload_too_early_now
    offloaders = [
        _ModuleOffloader(layer) if TODO else None
        for layer_id, layer in enumerate(layers)
    ]
    TODO
    return layers
