from typing import List

import torch
from torch.func import functional_call

from sglang.srt.utils import is_pin_memory_available


class _ModuleOffloader:
    def __init__(self, module: torch.nn.Module):
        device = next(module.parameters()).device
        TODO_if_cpu_then_do_not_offload
        assert device != torch.device("cpu")

        pin_memory = is_pin_memory_available()
        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        for p in module.parameters():
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


def wrap_layers_for_offload(layers: List[torch.nn.Module]):
    TODO
    return layers
