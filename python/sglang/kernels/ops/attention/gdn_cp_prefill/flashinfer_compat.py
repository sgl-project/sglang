"""FlashInfer helpers the pinned release (0.6.17) does not ship yet."""

import functools

import torch


@functools.cache
def get_device_name(device: torch.device) -> str:
    return torch.cuda.get_device_properties(device).name
