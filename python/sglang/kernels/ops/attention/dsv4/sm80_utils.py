"""Lightweight, exact opt-in gate for experimental DeepSeek-V4.1 on SM80.

Only depend on the lightweight, stdlib-only environ module from srt so model,
pool and backend policies can share the gate without runtime import cycles.
This is not an end-to-end support guarantee.
"""

import torch

from sglang.srt.environ import envs


def is_dsv41_sm80_experimental(model_type, device) -> bool:
    """Enable only for the literal env value ``1`` and the actual SM80 device."""
    if (
        envs.SGLANG_DSV41_SM80_EXPERIMENTAL.get() != "1"
        or model_type != "deepseek_v41"
        or torch.version.hip is not None
        or not torch.cuda.is_available()
    ):
        return False
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 0)
